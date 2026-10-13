# 工程中的 Continuous SE(2) 是怎么实现的

本文说明本工程 `continuous_se2` 分支的真实代码路径、数学含义、张量尺寸、训练目标和使用方式。结论先说：该实现用连续的

\[
\mathbf p=(x,y),\qquad \mathbf d=(\cos\theta,\sin\theta)
\]

表示 UAV 在局部 `2×2` 卫星图块中的平面位姿。它先从 UAV 与卫星稠密特征之间建立软对应，用加权二维 Procrustes 闭式求出连续位姿初值，再用 `affine_grid + grid_sample` 按当前位姿裁剪/旋转卫星特征，最后用共享残差网络迭代细化位置和航向。

核心实现位于：

- [`cvphr/models/continuous_se2/model.py`](../cvphr/models/continuous_se2/model.py)：模型、软对应、Procrustes、可微采样、位姿迭代；
- [`cvphr/models/continuous_se2/loss.py`](../cvphr/models/continuous_se2/loss.py)：最终位姿、中间位姿、对应和可靠性损失；
- [`cvphr/train/continuous_se2_train.py`](../cvphr/train/continuous_se2_train.py)：训练入口；
- [`cvphr/test/continuous_se2_test.py`](../cvphr/test/continuous_se2_test.py)：测试和指标导出；
- [`tests/test_continuous_se2.py`](../tests/test_continuous_se2.py)：几何求解、可微性和反向传播测试；
- [`scripts/continuous_se2_train_2d.sh`](../scripts/continuous_se2_train_2d.sh)、[`scripts/continuous_se2_test.sh`](../scripts/continuous_se2_test.sh)：运行脚本。

## 1. 输入、标签与坐标约定

数据集 `RSBlockDatasetPA_v3q` 每次返回五张图：

```text
patches[:, 0:4] = p1, p2, p3, p4 四张卫星 tile
patches[:, 4]   = UAV/target patch
```

默认输入形状是 `[B, 5, 3, 256, 256]`。四张卫星图按照以下方式拼成 `512×512` 的局部地图：

```text
p1 (00) | p2 (10)
--------+--------
p3 (01) | p4 (11)
```

拼接由 `PersistentBearing._stitch_tiles()` 完成。顺序不能改，否则位置标签与地图空间不再对应。

监督标签来自 `metadata.csv`：

| 数据字段 | 代码字段 | 含义 |
|---|---|---|
| `x_norm, y_norm` | `coords` | UAV 中心在局部地图可裁剪范围内的归一化位置，通常在 `[-1,1]` |
| `x_cosa, y_sina` | `agl_coords` | 航向单位向量 `[cos(theta), sin(theta)]` |
| `theta` | `theta` | 角度制航向；Continuous SE(2) 的训练损失不直接使用它 |

位置坐标采用图像轴约定：`x` 向右为正，`y` 向下为正。原始 `512×512` 地图上，`256×256` UAV crop 的合法中心范围是 `[128,384]`，因此像素中心和归一化位置满足：

\[
u=256+128x,\qquad v=256+128y.
\]

角度标签仍用 `[cos(theta), sin(theta)]` 表达，其中 `theta` 在元数据中为角度制；进入模型后不显式恢复角度，旋转一直保留为单位圆向量。需要角度时可用：

\[
\theta=\operatorname{atan2}(\sin\theta,\cos\theta).
\]

注意：图像坐标的 `y` 向下，而通常数学图中的 `y` 向上。判断可视旋转方向时必须结合 `grid_sample` 的“输出坐标反查输入坐标”语义，不应仅凭旋转矩阵外观判断正负号。

## 2. 总体数据流

```text
5 张输入图像 [B,5,3,256,256]
            │
            ▼
冻结的 DINOv3 ConvNeXt-Base，多尺度稠密特征
            │
            ├── Satellite domain adapter ── 4 个卫星特征
            └── UAV domain adapter       ── 1 个 UAV 特征
            │
            ▼
共享 metric MLP + reliability gate
            │
            ├── 拼接卫星特征图 [B,256,32,32]
            └── UAV 特征图      [B,256,16,16]
            │
            ▼
128 维匹配投影 + 双向 softmax 软对应
            │
            ▼
加权二维 Procrustes
            │
            ▼
初始连续位姿 (x0,y0,cosθ0,sinθ0)
            │
            ▼
按当前位姿可微采样卫星特征 → 与 UAV 特征比较 → 有界残差
            │                         ▲
            └──── 默认迭代 3 次 ──────┘
            │
            ▼
最终 (x,y,cosθ,sinθ)
```

对于默认 `256×256` 输入，DINOv3 ConvNeXt 的四级特征步长约为 `4/8/16/32`。Domain adapter 统一投影到第三级空间尺寸，因此单张图得到 `16×16` 稠密特征，四张卫星特征拼接成 `32×32`。

## 3. V1 稠密前端复用

`ContinuousSE2Bearing` 继承自 `PersistentBearing`，复用以下模块：

- DINOv3 ConvNeXt 稠密 backbone；
- 卫星域和 UAV 域各自的 `DomainFeatureAdapter`；
- 两个域共享的 metric MLP；
- 每个空间位置的可靠性预测。

四个尺度先分别用 `1×1 Conv + GroupNorm + GELU` 投影到统一通道数，再插值到第三级特征大小，求平均后用卷积融合。默认通道数是 256。

可靠性图由 adapter 预测：

\[
r=0.1+0.9\,\sigma(f),
\]

所以范围是 `[0.1,1.0]`，不会把某一位置完全清零。启用 `use_reliability_gate` 时：

\[
F_{sat}\leftarrow r_{sat}F_{sat},\qquad
F_{uav}\leftarrow r_{uav}F_{uav}.
\]

`load_v1_frontend()` 只接受 PersistentBearing V1 checkpoint 中 `backbone`、两个 domain adapter 和 shared metric MLP 前缀下形状兼容的参数；离散角度相关体、heading head 等不会加载。默认 V1 checkpoint 本身省略了被冻结的 backbone encoder，因此当前已有训练记录实际迁移的是两个 adapter 和 shared metric MLP，共 8,675,330 个参数；DINOv3 backbone 则从预训练权重单独初始化。

虽然父类构造时保留了 `num_angle_bins=36`，Continuous SE(2) 覆盖了 `forward_full()`，不会调用父类的角度枚举、旋转 query、三维离散相关体或 Top-K 解码。这里的航向来自连续 Procrustes 和连续角度 residual，不是从 36 个角度 bin 中分类得到的。

## 4. 稠密软对应

### 4.1 特征投影与余弦相似度

UAV 和卫星特征先经过共享的 `match_projection`：

```text
Conv 1×1: 256 → 128
GroupNorm
GELU
```

随后展平并做 L2 归一化：

```text
Q: [B, 16×16, 128] = [B, 256, 128]
K: [B, 32×32, 128] = [B,1024, 128]
```

匹配 logits 为：

\[
L_{ij}=\frac{\hat Q_i^\top\hat K_j}{\tau},
\]

默认温度 `tau=0.07`。点积和 softmax 强制在 FP32 中计算，以降低 AMP 下的数值风险。

### 4.2 双向约束

代码分别计算：

\[
P^{row}_{ij}=\operatorname{softmax}_{j}(L_{ij}),\qquad
P^{col}_{ij}=\operatorname{softmax}_{i}(L_{ij}).
\]

两者逐元素相乘，再对每个 UAV query 行归一化：

\[
A_{ij}=\frac{P^{row}_{ij}P^{col}_{ij}}
{\sum_j P^{row}_{ij}P^{col}_{ij}}.
\]

这会抑制“很多 UAV 点都匹配到同一个卫星点”的多对一对应，但它不是 Sinkhorn，也不保证完整的双随机矩阵。

卫星像素中心被归一化到 `[-1,1]×[-1,1]`。每个 UAV 点的连续卫星对应位置是概率期望：

\[
\hat{\mathbf y}_i=\sum_j A_{ij}\mathbf y_j.
\]

每个 query 的匹配置信度取 row-softmax 的归一化熵补：

\[
w_i=\max\left(1-\frac{H(P^{row}_{i})}{\log M},10^{-4}\right).
\]

因此，分布越尖锐，点在后续 Procrustes 中的权重越高。

## 5. 用加权 Procrustes 求连续初值

UAV 特征坐标记作 `q_i`，软匹配得到的卫星坐标记作 `y_i`。当前默认尺寸下，UAV 特征宽 16、卫星拼接特征宽 32，所以已知 crop 比例：

\[
s=\frac{16}{32}=0.5.
\]

模型假设二者满足：

\[
\mathbf y_i \approx sR(\theta)\mathbf q_i+\mathbf t,
\qquad
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta& \cos\theta
\end{bmatrix}.
\]

先归一化置信度权重并计算加权中心：

\[
\bar{\mathbf q}=\sum_iw_i\mathbf q_i,
\qquad
\bar{\mathbf y}=\sum_iw_i\mathbf y_i.
\]

加权协方差为：

\[
C=\sum_iw_i(\mathbf q_i-\bar{\mathbf q})
(\mathbf y_i-\bar{\mathbf y})^\top.
\]

二维旋转无需通用 SVD，代码直接计算：

\[
a=C_{00}+C_{11},\qquad b=C_{01}-C_{10},
\]

并归一化得到：

\[
(\cos\theta,\sin\theta)=
\frac{(a+\epsilon,b)}{\lVert(a+\epsilon,b)\rVert_2}.
\]

平移为：

\[
\mathbf t=\bar{\mathbf y}-sR(\theta)\bar{\mathbf q}.
\]

这里真正监督和输出的 `position` 不是 `t`，而是合法 crop 中心范围内的归一化中心：

\[
\mathbf p=\frac{\mathbf t}{1-s}.
\]

当 `s=0.5` 时，`p=(-1,-1)`、`(1,1)` 分别对应 crop 中心可达范围的左上和右下边界。

代码函数名是 `_solve_similarity_transform()`，但尺度 `s` 并不是待估参数，而是由特征图尺寸固定。因此本质上求的是“已知尺度的二维刚体变换”。

## 6. 可微卫星特征对齐

有了当前 `position=(x,y)` 和 `direction=(c,s)`，`_warp_satellite()` 构造传给 `affine_grid` 的矩阵。令：

\[
s_x=\frac{W_{out}}{W_{map}},\qquad
s_y=\frac{H_{out}}{H_{map}},
\]

则采样矩阵为：

\[
A=
\begin{bmatrix}
s_xc&-s_ys&(1-s_x)x\\
s_xs& s_yc&(1-s_y)y
\end{bmatrix}.
\]

`affine_grid` 生成输出 UAV 网格在输入卫星特征图中的采样位置，`grid_sample` 使用双线性插值获得与 UAV 特征同尺寸的对齐卫星特征。两处都使用 `align_corners=False`，与 `_pixel_center_grid()` 的像素中心归一化公式一致。

这个过程对位置和角度可导。单元测试会对对齐结果反向传播，并检查 `position.grad` 和 `angle.grad` 非零且有限。

## 7. 迭代位姿细化

卫星和 UAV 特征先各自投影到默认 128 维。每轮将以下四项沿通道拼接：

\[
[F_u,\ F_s,\ F_u-F_s,\ F_u\odot F_s].
\]

因此 refiner 的输入通道数为 `4×128=512`。经过两层 `3×3 Conv + GroupNorm + GELU` 和全局平均池化后，再拼入当前：

\[
(x,y,\cos\theta,\sin\theta),
\]

最后由 MLP 输出三维残差原值。残差使用 `tanh` 限幅：

\[
\Delta x,\Delta y=0.25\tanh(z_{xy}),
\]

\[
\Delta\theta=30^\circ\tanh(z_\theta).
\]

默认迭代三次，且最后一层权重和偏置初始化为 0，所以训练开始时 refiner 是恒等更新，不会立刻破坏 Procrustes 初值。

位置按地图归一化坐标直接相加：

\[
\mathbf p_{k+1}=\mathbf p_k+\Delta\mathbf p_k.
\]

旋转在单位圆上复合：

\[
\begin{bmatrix}c_{k+1}\\s_{k+1}\end{bmatrix}
=
\begin{bmatrix}
\cos\Delta\theta&-\sin\Delta\theta\\
\sin\Delta\theta& \cos\Delta\theta
\end{bmatrix}
\begin{bmatrix}c_k\\s_k\end{bmatrix},
\]

复合后再次 L2 归一化。`pose_stages` 保存 Procrustes 初值和每次细化后的结果；默认形状为 `[B,4,4]`，第二维依次是初值、迭代 1、迭代 2、迭代 3。

## 8. 损失函数

总损失是：

\[
\mathcal L=
\mathcal L_{final}
+0.5\mathcal L_{intermediate}
+0.2\mathcal L_{corr}
+0.01\mathcal L_{rel}.
\]

### 8.1 最终位姿损失

位置使用 Smooth L1：

\[
\mathcal L_{pos}=\operatorname{SmoothL1}(\hat{\mathbf p},\mathbf p^*).
\]

航向先归一化，再使用余弦距离：

\[
\mathcal L_{dir}=1-\hat{\mathbf d}^{\top}\mathbf d^*.
\]

组合权重为：

\[
\mathcal L_{final}=0.8\mathcal L_{pos}+0.2\mathcal L_{dir}.
\]

### 8.2 中间监督

除最终 stage 外，Procrustes 初值和此前每个 refiner stage 都用同样的 `0.8/0.2` 位姿损失监督。这使每一步都朝真值收敛，而不只约束最终输出。

### 8.3 对应监督

根据真值位置和方向，为每个 UAV query 坐标构造真值卫星位置：

\[
\mathbf y_i^*=(1-s)\mathbf p^*+sR(\theta^*)\mathbf q_i.
\]

仅保留仍落在 `[-1,1]^2` 内的有效点，并计算预测软对应 `matched_coords` 与 `y_i*` 的 Smooth L1。该项直接监督 dense matching，而不是只依赖最终位姿间接反传。

### 8.4 可靠性正则

卫星与 UAV 可靠性图的全局均值被拉向 0.75：

\[
\mathcal L_{rel}=(\operatorname{mean}(r_{sat})-0.75)^2
+(\operatorname{mean}(r_{uav})-0.75)^2.
\]

它主要用于避免 reliability gate 全部饱和到最低或最高值，不是真实置信度标定损失。

## 9. 输出接口

普通调用：

```python
position, direction = model(patches)
```

其中：

```text
position  : [B,2]，(x,y)
direction : [B,2]，(cosθ,sinθ)
```

调试或训练调用：

```python
output = model.forward_full(
    patches,
    return_attention=True,
    return_features=False,
)
```

主要输出包括：

| key | 形状（默认设置） | 含义 |
|---|---:|---|
| `position` | `[B,2]` | 最终归一化位置 |
| `direction` | `[B,2]` | 最终航向单位向量 |
| `pose_stages` | `[B,4,4]` | 初值和三次细化位姿 |
| `matched_coords` | `[B,256,2]` | 每个 UAV 点的软匹配卫星坐标 |
| `query_coords` | `[256,2]` | UAV 特征像素中心坐标 |
| `match_confidence` | `[B,256]` | 基于熵的匹配置信度 |
| `match_logits` | `[B,256,1024]` | UAV-to-map 相似度 |
| `match_probability` | `[B,256,1024]` | 对卫星位置的 row-softmax |
| `aligned_satellite` | `[B,128,16,16]` | 最后一轮更新前采样的卫星特征 |
| `sampling_grid` | `[B,16,16,2]` | 最后一轮更新前的采样网格 |

需要注意：`aligned_satellite` 和 `sampling_grid` 对应最后一轮 residual 计算时使用的“更新前位姿”，而 `position/direction` 是应用该 residual 后的最终位姿；二者存在一小步时间差。

## 10. 训练与测试

在 `bearing/` 目录下运行训练：

```bash
FOREGROUND=1 bash scripts/continuous_se2_train_2d.sh
```

脚本默认配置：

| 项目 | 默认值 |
|---|---:|
| backbone | `dinov3_convnext_base` |
| backbone 状态 | 冻结 |
| feature dim | 256 |
| match/refinement dim | 128 / 128 |
| refinement steps | 3 |
| micro batch / 累积步数 | 20 / 5 |
| effective batch | 100 |
| epoch | 100 |
| alignment/frontend/backbone LR | `2e-4 / 5e-5 / 5e-6` |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |

如果不设置 `FOREGROUND=1`，脚本会用 `nohup` 在后台启动。可通过环境变量覆盖配置，例如：

```bash
DEVICE_ID=1 BATCH_SIZE=8 GRAD_ACCUM_STEPS=8 \
REFINEMENT_STEPS=2 FOREGROUND=1 \
bash scripts/continuous_se2_train_2d.sh
```

测试：

```bash
CHECKPOINT=results/continuous_se2_2d/<run>/best_model.pth \
bash scripts/continuous_se2_test.sh
```

测试脚本使用训练时保存的 seed 和 `0.7/0.2/0.1` 比例重建 test split，并输出：

```text
metrics.json
test_mae.json
predictions.csv
```

位置误差先用各城市地图的米/像素比例转换为米；航向误差由两个方向向量的夹角计算，范围为 `[0°,180°]`。测试结果还同时报告 Procrustes 初值误差，因此可以直接判断三轮 refiner 是否真的带来收益。

## 11. 当前工程内已有运行记录

工程中已有一次完整的 2D 训练：

```text
results/continuous_se2_2d/
  continuous_se2_dinov3_convnext_base_d256_m128_r3_b20x5_20260923_113712/
```

该运行完成 100 epoch，最小验证总损失出现在 epoch 100：

```text
validation loss            = 0.0144383
validation position MAE    = 0.0451233（归一化坐标逐分量 MAE）
validation heading MAE     = 2.60449°
peak GPU memory            = 3.777 GiB
```

这里的 `position MAE` 不是米制 MLE，不能与测试脚本中的 `mean_location_error_m` 混用。该目录目前有训练配置、history、best/last checkpoint 和推理 benchmark，但没有 `continuous_se2_test_*` 测试输出，因此上述数字只能称为验证结果。

现有 RTX 4090 D、FP32、batch=1 benchmark 记录为平均 `14.81 ms`，约 `67.53 FPS`；这是本地单次记录，不代表跨硬件性能。

## 12. 它与严格 SE(2) Lie 群实现的区别

从任务变量看，模型确实估计二维平移和一维旋转，即 SE(2) 位姿。但代码实现不是完整的 Lie 群优化器：

1. **没有显式构造或维护 `3×3` 齐次变换矩阵。** 位姿始终存成 `(x,y,cosθ,sinθ)`。
2. **旋转更新是群复合。** 当前方向与 `Delta theta` 在单位圆上相乘。
3. **平移更新是地图坐标系中的直接加法。** `p_next = p + Delta p`，没有根据当前航向旋转 residual，也没有使用 `Exp/Log` 映射。
4. **初始化是已知尺度的加权二维刚体对齐。** 不是通用 SE(2) 非线性最小二乘。

因此更准确的描述是：**以 SE(2) 位姿为输出空间的连续、可微、迭代式跨视角对齐网络**。如果后续论文声称使用“SE(2) Lie-group refinement”，则还需要明确 residual 属于左扰动还是右扰动，并用相应的群乘法更新平移和旋转。

## 13. 当前实现的边界与注意事项

1. **单峰输出。** 软对应最终被压成一个期望坐标和一个 Procrustes 位姿，没有保留对称场景中的多模态假设或协方差。
2. **正方形假设较强。** 初始化只使用宽度比计算 `crop_scale`；loss 又通过元素数量开平方恢复边长，默认 UAV 和地图特征均为正方形。
3. **不估计尺度。** UAV 与卫星的相对尺度由特征图尺寸固定为 0.5，不能处理未知高度/FOV 引起的显著尺度变化。
4. **位置未 clamp。** 三轮 residual 后的 `position` 可以超出 `[-1,1]`；此时 `grid_sample` 用零填充越界区域。
5. **最后对齐特征与最终位姿不同步。** 如需可视化最终位姿对应的 crop，应在最后更新后再调用一次 `_warp_satellite()`。
6. **matching 内存随空间位置数乘积增长。** 默认 logits 为 `256×1024`；提高特征分辨率会使显存和计算快速增长。
7. **匹配置信度不等于校准概率。** 它只是 row attention 的熵补，未用成功/失败标签做校准。
8. **可靠性正则仅约束全局均值。** 它不能单独保证每个空间位置的 reliability 有正确语义。
9. **角度符号必须端到端核对。** 数据生成用 OpenCV 旋转、特征采样用 PyTorch 归一化网格，且图像 `y` 轴向下；修改旋转矩阵或数据生成方式时应先运行几何单测并用已知角度样本可视化。

## 14. 已有测试覆盖

`tests/test_continuous_se2.py` 当前覆盖：

- 用人造精确对应验证 Procrustes 能恢复连续位置和角度；
- 验证 `grid_sample` 对位置和角度可微；
- 验证完整 forward、组合 loss 和 backward 都为有限值，关键模块能收到梯度；
- 验证 V1 前端 checkpoint 能正确迁移。

运行方式：

```bash
cd bearing
python -m unittest tests.test_continuous_se2
```

尚未覆盖的关键情况包括：非正方形输入、越界位姿、全均匀/全错误对应、180° 对称场景、最后一轮对齐可视化一致性，以及训练后 refiner 对初值的统计增益。

## 15. 一句话总结

本工程的 Continuous SE(2) 路径可以概括为：

> 用 V1 稠密特征产生 UAV-to-satellite 双向软对应，以加权二维 Procrustes 得到连续 `(x,y,cosθ,sinθ)` 初值，再通过可微卫星特征采样和有界残差网络进行三轮位置加法与单位圆旋转复合，并用最终位姿、中间位姿、对应坐标和可靠性正则联合训练。
