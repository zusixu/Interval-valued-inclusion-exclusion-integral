# IVIE - 区间值模糊积分神经网络

## 概述

IVIE (Interval-Valued Integral Entropy) 是一个基于 PyTorch 的区间值模糊积分神经网络框架。该网络专门用于处理区间值数据，通过模糊测度和 IVIE 积分实现对不确定性数据的建模和预测。

## 最新更新 (2026年1月)

### 🚀 关键改进

1. **支持限制交互阶数 (additivity_order)**
   - 避免高阶特征组合导致的数值下溢问题
   - 特别适用于 `Algebraic_interval` 操作
   - 显著提升模型在实际数据集上的性能

2. **新增改进的损失函数**
   - `ImprovedIntervalLoss`: 包含端点MSE、区间有效性惩罚、宽度匹配
   - `HausdorffIntervalLoss`: 基于Hausdorff距离的损失函数
   - 更好地约束区间预测的合理性

3. **优化的训练策略**
   - 学习率调度 (CosineAnnealingLR)
   - 早停机制 (Early Stopping)
   - 梯度裁剪防止梯度爆炸
   - AdamW优化器配合权重衰减

## 项目结构

```
IVIE/
├── __init__.py          # 模块初始化文件
├── ivie.py              # IE 网络主类 (支持 additivity_order)
├── iv_loss.py           # 损失函数模块 (新增改进的损失函数)
├── narray_op.py         # 区间运算操作模块
├── feature_layer.py     # 特征矩阵构建模块
└── README.md            # 本文档
```

---

## 核心模块介绍

### 1. IE 网络主类 (`ivie.py`)

`IE` 类是整个框架的核心，继承自 `torch.nn.Module`，实现了区间值模糊积分神经网络。

#### 类定义

```python
class IE(nn.Module):
    def __init__(self, feature_size, additivity_order=None, 
                 op='Algebraic_interval', alpha=1, beta=0, device='cuda')
```

#### 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `feature_size` | int | 必填 | 输入特征的数量 |
| `additivity_order` | int | None | **可加性阶数**，控制特征交互的最大阶数。若为 None，则等于 feature_size。<br>⚠️ **重要**: 使用 `Algebraic_interval` 时建议设置为 2-3，避免数值下溢 |
| `op` | str | 'Algebraic_interval' | 区间运算类型，可选 `'Algebraic_interval'` 或 `'Min_interval'` |
| `alpha` | float | 1 | Min_interval 操作的 alpha 参数，用于区间比较 |
| `beta` | float | 0 | Min_interval 操作的 beta 参数，用于平局时的决策 |
| `device` | str | 'cuda' | 计算设备 ('cuda' 或 'cpu') |

#### ⚠️ 重要提示：additivity_order 参数

当使用 `Algebraic_interval` 操作时，高阶特征组合会通过连续乘法生成。对于归一化到 [0,1] 的数据：

- **问题**: 7个特征 × 所有阶数 → 127个组合，7阶组合约为 0.3^7 ≈ 2×10⁻⁴，导致数值下溢
- **解决方案**: 设置 `additivity_order=2` 或 `3`，只考虑低阶交互
- **效果**: 
  - `additivity_order=2`: 生成 C(n,1) + C(n,2) 个特征 (例如7个特征→28个组合)
  - 避免数值问题的同时保留主要交互信息

```python
# ❌ 不推荐：会导致数值下溢
model = IE(feature_size=7, op='Algebraic_interval')  # 生成127个特征，高阶特征趋近于0

# ✅ 推荐：限制交互阶数
model = IE(feature_size=7, additivity_order=2, op='Algebraic_interval')  # 生成28个特征
```

#### 网络结构

```
输入层 (区间值数据)
    │
    ▼
┌─────────────────────────────────────┐
│     输入解析                         │
│  x = [x_l, x_u]                     │
│  分离左端点 datal 和右端点 datau      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     区间运算层 (narray_op)           │
│  - Algebraic_interval: 区间乘法      │
│  - Min_interval: 区间最小值选择      │
│  生成所有特征组合的区间值             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     特征矩阵层 (feature_matrix)      │
│  稀疏 01 矩阵变换                    │
│  形状: (2^n-1, 2*(2^n-1))           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     模糊测度层 (FM)                  │
│  可学习参数 vars: (2^n-2, 1)        │
│  通过 ivie_nn_vars 转换为 FM         │
│  保证单调性约束                      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     IVIE 积分计算                    │
│  区间减法: left = min(a-c,b-d)       │
│           right = b - d             │
└─────────────────────────────────────┘
    │
    ▼
输出层 (预测区间 [left, right])
```

#### 关键方法

##### `forward(x)`
前向传播方法，接收区间值输入并返回预测区间。

- **输入**: `x` - 形状为 `(batch, 2*n_features)` 的张量，前半部分是左端点，后半部分是右端点
- **输出**: `(left, right)` - 预测区间的左端点和右端点，形状均为 `(batch, 1)`

##### `ivie_nn_vars(ivie_vars)`
将神经网络参数转换为满足单调性约束的模糊测度 (Fuzzy Measure)。

- 确保 FM 值非负（通过取绝对值）
- 确保单调性：对于子集关系 $A \subseteq B$，有 $\mu(A) \leq \mu(B)$
- 归一化：$\mu(\emptyset) = 0$，$\mu(X) = 1$

##### `fit_and_valid(train_Loader, test_Loader, criterion, optimizer, device, epochs)`
训练和验证方法。

- **参数**:
  - `train_Loader`: 训练数据加载器
  - `test_Loader`: 测试数据加载器
  - `criterion`: 损失函数
  - `optimizer`: 优化器
  - `device`: 计算设备
  - `epochs`: 训练轮数

---

### 2. 区间运算模块 (`narray_op.py`)

该模块实现了两种区间运算操作，用于计算特征的所有可能组合。

#### 2.1 Algebraic_interval 类

**区间代数乘法**运算，用于计算特征组合的乘积。

$$[a, b] \times [c, d] = [a \cdot c, b \cdot d]$$

```python
class Algebraic_interval(nn.Module):
    def __init__(self, add)
    def forward(self, xl, xu) -> (nodes_tnorml, nodes_tnormu)
```

- **add**: 可加性阶数，控制组合的最大长度
- **输入**: `xl` (左端点), `xu` (右端点)，形状为 `(batch, n_features)`
- **输出**: 所有组合的区间值，按位编码顺序排列

#### 2.2 Min_interval 类

**区间最小值选择**运算，基于 alpha-beta 参数选择较小的区间。

```python
class Min_interval(nn.Module):
    def __init__(self, add, alpha, beta)
    def forward(self, xl, xu) -> (nodes_tnorml, nodes_tnormu)
```

**选择规则**:
1. 计算代表值：$v = (1-\alpha) \cdot l + \alpha \cdot u$
2. 选择代表值较小的区间
3. 若相等，使用 beta 参数进行决策

---

### 3. 特征矩阵模块 (`feature_layer.py`)

#### FeatureMatrix 类

构建用于 Choquet 积分计算的稀疏 01 矩阵。

```python
class FeatureMatrix:
    def __init__(self, n: int, device: str = 'cpu')
    def build_sparse_matrix(self) -> torch.Tensor
```

**数学原理**:
- 超集表示: $T = S \cup E$, 其中 $E \subseteq \bar{S}$
- 差集大小: $|T \setminus S| = |E| = \text{popcount}(e)$
- 子集枚举: $e_{k+1} = (e_k - 1) \land \text{complement}$

**矩阵属性**:
- 形状: $(2^n - 1, 2 \times (2^n - 1))$
- 使用稀疏 COO 格式存储
- 非零元素数量约为 $3^n - 2^n$

---

## 损失函数模块 (`iv_loss.py`)

模块提供了三种损失函数，适用于不同的训练需求。

### 1. interval_loss (原始损失函数)

基于Hausdorff距离的简单损失函数。

```python
class interval_loss(nn.Module):
    def forward(self, rel, reu, ta) -> (loss, distance)
```

**计算公式**:
$$\text{loss} = \mathbb{E}\left[\left(\frac{1}{2}\sqrt{(r_l - t_l)^2 + (r_u - t_u)^2}\right)^2\right]$$

### 2. ImprovedIntervalLoss (推荐)

改进的损失函数，包含多个约束项。

```python
class ImprovedIntervalLoss(nn.Module):
    def __init__(self, validity_weight=0.1, width_weight=0.05)
    def forward(self, rel, reu, ta) -> (total_loss, distance)
```

**损失组成**:
1. **端点MSE损失**: $L_{MSE} = \mathbb{E}[(r_l - t_l)^2] + \mathbb{E}[(r_u - t_u)^2]$
2. **区间有效性损失**: $L_{valid} = \mathbb{E}[\max(0, r_l - r_u)]$ (惩罚无效区间)
3. **宽度匹配损失**: $L_{width} = \mathbb{E}[((r_u - r_l) - (t_u - t_l))^2]$

**总损失**:
$$L_{total} = L_{MSE} + w_v \cdot L_{valid} + w_w \cdot L_{width}$$

**优势**:
- ✅ 确保预测区间有效性 (下界 ≤ 上界)
- ✅ 匹配区间宽度，避免过大或过小的预测
- ✅ 更好的数值稳定性

### 3. HausdorffIntervalLoss

基于Hausdorff距离的改进版本。

```python
class HausdorffIntervalLoss(nn.Module):
    def __init__(self, validity_weight=0.1)
    def forward(self, rel, reu, ta) -> (total_loss, hausdorff)
```

**计算公式**:
$$d_H([r_l, r_u], [t_l, t_u]) = \max(|r_l - t_l|, |r_u - t_u|)$$

---


## 使用示例

### 基本使用

```python
import torch
from IVIE.ivie import IE

# 创建模型 (推荐配置)
model = IE(
    feature_size=7,            # 7 个特征
    additivity_order=2,        # 只考虑2阶交互，避免数值问题
    op='Algebraic_interval',   # 使用代数区间运算
    alpha=0.5,
    beta=0,
    device='cuda'
)
model = model.to('cuda')

# 准备区间值输入 [左端点们, 右端点们]
# 形状: (batch_size, 2 * n_features)
x = torch.rand(32, 14).to('cuda')  # 32 个样本，7 个特征的区间值

# 前向传播
pred_left, pred_right = model(x)
print(f"预测区间形状: {pred_left.shape}, {pred_right.shape}")  # (32, 1), (32, 1)
```

### 训练示例 (改进版配置)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from IVIE.ivie import IE
from IVIE.iv_loss import ImprovedIntervalLoss

# 准备数据
X_train = torch.rand(100, 14)  # 100 个样本，7 个特征的区间值
y_train = torch.rand(100, 2)   # 区间值标签 [下界, 上界]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

# 创建模型 (推荐配置)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = IE(
    feature_size=7,
    additivity_order=2,         # 限制交互阶数
    op='Algebraic_interval',
    device=device
).to(device)

# 使用改进的损失函数
criterion = ImprovedIntervalLoss(validity_weight=0.1, width_weight=0.05)

# 使用 AdamW 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 训练循环 (带早停和学习率调度)
best_val_loss = float('inf')
patience = 30
patience_counter = 0

for epoch in range(300):
    # 训练阶段
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        pred_l, pred_u = model(images)
        loss, _ = criterion(pred_l, pred_u, labels)
        train_loss += loss.item() * len(labels)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred_l, pred_u = model(images)
            loss, _ = criterion(pred_l, pred_u, labels)
            val_loss += loss.item() * len(labels)
    
    avg_val_loss = val_loss / len(test_loader.dataset)
    
    # 学习率调度
    scheduler.step()
    
    # 早停检查
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"早停触发于 epoch {epoch + 1}")
        model.load_state_dict(best_model_state)
        break
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/300], train_loss: {avg_train_loss:.6f}, '
              f'val_loss: {avg_val_loss:.6f}, lr: {optimizer.param_groups[0]["lr"]:.6f}')

print(f"最佳验证损失: {best_val_loss:.6f}")
```

### 快速训练 (使用内置方法)

```python
# 使用模型内置的训练方法
from IVIE.iv_loss import interval_loss as IntervalLoss

criterion = IntervalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.fit_and_valid(
    train_Loader=train_loader,
    test_Loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=100
)
```

### 数据预处理建议

对于UCI数据集等实际应用，推荐使用以下预处理方式构造区间值：

```python
import pandas as pd
import numpy as np

# 假设 df 是归一化后的特征数据 (值在 [0, 1])
spread_ratio = 0.1  # 使用10%的区间宽度

# 构造区间值
data_low = (df * (1 - spread_ratio)).clip(lower=0)
data_up = (df * (1 + spread_ratio)).clip(upper=1)

# ❌ 不推荐：使用标准差可能产生负值
# data_low = df - 2 * df.std()  # 可能 < 0
# data_up = df + 2 * df.std()   # 可能 > 1
```

---

## 常见问题与最佳实践

### ❓ 为什么预测值全是0或接近0？

**原因**: 使用 `Algebraic_interval` 且未限制 `additivity_order` 时，高阶乘法导致数值下溢。

**解决方案**:
```python
# ✅ 设置 additivity_order
model = IE(feature_size=7, additivity_order=2, op='Algebraic_interval')
```

### ❓ 如何选择 additivity_order？

| additivity_order | 特征组合数 (n=7) | 适用场景 |
|-----------------|-----------------|---------|
| 1 | 7 | 仅考虑单个特征，线性模型 |
| 2 | 28 | **推荐**，考虑特征对交互 |
| 3 | 63 | 考虑三元交互，计算量适中 |
| 7 (全部) | 127 | 所有交互，可能数值下溢 |

**推荐**: 从 2 开始，根据验证集性能调整到 3 或 4。

### ❓ 选择 Algebraic_interval 还是 Min_interval？

| 操作类型 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| **Algebraic_interval** | 连续可导，便于优化 | 需要限制阶数避免下溢 | 归一化数据，需要特征交互 |
| **Min_interval** | 数值稳定，无下溢问题 | 非光滑，优化可能较慢 | 原始数据，稳健性要求高 |

**建议**: 优先尝试 `Algebraic_interval` + `additivity_order=2`

### ❓ 损失函数如何选择？

| 损失函数 | 适用场景 | 权重建议 |
|---------|---------|---------|
| `interval_loss` | 基准测试，简单场景 | - |
| `ImprovedIntervalLoss` | **推荐**，大多数实际应用 | validity_weight=0.1, width_weight=0.05 |
| `HausdorffIntervalLoss` | 对区间端点误差敏感的场景 | validity_weight=0.1 |

### ❓ 学习率和优化器如何设置？

**推荐配置**:
```python
# AdamW + 余弦退火
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# 或 Adam + ReduceLROnPlateau
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
```

**学习率建议**:
- 初始学习率: 0.001 - 0.01
- 使用学习率调度器逐步衰减
- 添加梯度裁剪: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### ❓ 如何处理数据预处理？

**区间构造方式对比**:

```python
# ❌ 不推荐：标准差方法 (可能产生负值或越界)
data_low = df - 2 * df.std()  # 可能 < 0
data_up = df + 2 * df.std()   # 可能 > 1

# ✅ 推荐：比例偏移 (保证范围)
spread_ratio = 0.1
data_low = (df * (1 - spread_ratio)).clip(lower=0)
data_up = (df * (1 + spread_ratio)).clip(upper=1)

# ✅ 推荐：绝对偏移 + 裁剪
epsilon = 0.1
data_low = (df - epsilon).clip(lower=0)
data_up = (df + epsilon).clip(upper=1)
```

---

## 性能调优建议

### 1. 训练策略

✅ **使用早停**:
```python
patience = 30
best_val_loss = float('inf')
patience_counter = 0

# 在训练循环中
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    patience_counter = 0
    torch.save(model.state_dict(), 'best_model.pth')
else:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

✅ **批量大小**:
- 小数据集 (< 1000): batch_size = 16-32
- 中等数据集 (1000-10000): batch_size = 32-64
- 大数据集 (> 10000): batch_size = 64-128

✅ **训练轮数**:
- 配合早停: epochs = 300-500
- 无早停: epochs = 100-200

### 2. 模型初始化

当前模型使用均匀初始化。可以尝试改进:

```python
# 在 IE.__init__ 中修改
# 默认: dummy = (1./self.columns_num) * torch.ones((self.nVars, 1))
# 改进: 使用 Xavier 初始化
import torch.nn as nn
init_val = torch.empty((self.nVars, 1))
nn.init.xavier_uniform_(init_val)
init_val = torch.abs(init_val) * 0.5 + 0.1  # 保证正值
self.vars = torch.nn.Parameter(init_val)
```

### 3. 监控指标

建议在训练过程中监控:
- 训练损失和验证损失
- 区间有效性 (预测的下界 ≤ 上界的比例)
- 平均绝对误差 (MAE): `torch.mean(torch.abs(pred_l - true_l) + torch.abs(pred_u - true_u))`
- 区间宽度: `torch.mean(pred_u - pred_l)`

---

## 版本历史

### v1.1.0 (2026年1月)
- ✨ 新增 `additivity_order` 支持，解决高阶组合的数值下溢问题
- ✨ 新增 `ImprovedIntervalLoss` 和 `HausdorffIntervalLoss` 损失函数
- 🔧 优化 `forward` 方法，正确处理受限阶数的特征矩阵
- 🔧 优化 `ivie_nn_vars` 方法，支持受限阶数的模糊测度构建
- 📝 完善文档和使用示例
- 🎯 提升在UCI等实际数据集上的性能

### v1.0.0 (初始版本)
- 基础IE网络实现
- 支持 Algebraic_interval 和 Min_interval 操作
- 基础损失函数 interval_loss

---

**最后更新**: 2026年1月14日
