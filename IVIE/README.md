# IVIE - 区间值模糊积分神经网络

## 概述

IVIE (Interval-Valued Integral Entropy) 是一个基于 PyTorch 的区间值模糊积分神经网络框架。该网络专门用于处理区间值数据，通过模糊测度和 Choquet 积分实现对不确定性数据的建模和预测。

## 项目结构

```
IVIE/
├── __init__.py          # 模块初始化文件
├── ivie.py              # IE 网络主类
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
| `additivity_order` | int | None | 可加性阶数，控制特征交互的最大阶数。若为 None，则等于 feature_size |
| `op` | str | 'Algebraic_interval' | 区间运算类型，可选 `'Algebraic_interval'` 或 `'Min_interval'` |
| `alpha` | float | 1 | Min_interval 操作的 alpha 参数，用于区间比较 |
| `beta` | float | 0 | Min_interval 操作的 beta 参数，用于平局时的决策 |
| `device` | str | 'cuda' | 计算设备 ('cuda' 或 'cpu') |

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
│     Choquet 积分计算                 │
│  区间减法: left = min(a-c, b-d)     │
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

## 数学基础

### Choquet 积分

Choquet 积分是一种非线性积分，定义为：

$$C_\mu(f) = \sum_{i=1}^{n} [f_{(i)} - f_{(i-1)}] \cdot \mu(A_{(i)})$$

其中：
- $f_{(i)}$ 是排序后的第 i 个值
- $\mu$ 是模糊测度
- $A_{(i)} = \{x_{(i)}, x_{(i+1)}, ..., x_{(n)}\}$

### 区间值扩展

对于区间值数据 $[\underline{x}, \overline{x}]$，Choquet 积分扩展为：

$$C_\mu([\underline{f}, \overline{f}]) = [\underline{C}, \overline{C}]$$

输出区间的端点通过考虑所有可能的端点组合计算得出。

### 模糊测度约束

模糊测度 $\mu: 2^X \rightarrow [0, 1]$ 需满足：
1. **边界条件**: $\mu(\emptyset) = 0$, $\mu(X) = 1$
2. **单调性**: 若 $A \subseteq B$，则 $\mu(A) \leq \mu(B)$

网络通过 `ivie_nn_vars` 方法确保这些约束：
- 取绝对值保证非负
- 累加结构保证单调性
- 与 1 取最小值保证归一化

---

## 使用示例

### 基本使用

```python
import torch
from IVIE.ivie import IE

# 创建模型
model = IE(
    feature_size=3,           # 3 个特征
    op='Algebraic_interval',  # 使用代数区间运算
    alpha=1,
    beta=0,
    device='cuda'
)
model = model.to('cuda')

# 准备区间值输入 [左端点们, 右端点们]
# 形状: (batch_size, 2 * n_features)
x = torch.rand(32, 6).to('cuda')  # 32 个样本，3 个特征的区间值

# 前向传播
pred_left, pred_right = model(x)
print(f"预测区间: [{pred_left}, {pred_right}]")
```

### 训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from IVIE.ivie import IE

# 自定义区间损失函数
class IntervalLoss(nn.Module):
    def forward(self, pred_l, pred_u, target):
        target_l = target[:, 0:1]
        target_u = target[:, 1:2]
        loss_l = torch.mean((pred_l - target_l) ** 2)
        loss_u = torch.mean((pred_u - target_u) ** 2)
        loss = loss_l + loss_u
        distance = torch.abs(pred_l - target_l) + torch.abs(pred_u - target_u)
        return loss, distance

# 准备数据
X_train = torch.rand(100, 6)  # 100 个样本，3 个特征的区间值
y_train = torch.rand(100, 2)  # 区间值标签

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16)

# 创建模型和优化器
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = IE(feature_size=3, op='Algebraic_interval', device=device).to(device)
criterion = IntervalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
model.fit_and_valid(
    train_Loader=train_loader,
    test_Loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=100
)
```

---

## 依赖项

- Python >= 3.8
- PyTorch >= 1.9
- NumPy
- SciPy

---

## 参考文献

1. Choquet, G. (1954). Theory of capacities. *Annales de l'Institut Fourier*, 5, 131-295.
2. Grabisch, M. (1996). The application of fuzzy integrals in multicriteria decision making. *European Journal of Operational Research*, 89(3), 445-456.
3. Beliakov, G., Pradera, A., & Calvo, T. (2007). *Aggregation Functions: A Guide for Practitioners*. Springer.
