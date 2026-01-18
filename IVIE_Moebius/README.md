# IVIE_Moebius

区间值IE积分神经网络 - Moebius实现版本

## 模块结构

```
IVIE_Moebius/
├── __init__.py          # 模块初始化
├── ieinn.py             # IE网络主类
├── narray_op.py         # 区间运算 (Algebraic/Min)
├── order.py             # Admissible排序
├── output_layer.py      # 输出层 (模糊测度)
├── iv_loss.py           # 损失函数
└── README.md
```

## 快速使用

```python
from IVIE_Moebius.ieinn import IE
from IVIE_Moebius.iv_loss import HausdorffIntervalLoss
import torch

# 创建模型
model = IE(
    feature_size=7,
    additivity_order=2,
    op='Algebraic_interval',     # 或 'Min_interval'
    fuzzy_measure='OutputLayer_single',  # 或 'OutputLayer_interval'
    device='cuda'
)

# 训练
criterion = HausdorffIntervalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 预测
pred_l, pred_u = model(X_test)
```

## 核心组件

### 1. IE 网络主类 (`ieinn.py`)

**参数说明**:
- `feature_size`: 特征数量
- `additivity_order`: 特征交互最大阶数（默认等于feature_size）
- `op`: 区间运算类型
  - `'Algebraic_interval'`: 代数积
  - `'Min_interval'`: 最小T-norm
- `fuzzy_measure`: 模糊测度类型
  - `'OutputLayer_single'`: 单值模糊测度
  - `'OutputLayer_interval'`: 区间值模糊测度
- `alpha`, `beta`: Min_interval的排序参数
- `device`: 计算设备

**输入格式**: `[x1_l, x2_l, ..., xn_l, x1_u, x2_u, ..., xn_u]`  
**输出格式**: `(下界张量, 上界张量)`

### 2. 区间运算 (`narray_op.py`)

#### Algebraic_interval
- 计算所有特征组合的代数积
- 区间[a,b] × [c,d] = [ac, bd]

#### Min_interval
- 使用记忆化算法计算特征组合
- 基于admissible order选择最小区间

**特征组合顺序**: 从单特征 → 2特征组合 → 3特征组合 → ... → n阶组合

### 3. 排序模块 (`order.py`)

**Ordered类** - Admissible排序
- 主排序键: K_alpha = (1-α)·xl + α·xu
- 次排序键: K_beta = (1-β)·xl + β·xu
- 向量化实现，支持批量处理

### 4. 输出层 (`output_layer.py`)

#### OutputLayer_single - 单值模糊测度
```python
参数: weight[1, all_nodes], bias[1]
```

#### OutputLayer_interval - 区间值模糊测度
```python
参数: weight_left[1, all_nodes], weight_right[1, all_nodes]
     bias_left[1], bias_right[1]
```

### 5. 损失函数 (`iv_loss.py`)

**HausdorffIntervalLoss** - Hausdorff距离损失
- 距离计算: d_H([a,b], [c,d]) = max(|a-c|, |b-d|)
- 总损失 = 均方Hausdorff距离 + λ·有效性惩罚

## 数据流程

```
输入 [xl1,xl2,...,xu1,xu2,...]
    ↓ (分离下界和上界)
特征组合层 (Algebraic_interval / Min_interval)
    ↓ (生成所有特征子集组合)
输出层 (OutputLayer_single / OutputLayer_interval)
    ↓ (加权聚合)
区间值预测 [yl, yu]
```

## 网络参数统计

对于n个特征，additivity_order=k:
- **组合节点数**: Σ(i=1 to k) C(n,i)
- **OutputLayer_single参数**: 组合节点数 + 1
- **OutputLayer_interval参数**: 2×(组合节点数 + 1)

## 训练示例

```python
from torch.utils.data import DataLoader, TensorDataset

# 准备数据
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

# 训练循环
for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        pred_l, pred_u = model(X_batch)
        loss, _ = criterion(pred_l, pred_u, y_batch)
        loss.backward()
        optimizer.step()
```

