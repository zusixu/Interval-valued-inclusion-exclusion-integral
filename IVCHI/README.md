# IVCHI

区间值Choquet积分神经网络 - 区间加法版本

IVCHI是IVIE_FM的派生类，将区间减法运算改为区间加法运算。

## 模块结构

```
IVCHI/
├── __init__.py          # 模块初始化
├── ivchi.py             # IVCHI网络类
└── README.md
```

## 快速使用

```python
from IVCHI.ivchi import IVCHI
from IVIE_FM.iv_loss import HausdorffIntervalLoss
import torch

# 创建模型
model = IVCHI(
    feature_size=7,              # 特征数量
    additivity_order=2,          # 交互阶数 (推荐2-3)
    op='Algebraic_interval',     # 区间运算类型
    device='cuda'
)

# 训练
criterion = HausdorffIntervalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 准备数据 (格式: [x1_l, x2_l, ..., x1_u, x2_u, ...])
X = torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.3, 0.4]])
pred_l, pred_u = model(X)
```

## 与IVIE_FM的区别

| 特性 | IVIE_FM | IVCHI |
|------|---------|-------|
| 区间运算 | 区间减法 | **区间加法** |
| 运算公式 | [a,b]-[c,d]=[min(a-c,b-d), b-d] | [a,b]+[c,d]=[a+c, b+d] |
| 适用场景 | 一般预测任务 | 需要累积效应的场景 |

## 核心组件

### IVCHI 类 (`ivchi.py`)

继承自 `IVIE_FM.ivie.IE`，重写了forward方法中的区间运算部分。

**参数说明**:
- `feature_size`: 特征数量
- `additivity_order`: 特征交互最大阶数（建议2-3）
- `op`: 区间运算类型 ('Algebraic_interval' 或 'Min_interval')
- `alpha`, `beta`: Min_interval的排序参数
- `device`: 计算设备

**输入格式**: `[x1_l, x2_l, ..., xn_l, x1_u, x2_u, ..., xn_u]`  
**输出格式**: `(下界张量, 上界张量)`

## 数据流程

```
输入 [xl1,...,xln,xu1,...,xun]
    ↓
特征组合层 (继承自IVIE_FM)
    ↓
特征矩阵变换
    ↓
模糊测度加权
    ↓
区间加法计算 (与IVIE_FM不同)
    ↓
输出 (下界, 上界)
```

## 训练示例

```python
from torch.utils.data import DataLoader, TensorDataset
from IVIE_FM.iv_loss import ImprovedIntervalLoss

# 准备数据
X_train = torch.rand(100, 14)  # 7个特征的区间值数据
y_train = torch.rand(100, 2)   # 区间值标签 [下界, 上界]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)
test_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

# 创建模型
model = IVCHI(feature_size=7, additivity_order=2, device='cuda')

# 训练
criterion = ImprovedIntervalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.fit_and_valid(train_loader, test_loader, criterion, optimizer, epochs=100)
```

## 完整示例

参见 `tests/test_ivchi.py`，包含：
- 前向传播测试
- 与IVIE_FM的对比测试
- 区间加法正确性验证
- 完整训练流程

运行测试：
```bash
python tests/test_ivchi.py
```