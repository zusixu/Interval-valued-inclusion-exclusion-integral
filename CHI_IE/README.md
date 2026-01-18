# CHI_IE

集成学习神经网络 - 基于多个子模型的区间值预测

## 模块结构

```
CHI_IE/
├── __init__.py          # 模块初始化
├── ensemble_ie.py       # 集成网络主类
└── README.md
```

## 网络结构

```
输入数据 (batch_size, 2*feature_size)
    ↓
    ├─→ 子模型1 (IVCHI/IVIE_Moebius) → (下界1, 上界1)
    ├─→ 子模型2 (IVCHI/IVIE_Moebius) → (下界2, 上界2)
    ├─→ 子模型3 (IVCHI/IVIE_Moebius) → (下界3, 上界3)
    └─→ ...
    ↓
拼接 (batch_size, 2*m)
    ↓
集成层 (IVIE_FM/IVIE_Moebius)
    ↓
最终输出 (下界, 上界)
```

## 快速使用

```python
from CHI_IE.ensemble_ie import EnsembleIE
from IVIE_FM.iv_loss import HausdorffIntervalLoss
import torch

# 创建模型
model = EnsembleIE(
    feature_size=7,                     # 特征维度
    num_base_models=3,                  # 子模型数量
    base_model_type='IVCHI',            # 子模型类型
    base_model_configs=[                # 每个子模型的配置
        {'additivity_order': 2, 'alpha': 0.5},
        {'additivity_order': 2, 'alpha': 0.6},
        {'additivity_order': 3, 'alpha': 0.4}
    ],
    ensemble_type='FM',                 # 集成层类型
    ensemble_config={                   # 集成层配置
        'additivity_order': 2,
        'op': 'Min_interval',
        'alpha': 0.5
    },
    device='cuda'
)

# 训练
criterion = HausdorffIntervalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.fit_and_valid(train_loader, test_loader, criterion, optimizer, epochs=100)

# 预测
pred_l, pred_u = model(X_test)
```

## 核心组件

### EnsembleIE 类 (`ensemble_ie.py`)

**参数说明**:
- `feature_size`: 输入特征数量
- `num_base_models`: 子模型数量
- `base_model_type`: 子模型类型
  - `'IVCHI'`: 使用IVCHI作为子模型
  - `'IVIE_Moebius'`: 使用IVIE_Moebius作为子模型（需要op='Min_interval'）
- `base_model_configs`: 每个子模型的配置列表（长度应等于num_base_models）
- `ensemble_type`: 集成层类型
  - `'FM'`: 使用IVIE_FM作为集成层
  - `'Moebius'`: 使用IVIE_Moebius作为集成层
- `ensemble_config`: 集成层的配置字典
- `device`: 计算设备

**输入格式**: `[x1_l, x2_l, ..., xn_l, x1_u, x2_u, ..., xn_u]`  
**输出格式**: `(下界张量, 上界张量)`

## 配置示例

### 示例1: IVCHI子模型 + FM集成

```python
model = EnsembleIE(
    feature_size=7,
    num_base_models=3,
    base_model_type='IVCHI',
    base_model_configs=[
        {'additivity_order': 2, 'op': 'Algebraic_interval', 'alpha': 0.5},
        {'additivity_order': 2, 'op': 'Algebraic_interval', 'alpha': 0.6},
        {'additivity_order': 3, 'op': 'Algebraic_interval', 'alpha': 0.4}
    ],
    ensemble_type='FM',
    ensemble_config={
        'additivity_order': 2,
        'op': 'Min_interval',
        'alpha': 0.5,
        'beta': 0.0
    },
    device='cuda'
)
```

### 示例2: IVIE_Moebius子模型 + Moebius集成

```python
model = EnsembleIE(
    feature_size=7,
    num_base_models=4,
    base_model_type='IVIE_Moebius',
    base_model_configs=[
        {'additivity_order': 2, 'op': 'Min_interval', 'alpha': 0.5, 
         'fuzzy_measure': 'OutputLayer_single'},
        {'additivity_order': 2, 'op': 'Min_interval', 'alpha': 0.6, 
         'fuzzy_measure': 'OutputLayer_single'},
        {'additivity_order': 3, 'op': 'Min_interval', 'alpha': 0.4, 
         'fuzzy_measure': 'OutputLayer_single'},
        {'additivity_order': 2, 'op': 'Min_interval', 'alpha': 0.7, 
         'fuzzy_measure': 'OutputLayer_single'}
    ],
    ensemble_type='Moebius',
    ensemble_config={
        'additivity_order': 3,
        'op': 'Algebraic_interval',
        'fuzzy_measure': 'OutputLayer_single'
    },
    device='cuda'
)
```

## 训练示例

```python
from torch.utils.data import DataLoader, TensorDataset
from IVIE_FM.iv_loss import ImprovedIntervalLoss

# 准备数据
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# 创建模型
model = EnsembleIE(
    feature_size=7,
    num_base_models=3,
    base_model_type='IVCHI',
    ensemble_type='FM',
    device='cuda'
)

# 训练
criterion = ImprovedIntervalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.fit_and_valid(
    train_Loader=train_loader,
    test_Loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100
)
```

## 集成策略

### 子模型多样性

为了提高集成效果，建议让子模型具有多样性：
- 使用不同的 `additivity_order`
- 使用不同的 `alpha` 参数
- 使用不同的区间运算类型（如果子模型支持）

### 集成层选择

| 集成层类型 | 优势 | 适用场景 |
|-----------|------|---------|
| FM | 参数效率高，训练快 | 大多数场景 |
| Moebius | 表达能力强 | 复杂任务，子模型多 |

## 最佳实践

### 参数选择建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| num_base_models | 3-5 | 过多会增加计算量 |
| base_model additivity_order | 2-3 | 避免数值下溢 |
| ensemble additivity_order | 2 | 集成层一般用低阶即可 |
| learning_rate | 0.0005-0.001 | 集成模型建议用较小学习率 |

### 常见问题

**Q: 子模型数量如何选择？**  
A: 3-5个通常足够，过多会增加计算量但性能提升有限

**Q: 所有子模型必须使用相同配置吗？**  
A: 不需要，建议使用不同配置增加多样性

**Q: 何时使用IVCHI，何时使用IVIE_Moebius作为子模型？**  
A: IVCHI计算简单，IVIE_Moebius使用Min_interval时数值更稳定

**Q: 集成层的feature_size是什么？**  
A: 自动设置为num_base_models，因为集成层输入是各子模型的输出
