# IVIE_FM

区间值模糊积分神经网络 - FM实现版本

## 模块结构

```
IVIE_FM/
├── __init__.py          # 模块初始化
├── ivie.py              # IE网络主类
├── iv_loss.py           # 损失函数
├── narray_op.py         # 区间运算
├── feature_layer.py     # 特征组合层
└── README.md
```

## 快速使用

```python
from IVIE_FM.ivie import IE
from IVIE_FM.iv_loss import HausdorffIntervalLoss

# 创建模型
model = IE(
    feature_size=7,              # 特征数量
    additivity_order=2,          # 交互阶数 (推荐2-3)
    op='Algebraic_interval',     # 区间运算类型
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

### 1. IE 网络主类 (`ivie.py`)

**参数说明**:
- `feature_size`: 特征数量
- `additivity_order`: 特征交互最大阶数（建议2-3，避免数值下溢）
- `op`: 区间运算类型
  - `'Algebraic_interval'`: 代数积运算
  - `'Min_interval'`: 最小T-norm运算
- `alpha`, `beta`: Min_interval的排序参数
- `device`: 计算设备 ('cuda' 或 'cpu')

**输入格式**: `[x1_l, x2_l, ..., xn_l, x1_u, x2_u, ..., xn_u]`  
**输出格式**: `(下界张量, 上界张量)`

### 2. 区间运算 (`narray_op.py`)

#### Algebraic_interval - 代数积运算
- 计算所有特征组合的代数积
- 区间[a,b] × [c,d] = [ac, bd]

#### Min_interval - 最小T-norm运算
- 基于alpha-beta参数选择最小区间
- 选择规则: v = (1-α)·l + α·u

### 3. 特征矩阵 (`feature_layer.py`)

**FeatureMatrix** - 构建Choquet积分所需的稀疏01矩阵
- 形状: (2^n - 1, 2 × (2^n - 1))
- 用于特征组合的权重计算

### 4. 损失函数 (`iv_loss.py`)

#### interval_loss - 基础损失
- 基于Hausdorff距离

#### ImprovedIntervalLoss - 推荐使用
- 端点MSE损失 + 区间有效性惩罚 + 宽度匹配
- 参数: `validity_weight=0.1`, `width_weight=0.05`

#### HausdorffIntervalLoss - Hausdorff距离
- 最大端点误差距离
- 参数: `validity_weight=0.1`

## 数据流程

```
输入 [xl1,...,xln,xu1,...,xun]
    ↓
特征组合层 (narray_op)
    ↓
特征矩阵变换 (feature_layer)
    ↓
模糊测度加权 (FM层)
    ↓
IVIE积分计算
    ↓
输出 (下界, 上界)
```

## 训练示例

```python
from IVIE_FM.iv_loss import ImprovedIntervalLoss

# 配置优化器和调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 使用内置训练方法
criterion = ImprovedIntervalLoss(validity_weight=0.1, width_weight=0.05)
model.fit_and_valid(train_loader, test_loader, criterion, optimizer, epochs=100)
```

## 最佳实践

### 参数选择建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| additivity_order | 2-3 | 避免高阶特征组合导致数值下溢 |
| op | Algebraic_interval | 连续可导，便于优化 |
| learning_rate | 0.001-0.005 | 配合学习率调度器 |
| batch_size | 32-64 | 根据数据集大小调整 |



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
