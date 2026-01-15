# IVCHI 使用示例

IVCHI 是 IVIE 的派生类，将区间减法运算改为区间加法运算。

## 区间运算对比

- **IE (原始类)**: 区间减法 `[a,b]-[c,d]=[min(a-c,b-d), b-d]`
- **IVCHI (派生类)**: 区间加法 `[a,b]+[c,d]=[a+c, b+d]`

## 快速开始

```python
import torch
from IVCHI.ivchi import IVCHI
from IVIE.iv_loss import interval_loss

# 1. 创建模型
model = IVCHI(
    feature_size=3,              # 特征数量
    additivity_order=2,          # 限制交互阶数（推荐2-3）
    op='Algebraic_interval',     # 区间运算类型
    device='cpu'                 # 'cpu' 或 'cuda'
)

# 2. 准备数据
# 输入格式: [x1_l, x2_l, x3_l, x1_u, x2_u, x3_u]
# 前半部分是左端点，后半部分是右端点
X = torch.tensor([
    [0.1, 0.2, 0.3, 0.2, 0.3, 0.4],  # 样本1
    [0.5, 0.6, 0.7, 0.6, 0.7, 0.8],  # 样本2
])

# 3. 前向传播
output_l, output_u = model(X)
print(f"输出左端点: {output_l}")
print(f"输出右端点: {output_u}")

# 4. 训练（示例）
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = interval_loss()

# 假设有标签 y = [y_l, y_u]
y = torch.tensor([[0.5, 0.6], [0.7, 0.8]])

optimizer.zero_grad()
pred_l, pred_u = model(X)
loss, error = criterion(pred_l, pred_u, y)
loss.backward()
optimizer.step()
```

## 完整训练示例

参见 `tests/test_ivchi.py` 中的 `test_ivchi_training()` 函数，包含：
- 数据加载
- 训练循环
- 验证
- 损失计算

## 与IE的对比测试

运行以下测试查看IVCHI和IE的输出差异：

```bash
python tests/test_ivchi.py
```

测试包括：
1. ✓ 前向传播测试
2. ✓ 与IE的对比测试（验证区间加法vs减法）
3. ✓ 区间加法正确性测试
4. ✓ 训练功能测试

## API参数说明

### IVCHI 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `feature_size` | int | 必填 | 输入特征数量 |
| `additivity_order` | int | None | 特征交互的最大阶数，推荐2-3 |
| `op` | str | 'Algebraic_interval' | 区间运算类型：'Algebraic_interval' 或 'Min_interval' |
| `alpha` | float | 1 | Min_interval的alpha参数 |
| `beta` | float | 0 | Min_interval的beta参数 |
| `device` | str | 'cuda' | 计算设备：'cuda' 或 'cpu' |

### forward 方法

**输入**: 
- `x`: shape为`(batch_size, 2*feature_size)`的张量
- 前半部分是左端点，后半部分是右端点

**输出**:
- `output_l`: shape为`(batch_size, 1)`的左端点
- `output_u`: shape为`(batch_size, 1)`的右端点

## 数学原理

IVCHI使用区间加法而不是减法，适用于需要累积效应的场景：

```
输入: [a,b], [c,d]
IE输出:   [min(a-c,b-d), b-d]  (减法)
IVCHI输出: [a+c, b+d]          (加法)
```

这使得IVCHI更适合以下场景：
- 风险累积分析
- 正向影响叠加
- 区间值聚合
- 不确定性传播

## 注意事项

1. **特征阶数限制**: 对于较多特征（>5），建议设置`additivity_order=2`或`3`以避免数值问题
2. **数据归一化**: 输入数据建议归一化到[0,1]范围
3. **区间有效性**: 确保输入的左端点≤右端点
4. **设备选择**: 如果有GPU，使用`device='cuda'`可显著加速训练
