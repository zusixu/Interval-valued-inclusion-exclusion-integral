# 区间值神经网络框架集合

基于PyTorch的区间值数据处理神经网络框架，包含多种模型实现和集成方法。

## 项目结构

```
final_ie_chi/
├── IVIE_FM/              # 区间值模糊积分网络 - FM实现
├── IVIE_Moebius/         # 区间值IE积分网络 - Moebius实现
├── IVCHI/                # 区间值Choquet积分网络 - 区间加法版本
├── CHI_IE/               # 集成学习网络
├── tests/                # 测试脚本
├── data_build.py         # 数据构建工具
├── requirements.txt      # 依赖包列表
└── README.md             # 本文档
```

## 模块介绍

### 1. [IVIE_FM](IVIE_FM/README.md)

区间值IE积分神经网络 - FM（Fuzzy Measure）实现版本

**核心特性**:
- 支持Algebraic_interval和Min_interval两种区间运算
- 基于模糊测度的特征融合
- 支持限制交互阶数，避免数值下溢
- 提供多种损失函数


**快速开始**:
```python
from IVIE_FM.ivie import IE
model = IE(feature_size=7, additivity_order=2, op='Algebraic_interval')
```

📖 [查看详细文档](IVIE_FM/README.md)

---

### 2. [IVIE_Moebius](IVIE_Moebius/README.md)

区间值IE积分神经网络 - Moebius实现版本

**核心特性**:
- Admissible order排序机制
- 单值/区间值模糊测度可选
- 记忆化特征组合算法
- 向量化批量处理


**快速开始**:
```python
from IVIE_Moebius.ieinn import IE
model = IE(feature_size=7, additivity_order=2, op='Min_interval', 
           fuzzy_measure='OutputLayer_single')
```

📖 [查看详细文档](IVIE_Moebius/README.md)

---

### 3. [IVCHI](IVCHI/README.md)

区间值Choquet积分神经网络 - 区间加法版本

**核心特性**:
- 继承自IVIE_FM，使用区间加法替代区间减法
- 完全兼容IVIE_FM的接口


**快速开始**:
```python
from IVCHI.ivchi import IVCHI
model = IVCHI(feature_size=7, additivity_order=2, op='Algebraic_interval')
```

📖 [查看详细文档](IVCHI/README.md)

---

### 4. [CHI_IE](CHI_IE/README.md)

集成学习神经网络

**核心特性**:
- 支持多个子模型（IVCHI/IVIE_Moebius）集成
- 灵活的集成层配置（FM/Moebius）
- 子模型多样性配置
- 统一的训练接口


**快速开始**:
```python
from CHI_IE.ensemble_ie import EnsembleIE
model = EnsembleIE(feature_size=7, num_base_models=3, 
                   base_model_type='IVCHI', ensemble_type='FM')
```

📖 [查看详细文档](CHI_IE/README.md)

---

## 快速安装

### 环境要求

- Python 3.9+
- PyTorch 2.2.2+
- CUDA 11.8+ (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用示例

### 基础使用流程

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from IVIE_FM.ivie import IE
from IVIE_FM.iv_loss import HausdorffIntervalLoss

# 1. 准备数据
# 区间值格式: [x1_l, x2_l, ..., xn_l, x1_u, x2_u, ..., xn_u]
X_train = torch.rand(100, 14)  # 7个特征的区间值
y_train = torch.rand(100, 2)   # 区间值标签 [下界, 上界]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)
test_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

# 2. 创建模型
model = IE(
    feature_size=7,
    additivity_order=2,          # 限制交互阶数
    op='Algebraic_interval',
    device='cuda'
)

# 3. 训练
criterion = HausdorffIntervalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.fit_and_valid(
    train_Loader=train_loader,
    test_Loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100
)

# 4. 预测
pred_l, pred_u = model(X_test)
```

### 数据预处理

```python
import pandas as pd

# 构造区间值数据（推荐方法）
df = pd.read_csv('data.csv')

# 归一化
df_normalized = (df - df.min()) / (df.max() - df.min())

# 构造区间
spread_ratio = 0.1
data_low = (df_normalized * (1 - spread_ratio)).clip(lower=0)
data_up = (df_normalized * (1 + spread_ratio)).clip(upper=1)

# 拼接成模型输入格式
X = torch.cat([
    torch.tensor(data_low.values, dtype=torch.float32),
    torch.tensor(data_up.values, dtype=torch.float32)
], dim=1)
```

## 参数推荐配置

### 通用参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| additivity_order | 2-3 | 避免高阶组合导致数值下溢 |
| op | Algebraic_interval | 连续可导，便于优化 |
| learning_rate | 0.001-0.005 | 配合学习率调度器 |
| batch_size | 32-64 | 根据数据集大小调整 |

### 区间运算选择

| 运算类型 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| Algebraic_interval | 连续可导，优化快 | 需限制阶数避免下溢 | 归一化数据 |
| Min_interval | 数值稳定 | 非光滑 | 原始数据，稳健性要求高 |

### 损失函数选择

| 损失函数 | 特点 | 适用场景 |
|---------|------|---------|
| interval_loss | 基础Hausdorff距离 | 简单任务 |
| ImprovedIntervalLoss | MSE + 有效性 + 宽度匹配 | **推荐**，大多数场景 |
| HausdorffIntervalLoss | 最大端点误差 | 对端点误差敏感的任务 |

## 测试

运行测试脚本：

```bash
# 测试IVIE_FM
python tests/test_ie_train.py

# 测试IVIE_Moebius
python tests/test_ivie_moebius.py

# 测试IVCHI
python tests/test_ivchi.py

# 测试CHI_IE
python tests/test_ensemble_ie.py

# 比较不同框架
python tests/compare_frameworks.py
```

## 常见问题

### Q: 预测值全是0或接近0？

**A**: 设置 `additivity_order=2` 或 `3`，避免高阶组合的数值下溢



### Q: 区间数据如何构造？

**A**: 推荐使用比例偏移法：
```python
spread_ratio = 0.1
data_low = (df * (1 - spread_ratio)).clip(lower=0)
data_up = (df * (1 + spread_ratio)).clip(upper=1)
```

### Q: GPU内存不足怎么办？

**A**: 
- 减小batch_size
- 降低additivity_order
- 使用梯度累积

## 项目依赖

主要依赖包：
- PyTorch 2.2.2
- NumPy
- Pandas (用于数据处理)

完整依赖列表见 [requirements.txt](requirements.txt)

## 更新日志

### 2026年1月
- ✨ 更新所有模块的README文档
- 📝 简化文档，专注于结构和使用说明
- 🔧 统一文档格式和风格

---

**最后更新**: 2026年1月18日
