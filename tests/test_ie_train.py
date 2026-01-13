import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def check_conda_environment(expected_env='ieiv'):
    """检查当前是否在指定的 conda 环境中运行"""
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    
    # 从路径中提取环境名
    if conda_prefix:
        env_name = Path(conda_prefix).name
    else:
        env_name = conda_env
    
    if env_name != expected_env:
        print(f"警告: 当前 conda 环境为 '{env_name}'，期望为 '{expected_env}'")
        print(f"请运行: conda activate {expected_env}")
        print(f"Python 路径: {sys.executable}")
        return False
    
    print(f"✓ 正在使用 conda 环境: {env_name}")
    print(f"  Python 路径: {sys.executable}")
    return True


from IVIE.ivie import IE


class IntervalLoss(nn.Module):
    """区间值损失函数：计算预测区间与真实区间之间的距离"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_l, pred_u, target):
        """
        pred_l: 预测区间的左端点
        pred_u: 预测区间的右端点
        target: 真实区间 [左端点, 右端点]
        """
        target_l = target[:, 0:1]
        target_u = target[:, 1:2]
        
        # 计算区间端点之间的MSE
        loss_l = torch.mean((pred_l - target_l) ** 2)
        loss_u = torch.mean((pred_u - target_u) ** 2)
        loss = loss_l + loss_u
        
        # 计算误差距离
        distance = torch.abs(pred_l - target_l) + torch.abs(pred_u - target_u)
        
        return loss, distance


def generate_interval_data(n_samples=100, n_features=3, seed=42):
    """
    生成区间值测试数据
    
    输入格式: [x1_l, x2_l, ..., xn_l, x1_u, x2_u, ..., xn_u]
    其中 l 表示左端点（下界），u 表示右端点（上界）
    
    输出格式: [y_l, y_u] 区间值标签
    """
    torch.manual_seed(seed)
    
    # 生成基础值和区间宽度
    base = torch.rand(n_samples, n_features)
    spread = torch.rand(n_samples, n_features) * 0.3  # 区间宽度
    
    # 左端点和右端点
    x_lower = base
    x_upper = base + spread
    
    # 拼接成输入格式 [左端点们, 右端点们]
    X = torch.cat([x_lower, x_upper], dim=1)
    
    # 生成区间值标签（简单的线性组合 + 区间）
    y_lower = torch.sum(x_lower, dim=1, keepdim=True) / n_features
    y_upper = torch.sum(x_upper, dim=1, keepdim=True) / n_features
    y = torch.cat([y_lower, y_upper], dim=1)
    
    return X, y


def test_ie_forward():
    """测试IE模型的前向传播"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 3
    batch_size = 4
    
    # 创建模型（传入device参数）
    model = IE(feature_size=n_features, op='Min_interval', alpha=0.5, beta=0.5, device=device)
    model = model.to(device)
    
    # 生成测试数据
    X, y = generate_interval_data(n_samples=batch_size, n_features=n_features)
    X = X.to(device)
    
    # 前向传播
    output_l, output_u = model(X)
    
    assert output_l.shape == (batch_size, 1), f"输出左端点形状错误: {output_l.shape}"
    assert output_u.shape == (batch_size, 1), f"输出右端点形状错误: {output_u.shape}"
    
    print("前向传播测试通过!")
    print(f"输出左端点: {output_l.T}")
    print(f"输出右端点: {output_u.T}")


def test_ie_train():
    """测试IE模型的训练功能"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 3
    n_train = 80
    n_test = 20
    batch_size = 16
    epochs = 10
    
    # 创建模型（传入device参数）
    model = IE(feature_size=n_features, op='Min_interval', alpha=0.5, beta=0.5, device=device)
    model = model.to(device)
    
    # 生成训练和测试数据
    X_train, y_train = generate_interval_data(n_samples=n_train, n_features=n_features, seed=42)
    X_test, y_test = generate_interval_data(n_samples=n_test, n_features=n_features, seed=123)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 定义损失函数和优化器
    criterion = IntervalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    print(f"开始训练 (设备: {device})...")
    val_loss = model.fit_and_valid(
        train_Loader=train_loader,
        test_Loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )
    
    print(f"\n最终验证损失: {val_loss:.6f}")
    
    # 验证训练后模型仍然可以正常前向传播
    model.eval()
    with torch.no_grad():
        X_sample = X_test[:5].to(device)
        pred_l, pred_u = model(X_sample)
        print(f"\n预测样例:")
        print(f"预测左端点: {pred_l.T}")
        print(f"预测右端点: {pred_u.T}")
        print(f"真实左端点: {y_test[:5, 0]}")
        print(f"真实右端点: {y_test[:5, 1]}")
    
    assert len(model.train_loss_list) == epochs, "训练损失列表长度不正确"
    assert len(model.val_loss_list) == epochs, "验证损失列表长度不正确"
    
    print("\nIE模型训练测试通过!")


def test_ie_algebraic_interval():
    """测试使用Algebraic_interval操作的IE模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 3
    batch_size = 16
    epochs = 5
    
    # 创建使用Algebraic_interval的模型（传入device参数）
    model = IE(feature_size=n_features, op='Algebraic_interval', device=device)
    model = model.to(device)
    
    # 生成数据
    X_train, y_train = generate_interval_data(n_samples=50, n_features=n_features, seed=42)
    X_test, y_test = generate_interval_data(n_samples=10, n_features=n_features, seed=123)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    criterion = IntervalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("测试 Algebraic_interval 操作...")
    val_loss = model.fit_and_valid(
        train_Loader=train_loader,
        test_Loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )
    
    print(f"Algebraic_interval 模型验证损失: {val_loss:.6f}")
    print("Algebraic_interval 操作测试通过!")


def test_ie_with_uci_data():
    """使用 data_build.py 中的 generate_data() 构建的UCI数据集进行训练测试"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 直接实现 generate_data 函数，避免 data_build.py 中的导入问题
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from ucimlrepo import fetch_ucirepo
    
    print("正在从UCI获取数据集...")
    
    # fetch dataset (Auto MPG dataset)
    auto_mpg = fetch_ucirepo(id=9)

    # data (as pandas dataframes)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets
  
    X.fillna(X.mean(), inplace=True)
    df = X
    y = (y - y.min()) / (y.max() - y.min())
    df = (df - df.min()) / (df.max() - df.min())  # 归一化

    # 初始化两个空的DataFrame，用于存储低于和高于两倍标准差的值
    data_low = pd.DataFrame(index=df.index, columns=df.columns)
    data_up = pd.DataFrame(index=df.index, columns=df.columns)
    
    # 遍历每一列，计算标准差，并基于标准差创建新的列，获得了区间值集low,up表示上下界
    for feature in df.columns:
        std_deviation = df[feature].std()
        data_low[feature] = df[feature] - 2 * std_deviation
        data_up[feature] = df[feature] + 2 * std_deviation
    data_low = pd.concat((data_low, y), axis=1)
    data_up = pd.concat((data_up, y), axis=1)
    Df = pd.concat((data_low, data_up), axis=1)
    lenth = len(data_up.columns)  # 给出特征＋标签的数量

    data_train, data_test = train_test_split(Df, test_size=0.2, random_state=42)
    # 前一半是l后一半是u
    X_train = pd.concat((data_train.iloc[:, :lenth - 1], data_train.iloc[:, lenth:2 * lenth - 1]), axis=1)
    y_train = pd.concat((data_train.iloc[:, lenth - 1:lenth], data_train.iloc[:, 2 * lenth - 1:2 * lenth]), axis=1)
    X_test = pd.concat((data_test.iloc[:, :lenth - 1], data_test.iloc[:, lenth:2 * lenth - 1]), axis=1)
    y_test = pd.concat((data_test.iloc[:, lenth - 1:lenth], data_test.iloc[:, 2 * lenth - 1:2 * lenth]), axis=1)
    
    # 获取特征数量（输入数据的一半，因为包含左右端点）
    n_features = X_train.shape[1] // 2
    print(f"特征数量: {n_features}")
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    # 训练参数
    batch_size = 32
    epochs = 500
    learning_rate = 0.01
    
    # 创建模型，使用指定参数
    model = IE(
        feature_size=n_features, 
        op='Algebraic_interval', 
        alpha=1, 
        beta=0, 
        device=device
    )
    model = model.to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 定义损失函数和优化器
    criterion = IntervalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n训练参数:")
    print(f"  - op: Algebraic_interval")
    print(f"  - alpha: 1")
    print(f"  - beta: 0")
    print(f"  - epochs: {epochs}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - device: {device}")
    
    print("\n开始训练...")
    val_loss = model.fit_and_valid(
        train_Loader=train_loader,
        test_Loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )
    
    print(f"\n最终验证损失: {val_loss:.6f}")
    
    # 验证训练后模型可以正常前向传播
    model.eval()
    with torch.no_grad():
        X_sample = X_test_tensor[:5].to(device)
        pred_l, pred_u = model(X_sample)
        print(f"\n预测样例 (前5个):")
        print(f"预测左端点: {pred_l.squeeze().cpu().numpy()}")
        print(f"预测右端点: {pred_u.squeeze().cpu().numpy()}")
        print(f"真实左端点: {y_test_tensor[:5, 0].numpy()}")
        print(f"真实右端点: {y_test_tensor[:5, 1].numpy()}")
    
    assert len(model.train_loss_list) == epochs, "训练损失列表长度不正确"
    assert len(model.val_loss_list) == epochs, "验证损失列表长度不正确"
    
    print("\nUCI数据集训练测试通过!")
    return model


if __name__ == '__main__':
    # 检查 conda 环境
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    if not check_conda_environment('ieiv'):
        print("\n继续运行测试...\n")
    
    print("\n" + "=" * 50)
    print("测试1: IE模型前向传播")
    print("=" * 50)
    test_ie_forward()
    
    print("\n" + "=" * 50)
    print("测试2: IE模型训练 (Min_interval)")
    print("=" * 50)
    test_ie_train()
    
    print("\n" + "=" * 50)
    print("测试3: IE模型训练 (Algebraic_interval)")
    print("=" * 50)
    test_ie_algebraic_interval()
    
    print("\n" + "=" * 50)
    print("测试4: 使用UCI数据集训练 (Algebraic_interval, alpha=1, beta=0, epochs=500)")
    print("=" * 50)
    test_ie_with_uci_data()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
