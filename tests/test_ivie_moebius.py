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


from IVIE_Moebius.ieinn import IE
from IVIE_Moebius.iv_loss import HausdorffIntervalLoss


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


def test_moebius_ie_forward():
    """测试IVIE_Moebius IE模型的前向传播"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 3
    batch_size = 4
    
    # 创建模型 - 使用单值模糊测度
    model = IE(
        feature_size=n_features, 
        op='Min_interval', 
        alpha=0.5, 
        beta=0.5,
        fuzzy_measure='OutputLayer_single'
    )
    model = model.to(device)
    
    # 生成测试数据
    X, y = generate_interval_data(n_samples=batch_size, n_features=n_features)
    X = X.to(device)
    
    # 前向传播
    output_l, output_u = model(X)
    
    assert output_l.shape == (batch_size, 1), f"输出左端点形状错误: {output_l.shape}"
    assert output_u.shape == (batch_size, 1), f"输出右端点形状错误: {output_u.shape}"
    
    print("IVIE_Moebius 前向传播测试通过!")
    print(f"输出左端点: {output_l.T}")
    print(f"输出右端点: {output_u.T}")


def test_moebius_ie_train():
    """测试IVIE_Moebius IE模型的训练功能"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 3
    n_train = 80
    n_test = 20
    batch_size = 16
    epochs = 10
    
    # 创建模型 - 使用单值模糊测度
    model = IE(
        feature_size=n_features, 
        op='Min_interval', 
        alpha=0.5, 
        beta=0.5,
        fuzzy_measure='OutputLayer_single'
    )
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
    criterion = HausdorffIntervalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    print(f"开始训练 IVIE_Moebius (设备: {device})...")
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
    
    print("\nIVIE_Moebius 模型训练测试通过!")


def test_moebius_ie_algebraic_interval():
    """测试使用Algebraic_interval操作的IVIE_Moebius模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 3
    batch_size = 16
    epochs = 5
    
    # 创建使用Algebraic_interval的模型
    model = IE(
        feature_size=n_features, 
        op='Algebraic_interval',
        fuzzy_measure='OutputLayer_single'
    )
    model = model.to(device)
    
    # 生成数据
    X_train, y_train = generate_interval_data(n_samples=50, n_features=n_features, seed=42)
    X_test, y_test = generate_interval_data(n_samples=10, n_features=n_features, seed=123)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    criterion = HausdorffIntervalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("测试 IVIE_Moebius Algebraic_interval 操作...")
    val_loss = model.fit_and_valid(
        train_Loader=train_loader,
        test_Loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )
    
    print(f"IVIE_Moebius Algebraic_interval 模型验证损失: {val_loss:.6f}")
    print("IVIE_Moebius Algebraic_interval 操作测试通过!")


def test_moebius_ie_interval_output():
    """测试使用区间值输出层的IVIE_Moebius模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 3
    batch_size = 16
    epochs = 5
    
    # 创建使用区间值输出层的模型
    model = IE(
        feature_size=n_features, 
        op='Min_interval',
        alpha=0.3,
        beta=0.7,
        fuzzy_measure='OutputLayer_interval'
    )
    model = model.to(device)
    
    # 生成数据
    X_train, y_train = generate_interval_data(n_samples=50, n_features=n_features, seed=42)
    X_test, y_test = generate_interval_data(n_samples=10, n_features=n_features, seed=123)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    criterion = HausdorffIntervalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("测试 IVIE_Moebius 区间值输出层...")
    val_loss = model.fit_and_valid(
        train_Loader=train_loader,
        test_Loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )
    
    print(f"IVIE_Moebius 区间值输出层模型验证损失: {val_loss:.6f}")
    
    # 测试权重提取
    weights = model.get_output_weights()
    print(f"输出层权重数量: {len(weights)}")
    print("IVIE_Moebius 区间值输出层测试通过!")


def test_moebius_ie_comparison():
    """对比不同配置的IVIE_Moebius模型性能"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 3
    n_train = 100
    n_test = 30
    batch_size = 16
    epochs = 8
    
    # 生成一致的数据集
    X_train, y_train = generate_interval_data(n_samples=n_train, n_features=n_features, seed=42)
    X_test, y_test = generate_interval_data(n_samples=n_test, n_features=n_features, seed=123)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    configs = [
        {'op': 'Min_interval', 'fuzzy_measure': 'OutputLayer_single', 'name': 'Min_interval + 单值输出'},
        {'op': 'Algebraic_interval', 'fuzzy_measure': 'OutputLayer_single', 'name': 'Algebraic_interval + 单值输出'},
        {'op': 'Min_interval', 'fuzzy_measure': 'OutputLayer_interval', 'name': 'Min_interval + 区间值输出'},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n测试配置: {config['name']}")
        
        model = IE(
            feature_size=n_features,
            op=config['op'],
            alpha=0.5,
            beta=0.5,
            fuzzy_measure=config['fuzzy_measure']
        )
        model = model.to(device)
        
        criterion = HausdorffIntervalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        val_loss = model.fit_and_valid(
            train_Loader=train_loader,
            test_Loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs
        )
        
        results[config['name']] = val_loss
        print(f"配置 {config['name']} 验证损失: {val_loss:.6f}")
    
    print("\n=== IVIE_Moebius 模型对比结果 ===")
    for name, loss in results.items():
        print(f"{name}: {loss:.6f}")
    
    print("\nIVIE_Moebius 对比测试完成!")


if __name__ == "__main__":
    print("=" * 60)
    print("IVIE_Moebius 框架测试开始")
    print("=" * 60)
    
    # 检查环境
    check_conda_environment()
    
    # 运行各项测试
    print("\n1. 测试前向传播...")
    test_moebius_ie_forward()
    
    print("\n2. 测试模型训练...")
    test_moebius_ie_train()
    
    print("\n3. 测试 Algebraic_interval 操作...")
    test_moebius_ie_algebraic_interval()
    
    print("\n4. 测试区间值输出层...")
    test_moebius_ie_interval_output()
    
    print("\n5. 模型性能对比测试...")
    test_moebius_ie_comparison()
    
    print("\n" + "=" * 60)
    print("IVIE_Moebius 框架测试全部完成!")
    print("=" * 60)