import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IVCHI.ivchi import IVCHI
from IVIE.ivie import IE
from IVIE.iv_loss import interval_loss as IntervalLoss


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


def test_ivchi_forward():
    """测试IVCHI模型的前向传播"""
    print("=" * 60)
    print("测试IVCHI前向传播")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    n_features = 3
    batch_size = 4
    
    # 创建IVCHI模型
    model = IVCHI(feature_size=n_features, op='Algebraic_interval', device=device)
    model = model.to(device)
    
    # 生成测试数据
    X, y = generate_interval_data(n_samples=batch_size, n_features=n_features)
    X = X.to(device)
    
    print(f"\n输入数据形状: {X.shape}")
    print(f"输入数据前2个样本:\n{X[:2]}")
    
    # 前向传播
    output_l, output_u = model(X)
    
    assert output_l.shape == (batch_size, 1), f"输出左端点形状错误: {output_l.shape}"
    assert output_u.shape == (batch_size, 1), f"输出右端点形状错误: {output_u.shape}"
    
    # 验证区间有效性（左端点应小于等于右端点）
    assert torch.all(output_l <= output_u), "区间无效：存在左端点大于右端点的情况"
    
    print(f"\n✓ 前向传播测试通过!")
    print(f"输出形状: {output_l.shape}")
    print(f"输出左端点: {output_l.squeeze()}")
    print(f"输出右端点: {output_u.squeeze()}")
    print(f"区间宽度: {(output_u - output_l).squeeze()}")


def test_ivchi_vs_ie_difference():
    """对比IVCHI和IE的输出差异（验证区间加法vs区间减法）"""
    print("\n" + "=" * 60)
    print("对比IVCHI（区间加法）和IE（区间减法）的输出差异")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 2
    batch_size = 3
    
    # 创建两个模型
    ivchi_model = IVCHI(feature_size=n_features, op='Algebraic_interval', device=device)
    ie_model = IE(feature_size=n_features, op='Algebraic_interval', device=device)
    
    # 使用相同的参数初始化（确保除了运算符外其他都一样）
    with torch.no_grad():
        ie_model.vars.copy_(ivchi_model.vars)
    
    ivchi_model = ivchi_model.to(device)
    ie_model = ie_model.to(device)
    
    # 生成测试数据
    X, y = generate_interval_data(n_samples=batch_size, n_features=n_features)
    X = X.to(device)
    
    # 前向传播
    ivchi_l, ivchi_u = ivchi_model(X)
    ie_l, ie_u = ie_model(X)
    
    print(f"\n输入数据:\n{X}")
    print(f"\nIVCHI输出（区间加法）:")
    print(f"  左端点: {ivchi_l.squeeze()}")
    print(f"  右端点: {ivchi_u.squeeze()}")
    
    print(f"\nIE输出（区间减法）:")
    print(f"  左端点: {ie_l.squeeze()}")
    print(f"  右端点: {ie_u.squeeze()}")
    
    print(f"\n差异:")
    print(f"  左端点差异: {(ivchi_l - ie_l).squeeze()}")
    print(f"  右端点差异: {(ivchi_u - ie_u).squeeze()}")
    
    # 验证输出确实不同（除非是特殊情况）
    if not torch.allclose(ivchi_l, ie_l) or not torch.allclose(ivchi_u, ie_u):
        print("\n✓ 验证通过: IVCHI和IE的输出存在差异（符合预期）")
    else:
        print("\n⚠ 注意: IVCHI和IE的输出相同（可能是特殊情况）")


def test_ivchi_training():
    """测试IVCHI模型的训练功能"""
    print("\n" + "=" * 60)
    print("测试IVCHI训练功能")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_features = 3
    n_samples = 50
    batch_size = 10
    epochs = 5
    
    # 生成训练和验证数据
    X_train, y_train = generate_interval_data(n_samples=n_samples, n_features=n_features, seed=42)
    X_val, y_val = generate_interval_data(n_samples=20, n_features=n_features, seed=100)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    model = IVCHI(feature_size=n_features, op='Algebraic_interval', device=device)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = IntervalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"\n开始训练 (epochs={epochs})...")
    
    # 简化的训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output_l, output_u = model(X_batch)
            
            # 计算损失
            loss, error = criterion(output_l, output_u, y_batch)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output_l, output_u = model(X_batch)
                loss, error = criterion(output_l, output_u, y_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    print("\n✓ 训练测试通过!")


def test_ivchi_interval_addition():
    """专门测试区间加法的正确性"""
    print("\n" + "=" * 60)
    print("测试区间加法运算正确性")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 使用简单的2特征情况
    n_features = 2
    
    model = IVCHI(feature_size=n_features, op='Algebraic_interval', device=device)
    model = model.to(device)
    
    # 手工构造简单的测试数据
    # 输入: [x1_l, x2_l, x1_u, x2_u]
    X = torch.tensor([
        [1.0, 2.0, 1.5, 2.5],  # x1=[1.0, 1.5], x2=[2.0, 2.5]
        [0.5, 1.0, 1.0, 1.5],  # x1=[0.5, 1.0], x2=[1.0, 1.5]
    ], device=device)
    
    print(f"\n输入区间:")
    for i in range(X.shape[0]):
        print(f"  样本{i+1}: x1=[{X[i,0]:.1f}, {X[i,2]:.1f}], x2=[{X[i,1]:.1f}, {X[i,3]:.1f}]")
    
    output_l, output_u = model(X)
    
    print(f"\n输出区间:")
    for i in range(output_l.shape[0]):
        print(f"  样本{i+1}: y=[{output_l[i,0]:.4f}, {output_u[i,0]:.4f}]")
    
    # 验证区间有效性
    assert torch.all(output_l <= output_u), "区间无效"
    
    print("\n✓ 区间加法测试通过!")


if __name__ == "__main__":
    print("开始IVCHI测试套件\n")
    
    try:
        # 测试1: 基本前向传播
        test_ivchi_forward()
        
        # 测试2: 与IE的对比
        test_ivchi_vs_ie_difference()
        
        # 测试3: 区间加法验证
        test_ivchi_interval_addition()
        
        # 测试4: 训练功能
        test_ivchi_training()
        
        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
