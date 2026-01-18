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


from IVIE_FM.ivie import IE
from IVIE_FM.iv_loss import interval_loss as IntervalLoss
from IVIE_FM.iv_loss import ImprovedIntervalLoss, HausdorffIntervalLoss



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
        alpha=0.5, 
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
    print(f"  - alpha: 0.5")
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


def test_ie_with_uci_data_improved():
    """
    使用改进的训练配置在UCI数据集上训练
    
    改进点:
    1. 改进的数据预处理（避免负值和越界）
    2. 改进的损失函数（ImprovedIntervalLoss）
    3. 更好的优化器配置（AdamW + 学习率调度）
    4. 早停机制
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    
    # 归一化到 [0, 1]
    y = (y - y.min()) / (y.max() - y.min())
    df = (df - df.min()) / (df.max() - df.min())

    # ============== 改进1: 更好的区间构造方法 ==============
    # 使用比例偏移而不是标准差偏移，避免负值和越界
    data_low = pd.DataFrame(index=df.index, columns=df.columns)
    data_up = pd.DataFrame(index=df.index, columns=df.columns)
    
    spread_ratio = 0.1  # 使用10%的区间宽度
    for feature in df.columns:
        # 方法1: 比例偏移（推荐）
        data_low[feature] = (df[feature] * (1 - spread_ratio)).clip(lower=0)
        data_up[feature] = (df[feature] * (1 + spread_ratio)).clip(upper=1)
    
    # 对标签也进行类似处理
    y_spread = 0.05
    y_low = (y * (1 - y_spread)).clip(lower=0)
    y_up = (y * (1 + y_spread)).clip(upper=1)
    
    data_low = pd.concat((data_low, y_low), axis=1)
    data_up = pd.concat((data_up, y_up), axis=1)
    Df = pd.concat((data_low, data_up), axis=1)
    lenth = len(data_up.columns)

    data_train, data_test = train_test_split(Df, test_size=0.2, random_state=42)
    
    X_train = pd.concat((data_train.iloc[:, :lenth - 1], data_train.iloc[:, lenth:2 * lenth - 1]), axis=1)
    y_train = pd.concat((data_train.iloc[:, lenth - 1:lenth], data_train.iloc[:, 2 * lenth - 1:2 * lenth]), axis=1)
    X_test = pd.concat((data_test.iloc[:, :lenth - 1], data_test.iloc[:, lenth:2 * lenth - 1]), axis=1)
    y_test = pd.concat((data_test.iloc[:, lenth - 1:lenth], data_test.iloc[:, 2 * lenth - 1:2 * lenth]), axis=1)
    
    n_features = X_train.shape[1] // 2
    print(f"特征数量: {n_features}")
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 检查数据范围
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    print(f"\n数据范围检查:")
    print(f"  X_train 范围: [{X_train_tensor.min():.4f}, {X_train_tensor.max():.4f}]")
    print(f"  y_train 范围: [{y_train_tensor.min():.4f}, {y_train_tensor.max():.4f}]")
    
    # ============== 改进2: 更好的训练参数 ==============
    batch_size = 32
    epochs = 300  # 减少epochs，配合早停
    learning_rate = 0.005  # 稍大的学习率
    weight_decay = 1e-5  # 较小的权重衰减
    
    # 创建模型
    # 关键改进: 限制 additivity_order 为2或3，避免高阶组合导致的数值下溢
    # 当使用 Algebraic_interval 时，高阶乘法会使特征值趋近于0
    additivity_order = 2  # 只考虑2阶交互，避免数值问题
    
    model = IE(
        feature_size=n_features, 
        additivity_order=additivity_order,  # 限制交互阶数
        op='Algebraic_interval',
        alpha=0.5, 
        beta=0, 
        device=device
    )
    model = model.to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ============== 改进3: 使用改进的损失函数 ==============
    criterion = ImprovedIntervalLoss(validity_weight=0.1, width_weight=0.05)
    
    # ============== 改进4: 使用AdamW + 学习率调度 ==============
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    print(f"\n改进的训练参数:")
    print(f"  - op: Algebraic_interval")
    print(f"  - additivity_order: {additivity_order} (限制交互阶数避免数值下溢)")
    print(f"  - alpha: 0.5, beta: 0")
    print(f"  - epochs: {epochs}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - weight_decay: {weight_decay}")
    print(f"  - 损失函数: ImprovedIntervalLoss")
    print(f"  - 调度器: CosineAnnealingLR")
    print(f"  - device: {device}")
    
    print("\n开始训练...")
    
    # 使用改进的训练循环（带早停和学习率调度）
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    import time
    start = time.time()
    model.train_loss_list = []
    model.val_loss_list = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputsl, outputsu = model(images)
            loss, _ = criterion(outputsl, outputsu, labels)
            train_loss += loss.item() * len(labels)
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputsl, outputsu = model(images)
                loss, _ = criterion(outputsl, outputsu, labels)
                val_loss += loss.item() * len(labels)
        
        avg_val_loss = val_loss / len(test_loader.dataset)
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        model.train_loss_list.append(avg_train_loss)
        model.val_loss_list.append(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], train_loss: {avg_train_loss:.6f}, '
                  f'val_loss: {avg_val_loss:.6f}, lr: {current_lr:.6f}')
        
        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发于 epoch {epoch + 1}")
            model.load_state_dict(best_model_state)
            break
    
    print(f"\n训练时间: {time.time() - start:.2f}秒")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        X_sample = X_test_tensor[:10].to(device)
        pred_l, pred_u = model(X_sample)
        
        print(f"\n预测样例 (前10个):")
        print(f"{'预测下界':<12} {'预测上界':<12} {'真实下界':<12} {'真实上界':<12} {'下界误差':<12} {'上界误差':<12}")
        print("-" * 72)
        for i in range(10):
            pl = pred_l[i, 0].item()
            pu = pred_u[i, 0].item()
            tl = y_test_tensor[i, 0].item()
            tu = y_test_tensor[i, 1].item()
            el = abs(pl - tl)
            eu = abs(pu - tu)
            print(f"{pl:<12.4f} {pu:<12.4f} {tl:<12.4f} {tu:<12.4f} {el:<12.4f} {eu:<12.4f}")
        
        # 计算整体指标
        all_pred_l, all_pred_u = model(X_test_tensor.to(device))
        mae_lower = torch.mean(torch.abs(all_pred_l - y_test_tensor[:, 0:1].to(device))).item()
        mae_upper = torch.mean(torch.abs(all_pred_u - y_test_tensor[:, 1:2].to(device))).item()
        
        print(f"\n整体评估指标:")
        print(f"  下界MAE: {mae_lower:.6f}")
        print(f"  上界MAE: {mae_upper:.6f}")
        print(f"  平均MAE: {(mae_lower + mae_upper) / 2:.6f}")
    
    print("\n改进版UCI数据集训练测试完成!")
    return model


if __name__ == '__main__':
    # 检查 conda 环境
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    if not check_conda_environment('ieiv'):
        print("\n继续运行测试...\n")
    
    # print("\n" + "=" * 50)
    # print("测试1: IE模型前向传播")
    # print("=" * 50)
    # test_ie_forward()
    
    # print("\n" + "=" * 50)
    # print("测试2: IE模型训练 (Min_interval)")
    # print("=" * 50)
    # test_ie_train()
    
    # print("\n" + "=" * 50)
    # print("测试3: IE模型训练 (Algebraic_interval)")
    # print("=" * 50)
    # test_ie_algebraic_interval()
    
    # print("\n" + "=" * 50)
    # print("测试4: 使用UCI数据集训练 (原始版本)")
    # print("=" * 50)
    # test_ie_with_uci_data()
    
    print("\n" + "=" * 50)
    print("测试5: 使用UCI数据集训练 (改进版本)")
    print("=" * 50)
    test_ie_with_uci_data_improved()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
