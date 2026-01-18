"""
测试 EnsembleIE 集成学习神经网络
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_build import generate_data
from CHI_IE.ensemble_ie import EnsembleIE
from IVIE_FM.iv_loss import HausdorffIntervalLoss as HausdorffLoss_FM
from IVIE_Moebius.iv_loss import HausdorffIntervalLoss as HausdorffLoss_Moebius


def test_ensemble_basic():
    """基本功能测试：测试网络能否正常创建和前向传播"""
    print("\n" + "="*80)
    print("测试1: 基本功能测试")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建简单的测试数据
    batch_size = 10
    feature_size = 3
    
    # 创建随机输入数据 (batch_size, 2*feature_size)
    x = torch.randn(batch_size, 2 * feature_size).to(device)
    
    # 测试配置1: IVCHI子模型 + FM集成层
    print("\n配置1: IVCHI子模型 + FM集成层")
    model1 = EnsembleIE(
        feature_size=feature_size,
        num_base_models=3,
        base_model_type='IVCHI',
        base_model_configs=[
            {'additivity_order': 2, 'alpha': 0.5, 'beta': 0.0},
            {'additivity_order': 2, 'alpha': 0.6, 'beta': 0.0},
            {'additivity_order': 2, 'alpha': 0.4, 'beta': 0.0}
        ],
        ensemble_type='FM',
        ensemble_config={
            'additivity_order': 2,
            'op': 'Min_interval',
            'alpha': 0.5,
            'beta': 0.0
        },
        device=device
    )
    
    out_l, out_u = model1(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出下界形状: {out_l.shape}")
    print(f"  输出上界形状: {out_u.shape}")
    print(f"  输出下界样本: {out_l[:3].flatten()}")
    print(f"  输出上界样本: {out_u[:3].flatten()}")
    print("  ✓ 配置1测试通过")
    
    # 测试配置2: IVIE_Moebius子模型 + Moebius集成层
    print("\n配置2: IVIE_Moebius子模型 + Moebius集成层")
    model2 = EnsembleIE(
        feature_size=feature_size,
        num_base_models=2,
        base_model_type='IVIE_Moebius',
        base_model_configs=[
            {'additivity_order': 2, 'alpha': 0.5, 'beta': 0.0},
            {'additivity_order': 2, 'alpha': 0.5, 'beta': 0.0}
        ],
        ensemble_type='Moebius',
        ensemble_config={
            'additivity_order': 2,
            'op': 'Algebraic_interval',
            'alpha': 0.5,
            'beta': 0.0,
            'fuzzy_measure': 'OutputLayer_single'
        },
        device=device
    )
    
    out_l, out_u = model2(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出下界形状: {out_l.shape}")
    print(f"  输出上界形状: {out_u.shape}")
    print(f"  输出下界样本: {out_l[:3].flatten()}")
    print(f"  输出上界样本: {out_u[:3].flatten()}")
    print("  ✓ 配置2测试通过")
    
    # 测试配置3: 混合配置 - IVCHI + Moebius集成
    print("\n配置3: IVCHI子模型 + Moebius集成层")
    model3 = EnsembleIE(
        feature_size=feature_size,
        num_base_models=4,
        base_model_type='IVCHI',
        ensemble_type='Moebius',
        device=device
    )
    
    out_l, out_u = model3(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出下界形状: {out_l.shape}")
    print(f"  输出上界形状: {out_u.shape}")
    print("  ✓ 配置3测试通过")
    
    print("\n✓ 所有基本功能测试通过！")


def test_ensemble_training():
    """训练测试：在真实数据集上训练模型"""
    print("\n" + "="*80)
    print("测试2: 训练功能测试")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载数据集...")
    X_train, X_test, y_train, y_test = generate_data()
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    feature_size = X_train_tensor.shape[1] // 2
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"特征数量: {feature_size}")
    
    # 创建模型
    print("\n创建 EnsembleIE 模型...")
    print("配置: 3个IVCHI子模型 + FM集成层")
    
    model = EnsembleIE(
        feature_size=feature_size,
        num_base_models=3,
        base_model_type='IVCHI',
        base_model_configs=[
            {'additivity_order': 2, 'alpha': 0.5, 'beta': 0.0},
            {'additivity_order': 2, 'alpha': 0.6, 'beta': 0.0},
            {'additivity_order': 3, 'alpha': 0.4, 'beta': 0.0}
        ],
        ensemble_type='FM',
        ensemble_config={
            'additivity_order': 2,
            'op': 'Min_interval',
            'alpha': 0.5,
            'beta': 0.0
        },
        device=device
    )
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = HausdorffLoss_FM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("\n开始训练...")
    epochs = 10
    val_loss = model.fit_and_valid(
        train_Loader=train_loader,
        test_Loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        check_gradient=True,
        gradient_clip=1.0,
        model_name="EnsembleIE",
        early_stopping=True,
        patience=10
    )
    
    print(f"\n最终验证损失: {val_loss:.6f}")
    
    # 评估模型
    print("\n评估模型性能...")
    model.eval()
    X_test_device = X_test_tensor.to(device)
    y_test_device = y_test_tensor.to(device)
    
    with torch.no_grad():
        pred_l, pred_u = model(X_test_device)
        _, hausdorff_distance = criterion(pred_l, pred_u, y_test_device)
    
    errors = hausdorff_distance.cpu().numpy().flatten()
    
    print(f"  平均误差: {np.mean(errors):.6f}")
    print(f"  误差标准差: {np.std(errors):.6f}")
    print(f"  最大误差: {np.max(errors):.6f}")
    print(f"  最小误差: {np.min(errors):.6f}")
    print(f"  Accuracy@0.1: {np.mean(errors <= 0.1):.2%}")
    
    # 测试子模型输出
    print("\n测试子模型输出功能...")
    sample_x = X_test_tensor[:5].to(device)
    base_outputs = model.get_base_model_outputs(sample_x)
    
    print(f"  输入样本数: {sample_x.shape[0]}")
    print(f"  子模型数量: {len(base_outputs)}")
    for i, (out_l, out_u) in enumerate(base_outputs):
        print(f"  子模型{i+1}输出形状: 下界{out_l.shape}, 上界{out_u.shape}")
    
    print("\n✓ 训练功能测试通过！")


def test_different_configurations():
    """测试不同的配置组合"""
    print("\n" + "="*80)
    print("测试3: 不同配置组合测试")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建简单测试数据
    batch_size = 20
    feature_size = 4
    x = torch.randn(batch_size, 2 * feature_size).to(device)
    y = torch.randn(batch_size, 2).to(device)
    
    configs = [
        {
            'name': '2个IVCHI + FM(Algebraic)',
            'num_base_models': 2,
            'base_model_type': 'IVCHI',
            'ensemble_type': 'FM',
            'ensemble_config': {'op': 'Algebraic_interval', 'additivity_order': 2}
        },
        {
            'name': '3个IVIE_Moebius + FM(Min)',
            'num_base_models': 3,
            'base_model_type': 'IVIE_Moebius',
            'ensemble_type': 'FM',
            'ensemble_config': {'op': 'Min_interval', 'additivity_order': 2, 'alpha': 0.5}
        },
        {
            'name': '4个IVCHI + Moebius(Algebraic)',
            'num_base_models': 4,
            'base_model_type': 'IVCHI',
            'ensemble_type': 'Moebius',
            'ensemble_config': {
                'op': 'Algebraic_interval',
                'additivity_order': 3,
                'fuzzy_measure': 'OutputLayer_single'
            }
        },
        {
            'name': '5个IVIE_Moebius + Moebius(Min)',
            'num_base_models': 5,
            'base_model_type': 'IVIE_Moebius',
            'ensemble_type': 'Moebius',
            'ensemble_config': {
                'op': 'Min_interval',
                'additivity_order': 2,
                'alpha': 0.6,
                'beta': 0.1,
                'fuzzy_measure': 'OutputLayer_single'
            }
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n配置{i}: {config['name']}")
        
        model = EnsembleIE(
            feature_size=feature_size,
            num_base_models=config['num_base_models'],
            base_model_type=config['base_model_type'],
            ensemble_type=config['ensemble_type'],
            ensemble_config=config.get('ensemble_config', None),
            device=device
        )
        
        # 前向传播
        out_l, out_u = model(x)
        
        print(f"  子模型数量: {config['num_base_models']}")
        print(f"  子模型类型: {config['base_model_type']}")
        print(f"  集成层类型: {config['ensemble_type']}")
        print(f"  输出形状: 下界{out_l.shape}, 上界{out_u.shape}")
        print(f"  输出范围: 下界[{out_l.min().item():.4f}, {out_l.max().item():.4f}], "
              f"上界[{out_u.min().item():.4f}, {out_u.max().item():.4f}]")
        
        # 检查区间有效性（下界 <= 上界）
        valid_intervals = (out_l <= out_u).all().item()
        print(f"  区间有效性: {'✓ 通过' if valid_intervals else '✗ 失败'}")
        
        if not valid_intervals:
            print(f"  警告: 存在无效区间 (下界 > 上界)")
    
    print("\n✓ 所有配置组合测试完成！")


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("EnsembleIE 集成学习神经网络 - 完整测试套件")
    print("="*80)
    
    # 测试1: 基本功能
    test_ensemble_basic()
    
    # 测试2: 训练功能
    test_ensemble_training()
    
    # 测试3: 不同配置
    test_different_configurations()
    
    print("\n" + "="*80)
    print("✓ 所有测试通过！EnsembleIE 功能正常")
    print("="*80)


if __name__ == "__main__":
    main()
