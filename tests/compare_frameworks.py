"""
IVCHI vs IVIE_FM vs IVIE_Moebius 性能对比程序
使用UCI Auto MPG数据集和HausdorffIntervalLoss损失函数
绘制REC曲线进行可视化对比
支持并行训练多个模型
"""
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_INDEX_PATH = _PROJECT_ROOT / "data_test" / "data_index.xlsx"
MSE_RESULT_PATH = _PROJECT_ROOT / "data_test" / "mse_results.xlsx"
FIG_DIR = _PROJECT_ROOT / "figs"

from data_build import generate_data
from IVIE_FM.ivie import IE as IE_FM
from IVIE_FM.iv_loss import HausdorffIntervalLoss as HausdorffLoss_FM
from IVCHI.ivchi import IVCHI
from IVIE_Moebius.ieinn import IE as IE_Moebius
from IVIE_Moebius.iv_loss import HausdorffIntervalLoss as HausdorffLoss_Moebius



def plot_rec_curve(errors_dict, save_path='comparison_rec_curve.png'):
    """绘制REC曲线"""
    plt.figure(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'D']
    
    # 使用所有模型的最大误差作为统一的横坐标范围
    max_error = max(errors.max() for errors in errors_dict.values())
    tolerances = np.linspace(0, max_error, 100)
    
    for idx, (model_name, errors) in enumerate(errors_dict.items()):
        # 计算REC曲线 - 使用统一的tolerances
        accuracies = np.array([np.mean(errors <= tol) for tol in tolerances])
        
        plt.plot(tolerances, accuracies, 
                label=model_name, 
                linewidth=2.5,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                markevery=10,
                markersize=6)
    
    plt.xlabel('Error Tolerance', fontsize=12, fontweight='bold')
    plt.ylabel('Prediction Accuracy', fontsize=12, fontweight='bold')
    plt.title('Regression Error Characteristic (REC) Curve\nIVIE_FM vs IVIE_Moebius', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nREC曲线已保存到: {save_path}")
    # plt.show()


def plot_training_curves(history_dict, save_path='comparison_training_curves.png'):
    """绘制训练曲线对比 - 分成两张图"""
    
    # 分组：Min_interval操作的模型 vs Algebraic操作的模型
    min_interval_models = {}
    algebraic_models = {}
    
    for model_name, history in history_dict.items():
        if 'Algebraic' in model_name:
            algebraic_models[model_name] = history
        else:
            min_interval_models[model_name] = history
    
    colors = ['#2E86AB', '#A23B72']
    
    # 第一张图：Min_interval 操作的模型 (IVCHI_FM 和 IVCHI_Moebius)
    if min_interval_models:
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
        
        # 训练损失
        for idx, (model_name, history) in enumerate(min_interval_models.items()):
            axes1[0].plot(history['train_loss'], 
                        label=model_name, 
                        linewidth=2,
                        color=colors[idx % len(colors)])
        
        axes1[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes1[0].set_ylabel('Training Loss', fontsize=11, fontweight='bold')
        axes1[0].set_title('Training Loss Comparison (Min_interval)', fontsize=12, fontweight='bold')
        axes1[0].legend(fontsize=10)
        axes1[0].grid(True, alpha=0.3, linestyle='--')
        
        # 验证损失
        for idx, (model_name, history) in enumerate(min_interval_models.items()):
            axes1[1].plot(history['val_loss'], 
                        label=model_name, 
                        linewidth=2,
                        color=colors[idx % len(colors)])
        
        axes1[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes1[1].set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
        axes1[1].set_title('Validation Loss Comparison (Min_interval)', fontsize=12, fontweight='bold')
        axes1[1].legend(fontsize=10)
        axes1[1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_path1 = save_path.replace('.png', '_min_interval.png')
        plt.savefig(save_path1, dpi=300, bbox_inches='tight')
        print(f"Min_interval训练曲线已保存到: {save_path1}")
        plt.show()
    
    # 第二张图：Algebraic 操作的模型 (IVIE_FM (Algebraic) 和 IVIE_Moebius (Algebraic))
    if algebraic_models:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        
        # 训练损失
        for idx, (model_name, history) in enumerate(algebraic_models.items()):
            axes2[0].plot(history['train_loss'], 
                        label=model_name, 
                        linewidth=2,
                        color=colors[idx % len(colors)])
        
        axes2[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes2[0].set_ylabel('Training Loss', fontsize=11, fontweight='bold')
        axes2[0].set_title('Training Loss Comparison (Algebraic)', fontsize=12, fontweight='bold')
        axes2[0].legend(fontsize=10)
        axes2[0].grid(True, alpha=0.3, linestyle='--')
        
        # 验证损失
        for idx, (model_name, history) in enumerate(algebraic_models.items()):
            axes2[1].plot(history['val_loss'], 
                        label=model_name, 
                        linewidth=2,
                        color=colors[idx % len(colors)])
        
        axes2[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes2[1].set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
        axes2[1].set_title('Validation Loss Comparison (Algebraic)', fontsize=12, fontweight='bold')
        axes2[1].legend(fontsize=10)
        axes2[1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_path2 = save_path.replace('.png', '_algebraic.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"Algebraic训练曲线已保存到: {save_path2}")
        plt.show()


def print_fuzzy_measures(model, n_features):
    """
    打印IVIE_FM模型的模糊测度值
    
    Args:
        model: IVIE_FM模型实例
        n_features: 特征数量
    """
    print("\n" + "="*80)
    print("模糊测度值 (Fuzzy Measure Values)")
    print("="*80)
    
    # 获取模糊测度
    model.eval()
    with torch.no_grad():
        FM = model.ivie_nn_vars(model.vars)
        FM_values = FM.cpu().numpy().flatten()
    
    # 创建特征组合到测度值的映射
    measure_dict = {}
    
    # 遍历所有可能的特征组合（二进制掩码）
    for mask in range(1, 2**n_features):
        # 检查这个掩码的阶数是否在允许范围内
        order = bin(mask).count('1')
        if model.add is not None and order > model.add:
            continue
        
        # 找到FM中对应的索引
        if model.add < model.columns_num:
            # 当使用additivity_order限制时
            valid_masks = []
            for m in range(1, 2**n_features):
                if bin(m).count('1') <= model.add:
                    valid_masks.append(m)
            
            if mask in valid_masks:
                fm_idx = valid_masks.index(mask)
            else:
                continue
        else:
            fm_idx = mask - 1  # 直接映射
        
        # 生成特征组合的可读表示
        features_in_mask = []
        for i in range(n_features):
            if mask & (1 << i):
                features_in_mask.append(f"特征{i}")
        
        feature_combo = "{" + ", ".join(features_in_mask) + "}"
        measure_value = FM_values[fm_idx]
        measure_dict[feature_combo] = measure_value
    
    # 按测度值降序排序并打印
    sorted_measures = sorted(measure_dict.items(), key=lambda x: x[1], reverse=True)
    
    for feature_combo, measure_value in sorted_measures:
        print(f"{feature_combo}: {measure_value:.6f}")
    
    print("="*80)


def print_sample_predictions(models_dict, X_test, y_test, n_samples=5):
    """
    打印测试集前n个样本的预测结果对比
    
    Args:
        models_dict: 模型字典 {model_name: {'model': model, 'type': type}}
        X_test: 测试集特征
        y_test: 测试集标签
        n_samples: 要显示的样本数量
    """
    print("\n" + "="*80)
    print(f"测试集前 {n_samples} 个样本的预测结果对比")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 获取前n个样本
    X_sample = X_test[:n_samples].to(device)
    y_sample = y_test[:n_samples]
    
    # 存储所有模型的预测结果
    predictions = {}
    
    for model_name, model_info in models_dict.items():
        model = model_info['model'].to(device)
        model_type = model_info['type']
        
        # 创建对应的损失函数来计算Hausdorff距离
        if model_type == 'FM':
            from IVIE_FM.iv_loss import HausdorffIntervalLoss as HausdorffLoss_FM
            criterion = HausdorffLoss_FM()
        else:
            from IVIE_Moebius.iv_loss import HausdorffIntervalLoss as HausdorffLoss_Moebius
            criterion = HausdorffLoss_Moebius()
        
        model.eval()
        with torch.no_grad():
            pred_l, pred_u = model(X_sample)
            _, hausdorff = criterion(pred_l, pred_u, y_sample.to(device))
        
        predictions[model_name] = {
            'pred_l': pred_l.cpu().numpy().flatten(),
            'pred_u': pred_u.cpu().numpy().flatten(),
            'hausdorff': hausdorff.cpu().numpy().flatten()
        }
    
    # 真实值
    true_l = y_sample[:, 0].numpy()
    true_u = y_sample[:, 1].numpy()
    
    # 打印每个样本的结果
    for i in range(n_samples):
        print(f"\n样本 {i+1}:")
        print(f"  真实值: [{true_l[i]:.6f}, {true_u[i]:.6f}]")
        print(f"  {'模型':<30} {'预测下界':<12} {'预测上界':<12} {'Hausdorff距离':<12}")
        print(f"  {'-'*70}")
        
        for model_name, pred in predictions.items():
            print(f"  {model_name:<30} {pred['pred_l'][i]:<12.6f} {pred['pred_u'][i]:<12.6f} {pred['hausdorff'][i]:<12.6f}")
    
    print("\n" + "="*80)


def print_comparison_table(errors_dict, history_dict):
    """打印性能对比表格"""
    print("\n" + "="*80)
    print("性能对比统计表")
    print("="*80)
    
    results = []
    for model_name in errors_dict.keys():
        errors = errors_dict[model_name]
        history = history_dict[model_name]
        
        results.append({
            'Model': model_name,
            'Mean Error': f"{np.mean(errors):.6f}",
            'Std Error': f"{np.std(errors):.6f}",
            'Max Error': f"{np.max(errors):.6f}",
            'Accuracy@0.1': f"{np.mean(errors <= 0.1):.2%}",
            'Final Val Loss': f"{history['final_val_loss']:.6f}"
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*80)


def save_mse_records(records):
    """将新增的 MSE 记录合并写入统一表格"""
    if not records:
        return

    new_df = pd.DataFrame(records)
    if MSE_RESULT_PATH.exists():
        existing = pd.read_excel(MSE_RESULT_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["dataset", "experiment", "model"], keep="last", inplace=True)
    else:
        combined = new_df

    combined.to_excel(MSE_RESULT_PATH, index=False)
    print(f"MSE 结果已更新: {MSE_RESULT_PATH}")


def train_single_model(config, train_data, test_data, n_features, epochs=30, progress_line=None):
    """
    训练单个模型的函数（用于并行执行）
    
    Args:
        config: 模型配置字典
        train_data: 训练数据元组 (X_train, y_train)
        test_data: 测试数据元组 (X_test, y_test)
        n_features: 特征数量
        epochs: 训练轮数
        progress_line: 进度条所在行号（用于多行并行显示）
    
    Returns:
        包含模型名称、训练历史、误差的字典
    """
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 解包数据
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # 在子进程中创建DataLoader（DataLoader无法pickle序列化传递）
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    # batch_size = max(1, len(train_dataset)//5)
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model_name = config['name']
    start_time = time.time()
    
    # 创建模型
    if config['type'] == 'IVCHI':
        model = IVCHI(
            feature_size=n_features,
            additivity_order=config['additivity_order'],
            op=config['op'],
            alpha=config['alpha'],
            beta=config['beta'],
            device=device
        )
        criterion = HausdorffLoss_FM()  # IVCHI使用与FM相同的损失函数
    elif config['type'] == 'FM':
        model = IE_FM(
            feature_size=n_features,
            additivity_order=config['additivity_order'],
            op=config['op'],
            alpha=config['alpha'],
            beta=config['beta'],
            device=device
        )
        criterion = HausdorffLoss_FM()
    else:  # Moebius
        model = IE_Moebius(
            feature_size=n_features,
            additivity_order=config['additivity_order'],
            op=config['op'],
            alpha=config['alpha'],
            beta=config['beta'],
            fuzzy_measure=config['fuzzy_measure']
        )
        criterion = HausdorffLoss_Moebius()
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 降低学习率
    
    # 训练模型 - 传入模型名称和行号用于多行进度显示，启用早停机制
    val_loss = model.fit_and_valid(
        train_Loader=train_loader,
        test_Loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        check_gradient=True,
        gradient_clip=0.5,
        model_name=model_name,
        progress_line=progress_line,
        early_stopping=True,  # 启用早停
        patience=30           # 30个epoch验证损失不改善就停止
    )
    
    # 评估模型 - 使用criterion的hausdorff距离（第二个返回值）
    X_test_device = X_test.to(device)
    y_test_device = y_test.to(device)
    
    model.eval()
    with torch.no_grad():
        pred_l, pred_u = model(X_test_device)
        # 使用损失函数返回的hausdorff距离，而不是total_loss
        _, hausdorff_distance = criterion(pred_l, pred_u, y_test_device)
        errors = hausdorff_distance
        mse_value = torch.mean(
            ((pred_l - y_test_device[:, 0]) ** 2 + (pred_u - y_test_device[:, 1]) ** 2) / 2
        ).item()

    errors_np = errors.cpu().numpy().flatten()
    
    elapsed_time = time.time() - start_time
    print(f"[{model_name}] 训练完成! 耗时: {elapsed_time:.2f}秒, 平均误差: {np.mean(errors_np):.6f}")
    
    return {
        'name': model_name,
        'model': model,  # 返回模型实例
        'model_type': config['type'],  # 返回模型类型
        'history': {
            'train_loss': model.train_loss_list,
            'val_loss': model.val_loss_list,
            'final_val_loss': val_loss
        },
        'errors': errors_np,
        'mse': mse_value
    }


def run_dataset_experiment(data_name: str, data_id: int, epochs: int = 3000):
    print("\n" + "="*80)
    print(f"IVIE_FM vs IVIE_Moebius 框架性能对比 - 数据集: {data_name} (id={data_id})")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    print("加载数据集...")
    X_train, X_test, y_train, y_test = generate_data(data_id)

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)

    n_features = X_train_tensor.shape[1] // 2
    add_order = 3

    print(f"训练集大小: {len(X_train_tensor)} | 测试集大小: {len(X_test_tensor)} | 特征数量: {n_features}")

    configs = [
        {
            'name': 'IVCHI_FM',
            'type': 'IVCHI',
            'additivity_order': add_order,
            'op': 'Min_interval',
            'alpha': 0.5,
            'beta': 0.0
        },
        {
            'name': 'IVIE_FM (Algebraic)',
            'type': 'FM',
            'additivity_order': add_order,
            'op': 'Algebraic_interval',
            'alpha': 0.5,
            'beta': 0.0
        },
        {
            'name': 'IVCHI_Moebius',
            'type': 'Moebius',
            'additivity_order': add_order,
            'op': 'Min_interval',
            'alpha': 0.5,
            'beta': 0.0,
            'fuzzy_measure': 'OutputLayer_single'
        },
        {
            'name': 'IVIE_Moebius (Algebraic)',
            'type': 'Moebius',
            'additivity_order': add_order,
            'op': 'Algebraic_interval',
            'alpha': 0.5,
            'beta': 0.0,
            'fuzzy_measure': 'OutputLayer_single'
        }
    ]

    history_dict = {}
    errors_dict = {}
    models_dict = {}
    mse_records = []

    train_data = (X_train_tensor, y_train_tensor)
    test_data = (X_test_tensor, y_test_tensor)

    print(f"\n{'='*80}")
    print(f"使用并行训练 - 同时训练 {len(configs)} 个模型")
    print(f"CPU核心数: {mp.cpu_count()}")
    print(f"{'='*80}\n")

    # 为进度预留行
    for _ in range(len(configs)):
        print()

    total_start_time = time.time()

    with ProcessPoolExecutor(max_workers=min(len(configs), mp.cpu_count())) as executor:
        futures = {}
        for idx, cfg in enumerate(configs):
            future = executor.submit(train_single_model, cfg, train_data, test_data, n_features, epochs, idx)
            futures[future] = cfg['name']

        for future in as_completed(futures):
            result = future.result()
            model_name = result['name']
            history_dict[model_name] = result['history']
            errors_dict[model_name] = result['errors']
            models_dict[model_name] = {
                'model': result['model'],
                'type': result['model_type']
            }
            mse_records.append({
                'dataset': data_name,
                'experiment': '不同结构对比',
                'model': result['name'],
                'mse': result['mse']
            })

    total_elapsed = time.time() - total_start_time
    print(f"\n并行训练总耗时: {total_elapsed:.2f}秒")

    rec_path = FIG_DIR / f"{data_name}_不同结构对比.png"
    training_path = FIG_DIR / f"{data_name}_不同结构对比_training.png"

    print("\n生成可视化图表...")
    # plot_training_curves(history_dict, str(training_path))
    plot_rec_curve(errors_dict, str(rec_path))

    # print_comparison_table(errors_dict, history_dict)
    # print_sample_predictions(models_dict, X_test_tensor, y_test_tensor, n_samples=5)

    print("\n" + "="*80)
    print("对比完成！")
    print("="*80)

    return mse_records


def main():
    parser = argparse.ArgumentParser(description="IVIE_FM vs IVIE_Moebius 多数据集对比")
    parser.add_argument('--epochs', type=int, default=1500, help='训练轮数 (默认 1500)')
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data_index = pd.read_excel(DATA_INDEX_PATH)
    all_records = []

    for _, row in data_index.iterrows():
        data_name = str(row['data_name'])
        data_id = int(row['data_id'])
        all_records.extend(run_dataset_experiment(data_name, data_id, epochs=args.epochs))

    save_mse_records(all_records)


if __name__ == "__main__":
    main()
