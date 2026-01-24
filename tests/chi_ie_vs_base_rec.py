"""
CHI_IE 集成 vs 单模型性能对比（UCI Auto MPG）
- 数据：data_build.generate_data()
- CHI_IE：3 个 IVIE_Moebius 子模型（op=Min_interval），(alpha, beta) 为
  (1.0, 0.0)、(0.0, 1.0)、(0.5, 0.3)，集成层 IVIE_FM（op=Algebraic_interval）
- 单模型：3 个 IVIE_Moebius（op=Min_interval），(alpha, beta) 同上
- 可视化：REC 曲线，复用 tests/compare_frameworks.py 中的可视化接口
"""
import sys
from pathlib import Path
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_INDEX_PATH = _PROJECT_ROOT / "data_test" / "data_index.xlsx"
MSE_RESULT_PATH = _PROJECT_ROOT / "data_test" / "mse_results.xlsx"
FIG_DIR = _PROJECT_ROOT / "figs"

from data_build import generate_data
from CHI_IE.ensemble_ie import EnsembleIE
from IVIE_Moebius.ieinn import IE as IE_Moebius
from IVIE_Moebius.iv_loss import HausdorffIntervalLoss as HausdorffLoss_Moebius
from IVIE_FM.iv_loss import HausdorffIntervalLoss as HausdorffLoss_FM

# 复用现有对比与可视化函数
from tests.compare_frameworks import plot_rec_curve, print_comparison_table, train_single_model
import argparse


def save_mse_records(records):
    """将 MSE 结果合并写入统一表格"""
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


def train_ensemble_ie(X_train_t, y_train_t, X_test_t, y_test_t, n_features, epochs=3000):
    """训练 CHI_IE 集成模型并返回误差与训练历史"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    add_order = min(3, n_features)

    # 子模型配置（additivity_order 设为不受限：使用全部特征阶）
    base_model_configs = [
        {'additivity_order': add_order, 'alpha': 1.0, 'beta': 0.0},
        {'additivity_order': add_order, 'alpha': 0.0, 'beta': 1.0},
        {'additivity_order': add_order, 'alpha': 0.5, 'beta': 0.3},
    ]

    # 集成层配置（Algebraic_interval）
    ensemble_config = {
        'additivity_order': add_order,  # 不限制：等于子模型数量
        'op': 'Algebraic_interval'
    }

    model = EnsembleIE(
        feature_size=n_features,
        num_base_models=3,
        base_model_type='IVIE_Moebius',
        base_model_configs=base_model_configs,
        ensemble_type='Moebius',
        ensemble_config=ensemble_config,
        device=device,
    )

    criterion = HausdorffLoss_FM()  # 使用FM版本Hausdorff距离作为整体损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    batch_size = max(1, len(train_dataset) // 10)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    start = time.time()
    final_val_loss = model.fit_and_valid(
        train_Loader=train_loader,
        test_Loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        check_gradient=True,
        gradient_clip=0.5,
        model_name='CHI_IE Ensemble',
        progress_line=None,
        early_stopping=True,
        patience=30,
    )

    # 评估：使用Hausdorff距离作为误差
    model.eval()
    with torch.no_grad():
        y_test_device = y_test_t.to(device)
        pred_l, pred_u = model(X_test_t.to(device))
        _, hausdorff = criterion(pred_l, pred_u, y_test_device)
        mse_value = torch.mean(
            ((pred_l - y_test_device[:, 0]) ** 2 + (pred_u - y_test_device[:, 1]) ** 2) / 2
        ).item()
        errors_np = hausdorff.cpu().numpy().flatten()

    elapsed = time.time() - start
    print(f"\n[CHI_IE Ensemble] 训练完成! 耗时: {elapsed:.2f}s, 平均误差: {np.mean(errors_np):.6f}")

    history = {
        'train_loss': model.train_loss_list,
        'val_loss': model.val_loss_list,
        'final_val_loss': final_val_loss,
    }
    return errors_np, history, mse_value


def run_dataset_experiment(data_name: str, data_id: int, epochs: int = 3000):
    print("\n" + "="*80)
    print(f"CHI_IE 集成 vs 单模型性能对比 - 数据集: {data_name} (id={data_id})")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    print("加载数据集...")
    X_train, X_test, y_train, y_test = generate_data(data_id)

    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.FloatTensor(y_train.values)
    X_test_t = torch.FloatTensor(X_test.values)
    y_test_t = torch.FloatTensor(y_test.values)

    n_features = X_train_t.shape[1] // 2
    add_order = min(3, n_features)
    print(f"训练样本: {len(X_train_t)} | 测试样本: {len(X_test_t)} | 特征数: {n_features}")

    ensemble_errors, ensemble_history, ensemble_mse = train_ensemble_ie(
        X_train_t, y_train_t, X_test_t, y_test_t, n_features, epochs=epochs
    )

    single_configs = [
        {
            'name': 'Moebius (alpha=1.0, beta=0.0)',
            'type': 'Moebius',
            'additivity_order': add_order,
            'op': 'Min_interval',
            'alpha': 1.0,
            'beta': 0.0,
            'fuzzy_measure': 'OutputLayer_single',
        },
        {
            'name': 'Moebius (alpha=0.0, beta=1.0)',
            'type': 'Moebius',
            'additivity_order': add_order,
            'op': 'Min_interval',
            'alpha': 0.0,
            'beta': 1.0,
            'fuzzy_measure': 'OutputLayer_single',
        },
        {
            'name': 'Moebius (alpha=0.5, beta=0.3)',
            'type': 'Moebius',
            'additivity_order': add_order,
            'op': 'Min_interval',
            'alpha': 0.5,
            'beta': 0.3,
            'fuzzy_measure': 'OutputLayer_single',
        },
    ]

    train_data = (X_train_t, y_train_t)
    test_data = (X_test_t, y_test_t)

    history_dict = {
        'CHI_IE Ensemble (FM-Algebraic)': ensemble_history,
    }
    errors_dict = {
        'CHI_IE Ensemble (FM-Algebraic)': ensemble_errors,
    }
    mse_records = [
        {
            'dataset': data_name,
            'experiment': '集成学习对比',
            'model': 'CHI_IE Ensemble (FM-Algebraic)',
            'mse': ensemble_mse,
        }
    ]

    for cfg in single_configs:
        result = train_single_model(cfg, train_data, test_data, n_features, epochs=epochs, progress_line=None)
        history_dict[result['name']] = result['history']
        errors_dict[result['name']] = result['errors']
        mse_records.append({
            'dataset': data_name,
            'experiment': '集成学习对比',
            'model': result['name'],
            'mse': result['mse'],
        })

    rec_path = FIG_DIR / f"{data_name}_集成学习对比.png"
    print("\n生成REC曲线...")
    plot_rec_curve(errors_dict, save_path=str(rec_path))

    print_comparison_table(errors_dict, history_dict)

    print("\n" + "="*80)
    print(f"对比完成！图像已保存到 {rec_path}")
    print("="*80)

    return mse_records


def main():
    parser = argparse.ArgumentParser(description='CHI_IE 集成 vs 单模型多数据集对比')
    parser.add_argument('--epochs', type=int, default=3000, help='训练轮数 (默认 3000)')
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data_index = pd.read_excel(DATA_INDEX_PATH)
    all_records = []

    for _, row in data_index.iterrows():
        data_name = str(row['data_name'])
        data_id = int(row['data_id'])
        all_records.extend(run_dataset_experiment(data_name, data_id, epochs=args.epochs))

    save_mse_records(all_records)


if __name__ == '__main__':
    main()
