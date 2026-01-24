"""
集成学习神经网络：EnsembleIE
结构：输入 -> m个子模型(IVCHI/IVIE_Moebius) -> 集成层(IVIE) -> 输出

子模型可以是：
- IVCHI (from IVCHI.ivchi)
- IVIE_Moebius with Min_interval (from IVIE_Moebius.ieinn)

集成层可以是：
- IVIE_FM (from IVIE_FM.ivie)
- IVIE_Moebius (from IVIE_Moebius.ieinn)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IVCHI.ivchi import IVCHI
from IVIE_FM.ivie import IE as IE_FM
from IVIE_Moebius.ieinn import IE as IE_Moebius


class EnsembleIE(nn.Module):
    """
    集成学习神经网络
    
    参数:
        feature_size: 输入特征维度
        num_base_models: 子模型数量 m
        base_model_type: 子模型类型 'IVCHI' 或 'IVIE_Moebius'
        base_model_configs: 子模型配置列表，每个元素是一个字典，包含:
            - additivity_order: 加性阶数
            - alpha: alpha参数
            - beta: beta参数
        ensemble_type: 集成层类型 'FM' 或 'Moebius'
        ensemble_config: 集成层配置字典，包含:
            - additivity_order: 加性阶数
            - op: 操作类型 ('Algebraic_interval' 或 'Min_interval')
            - alpha: alpha参数 (仅用于Min_interval)
            - beta: beta参数 (仅用于Min_interval)
            - fuzzy_measure: 模糊测度类型 (仅用于Moebius)
        device: 设备类型
    """
    
    def __init__(self, 
                 feature_size,
                 num_base_models=3,
                 base_model_type='IVCHI',
                 base_model_configs=None,
                 ensemble_type='FM',
                 ensemble_config=None,
                 device='cuda'):
        super(EnsembleIE, self).__init__()
        
        self.feature_size = feature_size
        self.num_base_models = num_base_models
        self.base_model_type = base_model_type
        self.ensemble_type = ensemble_type
        self.device = device
        
        # 默认配置
        if base_model_configs is None:
            base_model_configs = [
                {
                    'additivity_order': 2,
                    'alpha': 0.5,
                    'beta': 0.0
                }
            ] * num_base_models
        
        # 如果只提供了一个配置，复制给所有子模型
        if len(base_model_configs) == 1 and num_base_models > 1:
            base_model_configs = base_model_configs * num_base_models
        
        if len(base_model_configs) != num_base_models:
            raise ValueError(f"base_model_configs长度({len(base_model_configs)})必须等于num_base_models({num_base_models})")
        
        if ensemble_config is None:
            ensemble_config = {
                'additivity_order': 2,
                'op': 'Min_interval',
                'alpha': 0.5,
                'beta': 0.0
            }
            if ensemble_type == 'Moebius':
                ensemble_config['fuzzy_measure'] = 'OutputLayer_single'
        
        # 创建子模型
        self.base_models = nn.ModuleList()
        for i, config in enumerate(base_model_configs):
            if base_model_type == 'IVCHI':
                model = IVCHI(
                    feature_size=feature_size,
                    additivity_order=config.get('additivity_order', 2),
                    op='Min_interval',  # IVCHI固定使用Min_interval
                    alpha=config.get('alpha', 0.5),
                    beta=config.get('beta', 0.0),
                    device=device
                )
            elif base_model_type == 'IVIE_Moebius':
                model = IE_Moebius(
                    feature_size=feature_size,
                    additivity_order=config.get('additivity_order', 2),
                    op='Min_interval',  # 子模型使用Min_interval
                    alpha=config.get('alpha', 0.5),
                    beta=config.get('beta', 0.0),
                    fuzzy_measure='OutputLayer_single'  # 默认使用单输出层
                )
                # IVIE_Moebius没有device参数，需要手动移动到设备
                model = model.to(device)
            else:
                raise ValueError(f"不支持的子模型类型: {base_model_type}. 请使用'IVCHI'或'IVIE_Moebius'")
            
            self.base_models.append(model)
        
        # 创建集成层
        # 集成层的输入特征数 = num_base_models（每个子模型输出一个区间）
        # 确保 additivity_order 不超过 num_base_models
        ensemble_additivity = min(
            ensemble_config.get('additivity_order', 2),
            num_base_models
        )
        
        if ensemble_type == 'FM':
            self.ensemble_model = IE_FM(
                feature_size=num_base_models,
                additivity_order=ensemble_additivity,
                op=ensemble_config.get('op', 'Min_interval'),
                alpha=ensemble_config.get('alpha', 0.5),
                beta=ensemble_config.get('beta', 0.0),
                device=device
            )
        elif ensemble_type == 'Moebius':
            self.ensemble_model = IE_Moebius(
                feature_size=num_base_models,
                additivity_order=ensemble_additivity,
                op=ensemble_config.get('op', 'Min_interval'),
                alpha=ensemble_config.get('alpha', 0.5),
                beta=ensemble_config.get('beta', 0.0),
                fuzzy_measure=ensemble_config.get('fuzzy_measure', 'OutputLayer_single')
            )
            # IVIE_Moebius没有device参数，需要手动移动到设备
            self.ensemble_model = self.ensemble_model.to(device)
        else:
            raise ValueError(f"不支持的集成层类型: {ensemble_type}. 请使用'FM'或'Moebius'")
        
        # 用于记录训练历史
        self.train_loss_list = []
        self.val_loss_list = []
        self.lrs_list = []
        self.gradient_info = []
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, 2*feature_size)
               前feature_size列是区间下界，后feature_size列是区间上界
        
        Returns:
            pred_l: 预测区间下界，形状 (batch_size, 1)
            pred_u: 预测区间上界，形状 (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # 收集所有子模型的输出
        base_outputs_l = []
        base_outputs_u = []
        
        for model in self.base_models:
            out_l, out_u = model(x)
            base_outputs_l.append(out_l)
            base_outputs_u.append(out_u)
        
        # 将子模型输出拼接成集成层的输入
        # 形状: (batch_size, num_base_models)
        ensemble_input_l = torch.cat(base_outputs_l, dim=1)
        ensemble_input_u = torch.cat(base_outputs_u, dim=1)
        
        # 拼接成集成层所需的格式: (batch_size, 2*num_base_models)
        # 前num_base_models列是下界，后num_base_models列是上界
        ensemble_input = torch.cat([ensemble_input_l, ensemble_input_u], dim=1)
        
        # 通过集成层
        final_l, final_u = self.ensemble_model(ensemble_input)
        
        return final_l, final_u
    
    def fit_and_valid(self, train_Loader, test_Loader, criterion, optimizer, 
                     device='cuda', epochs=100, check_gradient=True, 
                     gradient_clip=1.0, model_name=None, progress_line=None,
                     early_stopping=True, patience=100):
        """
        训练和验证模型
        
        Args:
            train_Loader: 训练数据加载器
            test_Loader: 测试数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 设备
            epochs: 训练轮数
            check_gradient: 是否检查梯度
            gradient_clip: 梯度裁剪阈值
            model_name: 模型名称（用于多行进度显示）
            progress_line: 进度条所在行号（0-based，用于多行并行训练）
            early_stopping: 是否启用早停
            patience: 早停的耐心值（验证损失不改善的最大epoch数）
        
        Returns:
            final_val_loss: 最终验证损失
        """
        start = time.time()
        self.train_loss_list = []
        self.val_loss_list = []
        self.lrs_list = []
        self.gradient_info = []  # 记录梯度信息
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            epoch_grad_norm = 0
            epoch_grad_max = 0
            epoch_grad_min = float('inf')
            batch_count = 0
            has_nan_grad = False
            
            # 训练阶段
            self.train()
            for i, (images, labels) in enumerate(train_Loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputsl, outputsu = self(images)
                loss, error = criterion(outputsl, outputsu, labels)
                train_loss += loss.item() * len(labels)
                
                # 检查loss是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: Epoch {epoch+1}, Batch {i+1} - Loss为NaN或Inf!")
                    continue
                
                loss.backward()
                
                # 梯度检查
                if check_gradient:
                    total_grad_norm = 0
                    total_grad_max = 0
                    total_grad_min = float('inf')
                    has_grad = False
                    
                    for param in self.parameters():
                        if param.grad is not None:
                            has_grad = True
                            grad = param.grad
                            grad_norm = grad.norm().item()
                            grad_max = grad.abs().max().item()
                            grad_min = grad.abs().min().item()
                            
                            total_grad_norm += grad_norm
                            total_grad_max = max(total_grad_max, grad_max)
                            total_grad_min = min(total_grad_min, grad_min)
                            
                            # 检查NaN梯度
                            if torch.isnan(grad).any() or torch.isinf(grad).any():
                                has_nan_grad = True
                                print(f"警告: Epoch {epoch+1}, Batch {i+1} - 梯度包含NaN或Inf!")
                                # 将NaN梯度置零
                                param.grad = torch.where(
                                    torch.isnan(grad) | torch.isinf(grad),
                                    torch.zeros_like(grad),
                                    grad
                                )
                    
                    if has_grad:
                        epoch_grad_norm += total_grad_norm
                        epoch_grad_max = max(epoch_grad_max, total_grad_max)
                        epoch_grad_min = min(epoch_grad_min, total_grad_min)
                        batch_count += 1
                
                # 梯度裁剪
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
                
                optimizer.step()
                
                # 记录学习率
                current_lr = optimizer.param_groups[0]["lr"]
                self.lrs_list.append(current_lr)
            
            avg_train_loss = train_loss / len(train_Loader.dataset)
            
            # 记录梯度统计信息
            if batch_count > 0:
                avg_grad_norm = epoch_grad_norm / batch_count
                self.gradient_info.append({
                    'epoch': epoch + 1,
                    'avg_grad_norm': avg_grad_norm,
                    'max_grad': epoch_grad_max,
                    'min_grad': epoch_grad_min,
                    'has_nan': has_nan_grad
                })
            
            # 验证阶段
            self.eval()
            with torch.no_grad():
                for images, labels in test_Loader:
                    images, labels = images.to(device), labels.to(device)
                    outputsl, outputsu = self(images)
                    loss, error = criterion(outputsl, outputsu, labels)
                    val_loss += loss.item() * len(labels)
            
            avg_val_loss = val_loss / len(test_Loader.dataset)
            
            # 打印训练信息（支持多行并行显示）
            progress = (epoch + 1) / epochs
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # 计算时间
            elapsed_time = time.time() - start
            if epoch > 0:
                avg_time_per_epoch = elapsed_time / (epoch + 1)
                eta = avg_time_per_epoch * (epochs - epoch - 1)
                time_str = f"{elapsed_time:.0f}s < {eta:.0f}s"
            else:
                time_str = f"{elapsed_time:.0f}s"
            
            # 构建输出信息
            if model_name:
                model_prefix = f"{model_name:<35}"
            else:
                model_prefix = ""
            
            if check_gradient and batch_count > 0:
                output = f'{model_prefix}[{bar}] {epoch+1}/{epochs} | loss:{avg_train_loss:.4f}/{avg_val_loss:.4f} | grad:{avg_grad_norm:.3f} | {time_str}'
            else:
                output = f'{model_prefix}[{bar}] {epoch+1}/{epochs} | loss:{avg_train_loss:.4f}/{avg_val_loss:.4f} | {time_str}'
            
            # 如果指定了行号，使用ANSI转义码定位到特定行
            if progress_line is not None:
                # 保存当前光标位置，移动到指定行，打印，恢复光标位置
                print(f'\033[s\033[{progress_line+1};0H{output}\033[K\033[u', end='', flush=True)
            else:
                # 单行刷新
                print(f'\r{output}', end='', flush=True)
            
            self.train_loss_list.append(avg_train_loss)
            self.val_loss_list.append(avg_val_loss)
            
            # 早停检查
            if early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if progress_line is not None:
                        early_stop_msg = f"{model_name:<35}⚠ Early stopping at epoch {epoch+1}/{epochs}"
                        print(f'\033[s\033[{progress_line+1};0H{early_stop_msg}\033[K\033[u', flush=True)
                    else:
                        print(f"\n早停触发于 epoch {epoch + 1}")
                    
                    # 恢复最佳模型
                    self.load_state_dict(best_model_state)
                    break
        
        # 训练结束，如果是多行模式，移动到指定行打印完成信息
        if progress_line is not None:
            final_msg = f"{model_name:<35}✓ Completed! Time: {time.time() - start:.1f}s"
            print(f'\033[s\033[{progress_line+1};0H{final_msg}\033[K\033[u', flush=True)
        else:
            print(f"\n\nTraining completed! Total time: {time.time() - start:.2f}s")
        
        return best_val_loss if early_stopping else avg_val_loss
    
    def _print_gradient_report(self):
        """打印梯度诊断报告"""
        if len(self.gradient_info) == 0:
            print("无梯度信息可供分析")
            return
        
        print("\n" + "="*60)
        print("梯度诊断报告")
        print("="*60)
        
        avg_norms = [info['avg_grad_norm'] for info in self.gradient_info]
        max_grads = [info['max_grad'] for info in self.gradient_info]
        min_grads = [info['min_grad'] for info in self.gradient_info]
        nan_count = sum(1 for info in self.gradient_info if info['has_nan'])
        
        print(f"总Epochs: {len(self.gradient_info)}")
        print(f"平均梯度范数: {np.mean(avg_norms):.6f}")
        print(f"梯度范数范围: [{np.min(avg_norms):.6f}, {np.max(avg_norms):.6f}]")
        print(f"最大梯度值: {np.max(max_grads):.6f}")
        print(f"最小梯度值: {np.min(min_grads):.10f}")
        print(f"包含NaN的Epoch数: {nan_count}")
        
        # 诊断问题
        if nan_count > 0:
            print("\n⚠️ 警告: 存在NaN梯度，可能原因:")
            print("  - 学习率过大")
            print("  - 数值溢出")
            print("  - 除零操作")
        
        if np.mean(avg_norms) < 1e-6:
            print("\n⚠️ 警告: 梯度过小（梯度消失），可能原因:")
            print("  - torch.abs()在0附近梯度不稳定")
            print("  - torch.min()可能阻断梯度")
            print("  - 网络层数过深")
        
        if np.max(max_grads) > 100:
            print("\n⚠️ 警告: 梯度过大（梯度爆炸），建议:")
            print("  - 减小学习率")
            print("  - 增加梯度裁剪")
        
        print("="*60)
    
    def plot_training_history(self, save_path=None, show=True):
        """
        可视化训练历史
        
        Args:
            save_path: 保存图片的路径，如果为None则不保存
            show: 是否显示图片
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 训练和验证损失
        ax1 = axes[0, 0]
        epochs = range(1, len(self.train_loss_list) + 1)
        ax1.plot(epochs, self.train_loss_list, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_loss_list, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. 学习率变化
        ax2 = axes[0, 1]
        if hasattr(self, 'lrs_list') and len(self.lrs_list) > 0:
            ax2.plot(self.lrs_list, 'g-', linewidth=2)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No LR data available', 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        # 3. 梯度范数变化
        ax3 = axes[1, 0]
        if hasattr(self, 'gradient_info') and len(self.gradient_info) > 0:
            grad_epochs = [info['epoch'] for info in self.gradient_info]
            avg_grad_norms = [info['avg_grad_norm'] for info in self.gradient_info]
            ax3.plot(grad_epochs, avg_grad_norms, 'purple', linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Average Gradient Norm', fontsize=12)
            ax3.set_title('Gradient Norm Over Time', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, 'No gradient data available', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Gradient Norm Over Time', fontsize=14, fontweight='bold')
        
        # 4. 损失对比（对数坐标）
        ax4 = axes[1, 1]
        ax4.semilogy(epochs, self.train_loss_list, 'b-', label='Train Loss', linewidth=2)
        ax4.semilogy(epochs, self.val_loss_list, 'r-', label='Val Loss', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss (log scale)', fontsize=12)
        ax4.set_title('Loss (Logarithmic Scale)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存至: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def get_base_model_outputs(self, x):
        """
        获取所有子模型的输出（用于调试和分析）
        
        Args:
            x: 输入张量
        
        Returns:
            outputs: 列表，包含每个子模型的输出元组 (out_l, out_u)
        """
        outputs = []
        self.eval()
        with torch.no_grad():
            for model in self.base_models:
                out_l, out_u = model(x)
                outputs.append((out_l, out_u))
        return outputs

    # ====== 训练/微调辅助 ======
    def freeze_base_models(self):
        """冻结子模型参数，用于仅训练集成层。"""
        for param in self.base_models.parameters():
            param.requires_grad = False

    def unfreeze_base_models(self):
        """解冻子模型参数。"""
        for param in self.base_models.parameters():
            param.requires_grad = True

    def freeze_ensemble(self):
        """冻结集成层参数，用于先行预训练子模型。"""
        for param in self.ensemble_model.parameters():
            param.requires_grad = False

    def unfreeze_ensemble(self):
        """解冻集成层参数。"""
        for param in self.ensemble_model.parameters():
            param.requires_grad = True

    def base_parameters(self):
        """便于单独给子模型创建优化器。"""
        return self.base_models.parameters()

    def ensemble_parameters(self):
        """便于单独给集成层创建优化器。"""
        return self.ensemble_model.parameters()
