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
                     early_stopping=True, patience=30):
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
            model_name: 模型名称
            progress_line: 进度条所在行号
            early_stopping: 是否启用早停
            patience: 早停的耐心值
        
        Returns:
            final_val_loss: 最终验证损失
        """
        start = time.time()
        self.train_loss_list = []
        self.val_loss_list = []
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            
            # 训练阶段
            self.train()
            for i, (images, labels) in enumerate(train_Loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputsl, outputsu = self(images)
                loss, error = criterion(outputsl, outputsu, labels)
                train_loss += loss.item() * len(labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: Epoch {epoch+1}, Batch {i+1} - Loss为NaN或Inf!")
                    continue
                
                loss.backward()
                
                # 梯度裁剪
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
                
                optimizer.step()
            
            train_loss = train_loss / len(train_Loader.dataset)
            self.train_loss_list.append(train_loss)
            
            # 验证阶段
            self.eval()
            with torch.no_grad():
                for images, labels in test_Loader:
                    images, labels = images.to(device), labels.to(device)
                    outputsl, outputsu = self(images)
                    loss, error = criterion(outputsl, outputsu, labels)
                    val_loss += loss.item() * len(labels)
            
            val_loss = val_loss / len(test_Loader.dataset)
            self.val_loss_list.append(val_loss)
            
            # 早停检查
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\n[{model_name}] 早停触发！在epoch {epoch+1}停止训练。")
                    # 恢复最佳模型
                    self.load_state_dict(best_model_state)
                    break
            
            # 打印进度（每10个epoch）
            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - start
                if model_name:
                    print(f"[{model_name}] Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                          f"Time: {elapsed:.2f}s")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        elapsed = time.time() - start
        print(f"\n训练完成！总耗时: {elapsed:.2f}秒")
        
        return val_loss if not early_stopping else best_val_loss
    
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
