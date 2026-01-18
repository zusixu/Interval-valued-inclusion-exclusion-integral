import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import numpy as np
from scipy.special import comb
from tqdm import tqdm
from . import narray_op,feature_layer


# Convert decimal to binary string
def sources_and_subsets_nodes(N):
    str1 = "{0:{fill}"+str(N)+"b}"
    a = []
    for i in range(1,2**N):
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))
    # find subset nodes of a node
    def node_subset(node, sourcesInNodes):
        return [node - 2**(i) for i in sourcesInNodes]
    
    # convert binary encoded string to integer list
    def string_to_integer_array(s, ch):
        N = len(s) 
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]
    
    for j in range(len(a)):
        # index from right to left
        idxLR = string_to_integer_array(a[j],'1')
        sourcesInNode.append(idxLR)  
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))
        subset.append(node_subset(j,idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices):
    return [i for i in indices]




class IE(nn.Module):
    def __init__(self, feature_size, additivity_order=None, op='Algebraic_interval', alpha=1, beta=0, device='cuda'):
        super(IE, self).__init__()
        self.add = additivity_order
        self.narray_op = op
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.error = torch.tensor((), device=device)
        self.columns_num = feature_size
        self.nVars = 2**self.columns_num - 2

        self.feature_matrix = feature_layer.FeatureMatrix(self.columns_num, device=device).build_sparse_matrix()
        
        # 优化后的初始化策略
        # 单个特征均匀初始化，特征组合随机初始化（保证单调性）
        dummy = self._init_vars_monotonic()
        self.vars = torch.nn.Parameter(dummy)
        
        # following function uses numpy vs pytorch
        self.sourcesInNode, self.subset = sources_and_subsets_nodes(self.columns_num)
        
        self.sourcesInNode = [torch.tensor(x) for x in self.sourcesInNode]
        self.subset = [torch.tensor(x) for x in self.subset]

        if self.add == None:
            self.add = self.columns_num

        if self.add > self.columns_num:
            raise IndexError('"additivity_order" must be less than the "number of features"')
        if self.narray_op not in ['Algebraic_interval', 'Min_interval']:  
            raise ValueError('narray_op / Algebraic_interval, Min_interval') 

        if self.narray_op == 'Min_interval':
            self.op = narray_op.Min_interval(self.add, self.alpha, self.beta)
        elif self.narray_op == 'Algebraic_interval':
            self.op = narray_op.Algebraic_interval(self.add)  
    
    def _init_vars_monotonic(self):
        """
        优化的初始化策略：
        - 单个特征：均匀初始化为 1/n
        - 特征组合：小随机值初始化（累加后自动满足单调性）
        
        vars的索引使用二进制掩码：
        - mask=1 (001) -> 特征0
        - mask=2 (010) -> 特征1
        - mask=3 (011) -> 特征0+1组合
        - mask=4 (100) -> 特征2
        - ...
        
        单个特征的mask是2的幂次：1, 2, 4, 8, ...
        """
        n = self.columns_num
        nVars = 2**n - 2  # 不包括空集和全集
        
        # 先创建不需要梯度的tensor进行初始化，确保在正确的设备上
        init_values = torch.zeros((nVars, 1), device=self.device)
        
        for mask in range(1, 2**n - 1):  # mask从1到2^n-2
            idx = mask - 1  # 转换为0-based索引
            order = bin(mask).count('1')  # 该子集包含的特征数量
            
            if order == 1:
                # 单个特征：均匀初始化
                init_values[idx, 0] = 1.0 / n
            else:
                # 特征组合：小随机值，这样累加后仍然接近子集的和
                # 使用较小的随机值作为增量
                init_values[idx, 0] = torch.rand(1).item() * 0.1 / n
        
        # 最后转换为需要梯度的tensor
        return init_values.requires_grad_(True)  
        

    def forward(self, x):
        self.FM = self.ivie_nn_vars(self.vars)

        columns_num = x.size()[1]
        columns_num = int(columns_num)
        index = columns_num / 2
        index = int(index)
        datal = x[:, :index]
        datau = x[:, index:]
        
        featuers_datal, featuers_datau = self.op(datal, datau)
        
        # 当 additivity_order < columns_num 时，需要调整 feature_matrix
        # op 生成的特征按位掩码顺序排列，数量由 additivity_order 决定
        actual_num_features = featuers_datal.size(1)
        expected_num_features = 2**self.columns_num - 1
        
        feature_matrix_dense = self.feature_matrix.to_dense()
        
        if actual_num_features < expected_num_features:
            # feature_matrix 的前 actual_num_features 行对应 op 生成的特征
            # 提取对应的列：每个特征对应完整的列空间的子集
            # 具体来说，需要提取前 actual_num_features 对应的列（每个特征2列）
            # feature_matrix 形状: (2^n-1, 2*(2^n-1))
            # 我们需要: (actual_num_features, 2*actual_num_features)
            
            # 提取前 actual_num_features 行和对应的列
            feature_matrix_to_use = feature_matrix_dense[:actual_num_features, :2*actual_num_features]
            
            # FM 也需要只保留前 actual_num_features 个（对应相同的特征组合）
            FM_to_use = self.FM[:actual_num_features, :]
        else:
            # 使用完整的 feature_matrix 和 FM（不包括全集，因为FM包含全集）
            # FM形状: (2^n, 1) 包含全集
            # feature_matrix形状: (2^n-1, 2*(2^n-1))
            # 去掉FM的最后一个元素（全集）以匹配feature_matrix的行数
            feature_matrix_to_use = feature_matrix_dense
            FM_to_use = self.FM[:expected_num_features, :]
        
        featuers_datal_mt = torch.matmul(featuers_datal, feature_matrix_to_use)
        featuers_datau_mt = torch.matmul(featuers_datau, feature_matrix_to_use)
        
        # 拆分左右端点
        a = torch.matmul(FM_to_use.T, featuers_datal_mt[:, ::2].T)  # 偶数列（左端点）
        b = torch.matmul(FM_to_use.T, featuers_datau_mt[:, ::2].T)  # 偶数列（左端点）
        c = torch.matmul(FM_to_use.T, featuers_datal_mt[:, 1::2].T) # 奇数列（右端点）
        d = torch.matmul(FM_to_use.T, featuers_datau_mt[:, 1::2].T) # 奇数列（右端点）
        
        # 区间减法
        left = torch.min(a - c, b - d)
        right = b - d

        # 转置为 (batch, 1) 格式
        return left.T, right.T
    
        # Converts NN-vars to FM vars
    def ivie_nn_vars(self, ivie_vars):
        ivie_vars = torch.abs(ivie_vars)
        
        # 当使用 additivity_order 时，只处理对应阶数的子集
        if self.add < self.columns_num:
            # 计算有效的掩码（阶数 <= additivity_order）
            valid_masks = []
            for mask in range(1, 2**self.columns_num):
                order = bin(mask).count('1')
                if order <= self.add:
                    valid_masks.append(mask - 1)  # 转换为0-based索引
            
            # 只使用有效掩码对应的vars
            num_valid = len(valid_masks)
            FM = ivie_vars[None, valid_masks[0], :]
            
            for idx in range(1, num_valid):
                var_idx = valid_masks[idx]
                indices = subset_to_indices(self.subset[var_idx])
                
                # 过滤indices，只保留在valid_masks中的
                valid_indices_in_FM = []
                for sub_idx in indices:
                    if sub_idx in valid_masks:
                        # 找到sub_idx在valid_masks中的位置
                        pos = valid_masks.index(sub_idx)
                        valid_indices_in_FM.append(pos)
                
                if len(valid_indices_in_FM) == 0:
                    # 没有有效的子集，直接使用当前值
                    FM = torch.cat((FM, ivie_vars[None, var_idx, :]), 0)
                elif len(valid_indices_in_FM) == 1:
                    FM = torch.cat((FM, ivie_vars[None, var_idx, :]), 0)
                else:
                    maxVal, _ = torch.max(FM[valid_indices_in_FM, :], 0)
                    temp = torch.add(maxVal, ivie_vars[var_idx, :])
                    FM = torch.cat((FM, temp[None, :]), 0)
            
            FM = torch.cat([FM, torch.ones((1, 1), device=self.device)], 0)
            FM = torch.min(FM, torch.ones(1, device=self.device))
            
        else:
            # 原始逻辑：使用所有vars
            FM = ivie_vars[None, 0, :]
            for i in range(1, self.nVars):
                indices = subset_to_indices(self.subset[i])
                if (len(indices) == 1):
                    FM = torch.cat((FM, ivie_vars[None, i, :]), 0)
                else:
                    maxVal, _ = torch.max(FM[indices, :], 0)
                    temp = torch.add(maxVal, ivie_vars[i, :])
                    FM = torch.cat((FM, temp[None, :]), 0)
                  
            FM = torch.cat([FM, torch.ones((1, 1), device=self.device)], 0)
            FM = torch.min(FM, torch.ones(1, device=self.device))
        
        return FM

    def fit_and_valid(self, train_Loader, test_Loader, criterion, optimizer, device='cuda', epochs=100, 
                       check_gradient=True, gradient_clip=1.0, model_name=None, progress_line=None,
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
            model_name: 模型名称（用于多行进度显示）
            progress_line: 进度条所在行号（0-based，用于多行并行训练）
            early_stopping: 是否启用早停
            patience: 早停的耐心值（验证损失不改善的最大epoch数）
        """
        start = time.time()
        self.train_loss_list = []
        self.val_loss_list = []
        self.lrs_list = []
        self.gradient_info = []  # 记录梯度信息
        
        # 初始化error张量到正确设备
        self.error = torch.tensor((), device=device)
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.train_loss = 0
            self.val_loss = 0
            epoch_grad_norm = 0
            epoch_grad_max = 0
            epoch_grad_min = float('inf')
            batch_count = 0
            has_nan_grad = False

            # train
            self.train()
            for i, (images, labels) in enumerate(train_Loader):
                images, labels = images.to(device), labels.to(device)

                # Zero your gradients for every batch
                optimizer.zero_grad()
                # Make predictions for this batch
                outputsl, outputsu = self(images)
                # Compute the loss
                loss, error = criterion(outputsl, outputsu, labels)
                self.train_loss += loss.item() * len(labels)
                
                # 检查loss是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: Epoch {epoch+1}, Batch {i+1} - Loss为NaN或Inf!")
                    continue
                
                # Compute the its gradients
                loss.backward()
                
                # 梯度检查
                if check_gradient and self.vars.grad is not None:
                    grad = self.vars.grad
                    grad_norm = grad.norm().item()
                    grad_max = grad.abs().max().item()
                    grad_min = grad.abs().min().item()
                    
                    epoch_grad_norm += grad_norm
                    epoch_grad_max = max(epoch_grad_max, grad_max)
                    epoch_grad_min = min(epoch_grad_min, grad_min)
                    batch_count += 1
                    
                    # 检查NaN梯度
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        has_nan_grad = True
                        print(f"警告: Epoch {epoch+1}, Batch {i+1} - 梯度包含NaN或Inf!")
                        # 将NaN梯度置零
                        self.vars.grad = torch.where(
                            torch.isnan(grad) | torch.isinf(grad),
                            torch.zeros_like(grad),
                            grad
                        )
                
                # 梯度裁剪
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
                
                # Adjust learning weights
                optimizer.step()
                self.lrs = optimizer.param_groups[0]["lr"]
                self.lrs_list.append(optimizer.param_groups[0]["lr"])

            avg_train_loss = self.train_loss / len(train_Loader.dataset)
            
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

            self.error_max = torch.tensor(0, device=device)
            # val 在测试集上的损失
            self.eval()
            with torch.no_grad():
                for images, labels in test_Loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputsl, outputsu = self(images)
                    loss, distance = criterion(outputsl, outputsu, labels)
                    self.error = torch.cat((self.error, distance), dim=0)

                    self.val_loss += loss.item() * len(labels)
            self.avg_val_loss = self.val_loss / len(test_Loader.dataset)

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
                output = f'{model_prefix}[{bar}] {epoch+1}/{epochs} | loss:{avg_train_loss:.4f}/{self.avg_val_loss:.4f} | grad:{avg_grad_norm:.3f} | {time_str}'
            else:
                output = f'{model_prefix}[{bar}] {epoch+1}/{epochs} | loss:{avg_train_loss:.4f}/{self.avg_val_loss:.4f} | {time_str}'
            
            # 如果指定了行号，使用ANSI转义码定位到特定行
            if progress_line is not None:
                # 保存当前光标位置，移动到指定行，打印，恢复光标位置
                print(f'\033[s\033[{progress_line+1};0H{output}\033[K\033[u', end='', flush=True)
            else:
                # 单行刷新
                print(f'\r{output}', end='', flush=True)
            
            self.train_loss_list.append(avg_train_loss)
            self.val_loss_list.append(self.avg_val_loss)
            
            # 早停检查
            if early_stopping:
                if self.avg_val_loss < best_val_loss:
                    best_val_loss = self.avg_val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict().copy()
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
        
        # # 打印梯度诊断报告
        # if check_gradient and len(self.gradient_info) > 0:
        #     self._print_gradient_report()

        return self.val_loss
    
    def _print_gradient_report(self):
        """打印梯度诊断报告"""
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