import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import numpy as np
from scipy.special import comb
from tqdm import tqdm
from . import output_layer, iv_loss, narray_op


class IE(nn.Module):
    def __init__(self, feature_size, additivity_order=None, op='Algebraic_interval', alpha=1, beta=1,
                 fuzzy_measure='OutputLayer_single', output_weights=None, output_bias=None):
        super(IE, self).__init__()
        self.add = additivity_order
        self.narray_op = op
        self.alpha = alpha
        self.beta = beta
        self.error = torch.tensor(())  
        self.columns_num = feature_size
        self.fuzzy_measure = fuzzy_measure

        if self.add == None:
            self.add = self.columns_num

        if self.add > self.columns_num:
            raise IndexError('"additivity_order" must be less than the "number of features"')
        if self.narray_op not in ['Algebraic_interval',  'Min_interval']:  
            raise ValueError('narray_op / Algebraic_interval, Min_interval') 
        # OutputLayer
        self.output_nodesSum = 0
        for i in range(1, self.add + 1):
            self.output_nodesSum += comb(self.columns_num, i, exact=True)
        # 初始化模糊测度
        if self.fuzzy_measure == 'OutputLayer_single':
            self.output = output_layer.OutputLayer_single(self.columns_num, self.output_nodesSum)
            if output_weights is not None:
                self.output.weight.data = output_weights
            if output_bias is not None:
                self.output.bias.data = output_bias
        elif self.fuzzy_measure == 'OutputLayer_interval':
            self.output = output_layer.OutputLayer_interval(self.columns_num, self.output_nodesSum)
            if output_weights is not None:
                self.output.weight_left.data = output_weights[0]
                self.output.weight_right.data = output_weights[1]
            if output_bias is not None:
                self.output.bias_left.data = output_bias[0]
                self.output.bias_right.data = output_bias[1]
        else:
            raise ValueError(
                'Invalid fuzzy_measure value. It should be either "OutputLayer_single" or "OutputLayer_interval".')
        
        if self.narray_op == 'Min_interval':
            self.op = narray_op.Min_interval(self.add, self.alpha, self.beta)
        elif self.narray_op == 'Algebraic_interval':
            self.op = narray_op.Algebraic_interval(self.add)  

    def forward(self, x):
        
        columns_num = x.size()[1]
        columns_num = int(columns_num)
        index = columns_num / 2
        index = int(index)
        datal = x[:, :index]
        datau = x[:, index:]
        self.datal, self.datau = self.op(datal, datau)
        
        return self.output(self.datal, self.datau)

    def fit_and_valid(self, train_Loader, test_Loader, criterion, optimizer, device='cpu', epochs=100,
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
        
        # 初始化error张量到正确的设备上
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
                if check_gradient:
                    # 收集所有参数的梯度
                    all_grads = []
                    for param in self.parameters():
                        if param.grad is not None:
                            all_grads.append(param.grad.view(-1))
                    
                    if len(all_grads) > 0:
                        all_grads_tensor = torch.cat(all_grads)
                        grad_norm = all_grads_tensor.norm().item()
                        grad_max = all_grads_tensor.abs().max().item()
                        grad_min = all_grads_tensor.abs().min().item()
                        
                        epoch_grad_norm += grad_norm
                        epoch_grad_max = max(epoch_grad_max, grad_max)
                        epoch_grad_min = min(epoch_grad_min, grad_min)
                        batch_count += 1
                        
                        # 检查NaN梯度
                        if torch.isnan(all_grads_tensor).any() or torch.isinf(all_grads_tensor).any():
                            has_nan_grad = True
                            print(f"警告: Epoch {epoch+1}, Batch {i+1} - 梯度包含NaN或Inf!")
                            # 将NaN梯度置零
                            for param in self.parameters():
                                if param.grad is not None:
                                    param.grad = torch.where(
                                        torch.isnan(param.grad) | torch.isinf(param.grad),
                                        torch.zeros_like(param.grad),
                                        param.grad
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
            print("  - 激活函数饱和")
            print("  - 网络层数过深")
            print("  - 初始化不当")
        
        if np.max(max_grads) > 100:
            print("\n⚠️ 警告: 梯度过大（梯度爆炸），建议:")
            print("  - 减小学习率")
            print("  - 增加梯度裁剪")
        
        print("="*60)


    def get_output_weights(self):
        if self.fuzzy_measure == 'OutputLayer_single':
            return self.output.weight.data, self.output.bias.data

        elif self.fuzzy_measure == 'OutputLayer_interval':
            return self.output.weight_left.data, self.output.weight_right.data, self.output.bias_left.data, self.output.bias_right.data

        else:
            raise ValueError("Invalid fuzzy_measure value.")

    def plot_train(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_loss_list, label='train data', lw=3, c='b')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0

    def plot_test(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.val_loss_list, label='test data', lw=3, c='b')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_loss_list, label='train data', lw=3, c='b')
        plt.plot(self.val_loss_list, label='test data', lw=3, c='r')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0

    def rec(self):
        errors = self.error.cpu().numpy()
        tolerances = np.linspace(0, errors.max(), 100)
        accuracies = np.array([np.mean(errors <= tol) for tol in tolerances])

        plt.plot(tolerances, accuracies)
        plt.xlabel('Error Tolerance')
        plt.ylabel('Prediction Accuracy')
        plt.title('Regression Error Characteristic (REC)')
        plt.show()

    def r2_score(self, test_Loader, device='cpu'):
        test_label = test_Loader.dataset[:][1].to(device)
        # 对第0维取平均,test_mean：tensor(2,)
        test_mean = torch.mean(test_label, dim=0)
        test_mean_list = torch.ones(len(test_label), 2) * test_mean
        
        mse_loss = iv_loss.HausdorffIntervalLoss()
        mean_loss, error = (mse_loss(test_label[:, 0], test_label[:, 1], test_mean_list.to(device)))

        mean_loss = mean_loss * len(test_label)
        print('val_loss: {val_loss:.8f}'
              .format(val_loss=self.avg_val_loss))
        self.r2 = 1 - self.val_loss / mean_loss.item()
        return self.r2
