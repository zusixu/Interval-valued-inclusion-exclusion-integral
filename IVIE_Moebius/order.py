import torch
import torch.nn as nn

'''
这里是为了实现对输入的区间值排序并返回最小值，输入了两个tensor分别表示了区间的上下界
cols表示的是每个样本输入了多少个区间值
'''


class Ordered(nn.Module):
    def __init__(self, alpha, beta):
        """
        :param alpha: float
        :param beta: float
        admissible order
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, xl, xu):
        """
        向量化版本的区间排序，返回每行最小的区间
        
        :param xl: tensor(batch_size, cols) 下界
        :param xu: tensor(batch_size, cols) 上界
        :return: (tensor(batch_size, 1), tensor(batch_size, 1)) 最小区间的下界和上界
        """
        # 计算 admissible order 的排序键值
        # K_alpha = (1 - alpha) * xl + alpha * xu
        k_alpha = (1 - self.alpha) * xl + self.alpha * xu
        
        # 找到每行 K_alpha 最小值的索引
        min_k_alpha, min_indices = torch.min(k_alpha, dim=1, keepdim=True)
        
        # 处理 K_alpha 相等的情况，使用 K_beta 作为第二排序键
        # K_beta = (1 - beta) * xl + beta * xu
        k_beta = (1 - self.beta) * xl + self.beta * xu
        
        # 找出所有等于最小 K_alpha 的位置
        is_min_alpha = (k_alpha == min_k_alpha)
        
        # 对于非最小位置，将 k_beta 设为无穷大，这样它们不会被选中
        k_beta_masked = torch.where(is_min_alpha, k_beta, torch.full_like(k_beta, float('inf')))
        
        # 找到 K_beta 最小的索引（在 K_alpha 相等的情况下）
        _, final_indices = torch.min(k_beta_masked, dim=1, keepdim=True)
        
        # 根据索引提取对应的下界和上界
        resultl = torch.gather(xl, 1, final_indices)
        resultu = torch.gather(xu, 1, final_indices)
        
        return resultl, resultu
    
    def forward_legacy(self, rowsl, rowsu, cols):
        """
        保留旧版本接口以兼容（已废弃）
        """
        # 将 tuple 转换为 tensor
        xl = torch.cat(rowsl, dim=0)
        xu = torch.cat(rowsu, dim=0)
        return self.forward(xl, xu)
