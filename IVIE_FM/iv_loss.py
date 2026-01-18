import torch
import torch.nn as nn


class interval_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rel, reu, ta):
        """

        :param rel: prediction lower
        :param reu: prediction upper
        :param ta: target
        :return: interval-value loss
        """
        tal = ta[:, 0]
        tal = tal.reshape((len(tal), 1))
        tau = ta[:, 1]
        tau = tau.reshape((len(tau), 1))

        # 使用hausdorff距离
        distance = 1/2 * torch.sqrt(torch.pow(rel - tal, 2) + torch.pow(reu - tau, 2))
        loss = torch.mean(torch.pow(distance, 2))

        return loss, distance


class ImprovedIntervalLoss(nn.Module):
    """
    改进的区间损失函数
    
    包含三个部分:
    1. 端点MSE损失: 确保预测端点接近真实端点
    2. 区间有效性损失: 惩罚 lower > upper 的无效区间
    3. 区间宽度匹配损失: 确保预测区间宽度与真实区间宽度匹配
    """
    def __init__(self, validity_weight=0.1, width_weight=0.05):
        super().__init__()
        self.validity_weight = validity_weight
        self.width_weight = width_weight

    def forward(self, rel, reu, ta):
        """
        :param rel: prediction lower (batch, 1)
        :param reu: prediction upper (batch, 1)
        :param ta: target (batch, 2) - [lower, upper]
        :return: total loss, distance
        """
        tal = ta[:, 0:1]  # (batch, 1)
        tau = ta[:, 1:2]  # (batch, 1)
        
        # 1. 端点MSE损失（分别计算，权重相等）
        mse_lower = torch.mean((rel - tal) ** 2)
        mse_upper = torch.mean((reu - tau) ** 2)
        endpoint_loss = mse_lower + mse_upper
        
        # 2. 区间有效性损失（惩罚 rel > reu 的情况）
        validity_loss = torch.mean(torch.relu(rel - reu))
        
        # 3. 区间宽度匹配损失
        pred_width = reu - rel
        true_width = tau - tal
        width_loss = torch.mean((pred_width - true_width) ** 2)
        
        # 总损失
        total_loss = (endpoint_loss + 
                      self.validity_weight * validity_loss + 
                      self.width_weight * width_loss)
        
        # 计算距离用于评估
        distance = torch.sqrt((rel - tal) ** 2 + (reu - tau) ** 2 + 1e-8)
        
        return total_loss, distance


class HausdorffIntervalLoss(nn.Module):
    """
    基于Hausdorff距离的区间损失函数
    
    Hausdorff距离: d_H([a,b], [c,d]) = max(|a-c|, |b-d|)
    """
    def __init__(self, validity_weight=0.1):
        super().__init__()
        self.validity_weight = validity_weight

    def forward(self, rel, reu, ta):
        tal = ta[:, 0:1]
        tau = ta[:, 1:2]
        
        # Hausdorff距离
        diff_lower = torch.abs(rel - tal)
        diff_upper = torch.abs(reu - tau)
        hausdorff = torch.max(diff_lower, diff_upper)
        
        # 主损失
        loss = torch.mean(hausdorff ** 2)
        
        # 有效性惩罚
        validity_loss = torch.mean(torch.relu(rel - reu))
        
        total_loss = loss + self.validity_weight * validity_loss
        
        return total_loss, hausdorff