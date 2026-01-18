import torch
import torch.nn as nn

    

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