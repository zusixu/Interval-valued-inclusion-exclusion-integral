import torch
import torch.nn as nn
import sys
sys.path.append('..')
from IVIE.ivie import IE


class IVCHI(IE):
    """
    IVCHI类：IE的派生类
    主要修改：将区间减法改为区间加法
    区间加法规则：[a,b]+[c,d]=[a+c,b+d]
    """
    
    def __init__(self, feature_size, additivity_order=None, op='Algebraic_interval', alpha=1, beta=0, device='cuda'):
        """
        初始化IVCHI模型
        
        参数:
            feature_size: 特征维度
            additivity_order: 加性阶数，默认为None（等于feature_size）
            op: 操作类型，'Algebraic_interval' 或 'Min_interval'
            alpha: Min_interval操作的alpha参数
            beta: Min_interval操作的beta参数
            device: 设备类型，'cuda' 或 'cpu'
        """
        super(IVCHI, self).__init__(feature_size, additivity_order, op, alpha, beta, device)
    
    def forward(self, x):
        """
        前向传播
        修改点：将区间减法 [a,b]-[c,d]=[min(a-c,b-d), b-d] 
               改为区间加法 [a,b]+[c,d]=[a+c, b+d]
        """
        self.FM = self.ivie_nn_vars(self.vars)

        columns_num = x.size()[1]
        columns_num = int(columns_num)
        index = columns_num / 2
        index = int(index)
        datal = x[:, :index]
        datau = x[:, index:]
        
        featuers_datal, featuers_datau = self.op(datal, datau)
        
        # 当 additivity_order < columns_num 时，op 返回的特征数量会减少
        # 需要构建对应的 feature_matrix 子集
        actual_num_features = featuers_datal.size(1)
        expected_num_features = 2**self.columns_num - 1
        
        if actual_num_features < expected_num_features:
            # 计算哪些位掩码被保留（阶数 <= additivity_order）
            valid_masks = []
            for mask in range(1, 2**self.columns_num):
                order = bin(mask).count('1')
                if order <= self.add:
                    valid_masks.append(mask)
            
            # 构建子feature_matrix：只保留valid_masks对应的行和列
            # feature_matrix 形状: (2^n-1, 2*(2^n-1))
            # 我们需要: (num_valid, 2*num_valid)
            feature_matrix_dense = self.feature_matrix.to_dense()
            
            # 行索引：valid_masks - 1 (因为mask从1开始，索引从0开始)
            row_indices = [m - 1 for m in valid_masks]
            
            # 列索引：每个valid_mask对应两列 (2*idx, 2*idx+1)
            col_indices = []
            for idx in row_indices:
                col_indices.extend([2*idx, 2*idx+1])
            
            # 提取子矩阵
            sub_matrix = feature_matrix_dense[row_indices, :][:, col_indices]
            feature_matrix_to_use = sub_matrix
            
            # FM已经在ivie_nn_vars中被正确缩小，直接使用
            FM_to_use = self.FM
        else:
            feature_matrix_to_use = self.feature_matrix.to_dense()
            FM_to_use = self.FM
        
        featuers_datal_mt = torch.matmul(featuers_datal, feature_matrix_to_use)
        featuers_datau_mt = torch.matmul(featuers_datau, feature_matrix_to_use)
        
        # 拆分左右端点
        a = torch.matmul(FM_to_use.T, featuers_datal_mt[:, ::2].T)  # 奇数列（左端点）
        b = torch.matmul(FM_to_use.T, featuers_datau_mt[:, ::2].T)  # 奇数列（左端点）
        c = torch.matmul(FM_to_use.T, featuers_datal_mt[:, 1::2].T) # 偶数列（右端点）
        d = torch.matmul(FM_to_use.T, featuers_datau_mt[:, 1::2].T) # 偶数列（右端点）
        
        # ==================== 修改点 ====================
        # 原始IE: 区间减法 [a,b]-[c,d]=[min(a-c,b-d), b-d]
        # left = torch.min(a - c, b - d)
        # right = b - d
        
        # IVCHI: 区间加法 [a,b]+[c,d]=[a+c, b+d]
        left = a + c
        right = b + d
        # ===============================================

        # 转置为 (batch, 1) 格式
        return left.T, right.T
