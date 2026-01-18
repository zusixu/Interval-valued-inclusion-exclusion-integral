import torch
import torch.nn as nn
import sys
sys.path.append('..')
from IVIE_FM.ivie import IE


class IVCHI(IE):
    """
    IVCHI类：IE的派生类
    主要修改：将区间减法改为区间加法
    区间加法规则：[a,b]+[c,d]=[a+c,b+d]
    """
    
    def __init__(self, feature_size, additivity_order=None, op='Min_interval', alpha=1, beta=0, device='cuda'):
        """
        初始化IVCHI模型
        
        参数:
            feature_size: 特征维度
            additivity_order: 加性阶数，默认为None（等于feature_size）
            op: 操作类型'Min_interval'
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
