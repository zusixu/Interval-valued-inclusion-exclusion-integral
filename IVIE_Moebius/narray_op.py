import torch
import tqdm
import torch.nn as nn
from . import order
from itertools import combinations


class Algebraic_interval(nn.Module):
    def __init__(self, add):
        super().__init__()
        self.add = add

    def forward(self, xl, xu):
        """
        区间值计算版本的代数积（Algebraic Product）
        
        Args:
            xl: 下界张量，shape为(batch_size, columns_num)
            xu: 上界张量，shape为(batch_size, columns_num)
        
        Returns:
            nodes_tnorml: 下界结果
            nodes_tnormu: 上界结果
            
        特征组合顺序:
            1. 原始特征: 0, 1, 2, ..., n-1
            2. 2个特征组合: (0,1), (0,2), ..., (n-2,n-1)
            3. 3个特征组合: (0,1,2), (0,1,3), ..., (n-3,n-2,n-1)
            ...以此类推直到add个特征的组合
        """
        columns_num = xl.size()[1]
        self.nodes_tnorml = xl  # 下界
        self.nodes_tnormu = xu  # 上界
        items = [i for i in range(0, columns_num)]
        for i in range(columns_num + 1):
            if i > self.add:  # add加法的まで
                break
            for c in combinations(items, i):
                if len(c) >= 2:  # cols(c)=1のときはそのまま出力層へ
                    c = list(c)  # tuple to list
                    subsetl = xl[:, c]
                    subsetu = xu[:, c]
                    # 对于非负区间，乘积的下界=各下界之积，上界=各上界之积
                    resultl = torch.prod(subsetl, 1, keepdim=True)
                    resultu = torch.prod(subsetu, 1, keepdim=True)
                    self.nodes_tnorml = torch.cat((self.nodes_tnorml, resultl), dim=1)
                    self.nodes_tnormu = torch.cat((self.nodes_tnormu, resultu), dim=1)
        return self.nodes_tnorml, self.nodes_tnormu



class Min_interval(nn.Module):
    def __init__(self, add, alpha, beta):
        super().__init__()
        self.add = add
        self.alpha = alpha
        self.beta = beta
        self.admissible_order = order.Ordered(alpha, beta)

    def forward(self, xl, xu):
        """
        区间值最小T-norm计算
        
        Args:
            xl: 下界张量，shape为(batch_size, columns_num)
            xu: 上界张量，shape为(batch_size, columns_num)
        
        Returns:
            nodes_tnorml: 下界结果
            nodes_tnormu: 上界结果
            
        特征组合顺序（与Algebraic_interval相同）:
            1. 原始特征: 0, 1, 2, ..., n-1
            2. 2个特征组合: (0,1), (0,2), ..., (n-2,n-1)
            3. 3个特征组合: (0,1,2), (0,1,3), ..., (n-3,n-2,n-1)
            ...以此类推直到add个特征的组合
        """
        columns_num = xl.size()[1]
        items = list(range(columns_num))
        
        # 预先计算所有组合并分配索引
        index_dict = {}
        index = 0
        for i in range(columns_num + 1):
            if i > self.add:
                break
            for c in combinations(items, i):
                if len(c) >= 1:
                    index_dict[c] = index
                    index += 1
        
        # 预分配结果张量列表（避免频繁cat）
        result_cols_l = [xl[:, i:i+1] for i in range(columns_num)]
        result_cols_u = [xu[:, i:i+1] for i in range(columns_num)]
        
        # 计算组合特征
        for i in range(2, self.add + 1):
            if i > columns_num:
                break
            for c in combinations(items, i):
                # 利用记忆化：min(a,b,c) = min(min(a,b), min(b,c))
                index_l = index_dict[c[1:]]  # 去掉第一个元素的组合
                index_r = index_dict[c[:-1]]  # 去掉最后一个元素的组合
                
                # 获取两个子区间（向量化操作）
                if index_l < columns_num:
                    subset_l_l = xl[:, index_l:index_l+1]
                    subset_l_u = xu[:, index_l:index_l+1]
                else:
                    subset_l_l = result_cols_l[index_l]
                    subset_l_u = result_cols_u[index_l]
                
                if index_r < columns_num:
                    subset_r_l = xl[:, index_r:index_r+1]
                    subset_r_u = xu[:, index_r:index_r+1]
                else:
                    subset_r_l = result_cols_l[index_r]
                    subset_r_u = result_cols_u[index_r]
                
                # 拼接两个候选区间，shape: (batch_size, 2)
                candidates_l = torch.cat([subset_r_l, subset_l_l], dim=1)
                candidates_u = torch.cat([subset_r_u, subset_l_u], dim=1)
                
                # 向量化求最小区间
                min_l, min_u = self.admissible_order(candidates_l, candidates_u)
                
                result_cols_l.append(min_l)
                result_cols_u.append(min_u)
        
        # 一次性拼接所有结果
        self.nodes_tnorml = torch.cat(result_cols_l, dim=1)
        self.nodes_tnormu = torch.cat(result_cols_u, dim=1)

        return self.nodes_tnorml, self.nodes_tnormu
