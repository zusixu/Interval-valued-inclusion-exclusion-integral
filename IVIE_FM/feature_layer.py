'''
对特征进行处理,构建01稀疏矩阵
'''
import torch
from typing import Tuple, Optional


class FeatureMatrix:
    """
    特征稀疏矩阵构建类
    
    用于构建和管理01稀疏矩阵，支持特征子集的数学运算。
    
    数学原理:
    - 超集表示: T = S ∪ E, 其中 E ⊆ S̄
    - 差集大小: |T \ S| = |E| = popcount(e)
    - 子集枚举: e_{k+1} = (e_k - 1) & complement
    
    属性:
        n: 特征数量
        device: 计算设备
        _matrix: 缓存的稀疏矩阵
        _statistics: 缓存的统计信息
    """
    
    def __init__(self, n: int, device: str = 'cpu'):
        """
        初始化特征矩阵
        
        参数:
            n: 特征数量
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.n = n
        self.device = device
        self._matrix: Optional[torch.Tensor] = None
        self._statistics: Optional[dict] = None
    
    @property
    def num_subsets(self) -> int:
        """返回非空子集数量 (2^n - 1)"""
        return (1 << self.n) - 1
    
    @property
    def shape(self) -> Tuple[int, int]:
        """返回矩阵形状"""
        return (self.num_subsets, 2 * self.num_subsets)
    
    def build_sparse_matrix(self) -> torch.Tensor:
        """
        构建稀疏矩阵（带缓存）
        
        返回:
            稀疏COO张量，形状 (2^n-1, 2*(2^n-1))
        """
        if self._matrix is not None:
            return self._matrix
        
        self._matrix = self._build_sparse_matrix_impl()
        return self._matrix
    
    def _build_sparse_matrix_impl(self) -> torch.Tensor:
        """
        使用数学公式的优化实现（内部方法）
        
        返回:
            稀疏COO张量，形状 (2^n-1, 2*(2^n-1))
        """
        num_subsets = self.num_subsets
        full_mask = num_subsets
        
        # 预计算popcount查找表
        popcount_lut = torch.zeros(num_subsets + 1, dtype=torch.int32)
        for i in range(num_subsets + 1):
            popcount_lut[i] = bin(i).count('1')
        
        # 预估非零元素数量: 约 3^n - 2^n
        estimated_nnz = 3**self.n - 2**self.n + self.n
        row_indices = torch.zeros(estimated_nnz, dtype=torch.long)
        col_indices = torch.zeros(estimated_nnz, dtype=torch.long)
        
        idx = 0
        for s in range(1, num_subsets + 1):
            complement = full_mask ^ s  # S̄ 的位掩码
            
            # 枚举补集的所有子集（包括空集）
            e = complement
            while True:
                t = s | e  # 超集 T = S ∪ E
                diff_parity = popcount_lut[e].item() % 2  # |E| mod 2
                
                row_idx = t - 1  # 0-indexed 行
                # 奇数列: 2s-2, 偶数列: 2s-1 (0-indexed)
                col_idx = 2 * s - 2 + (1 - diff_parity)
                
                row_indices[idx] = row_idx
                col_indices[idx] = col_idx
                idx += 1
                
                if e == 0:
                    break
                e = (e - 1) & complement  # 下一个子集
        
        # 截断到实际大小
        row_indices = row_indices[:idx]
        col_indices = col_indices[:idx]
        
        # 构建稀疏张量
        indices = torch.stack([row_indices, col_indices])
        values = torch.ones(idx, dtype=torch.float32)
        size = (num_subsets, 2 * num_subsets)
        
        sparse_matrix = torch.sparse_coo_tensor(
            indices, values, size, device=self.device
        )
        return sparse_matrix.coalesce()
    
    def get_statistics(self) -> dict:
        """
        获取矩阵的统计信息（带缓存）
        
        返回:
            包含矩阵维度、非零元素数量、稀疏度等信息的字典
        """
        if self._statistics is not None:
            return self._statistics
        
        num_subsets = self.num_subsets
        total_elements = num_subsets * 2 * num_subsets
        nnz = 3**self.n - 2**self.n  # 精确的非零元素数量
        
        self._statistics = {
            'n': self.n,
            'rows': num_subsets,
            'cols': 2 * num_subsets,
            'total_elements': total_elements,
            'nnz': nnz,
            'sparsity': 1 - nnz / total_elements,
            'nnz_per_row_avg': nnz / num_subsets,
            'memory_dense_MB': total_elements * 4 / (1024**2),
            'memory_sparse_MB': nnz * 12 / (1024**2),  # COO: row + col + value
        }
        return self._statistics
    
    def visualize(self) -> None:
        """可视化矩阵结构（仅适用于小规模矩阵）"""
        if self.n > 4:
            print(f"n={self.n} 太大，跳过可视化")
            return
        
        M = self.build_sparse_matrix().to_dense()
        num_subsets = self.num_subsets
        
        print(f"\n=== n={self.n} 矩阵结构 ===")
        print(f"形状: {M.shape}")
        
        # 打印带标签的矩阵
        header = "     "
        for s in range(1, num_subsets + 1):
            header += f" {s}o {s}e"
        print(header)
        print("-" * len(header))
        
        for t in range(1, num_subsets + 1):
            row_str = f"{t:3d} |"
            for s in range(1, num_subsets + 1):
                odd_val = int(M[t-1, 2*s-2].item())
                even_val = int(M[t-1, 2*s-1].item())
                row_str += f" {odd_val}  {even_val}"
            print(row_str)
    
    def to_dense(self) -> torch.Tensor:
        """返回稠密矩阵表示"""
        return self.build_sparse_matrix().to_dense()
    
    def to(self, device: str) -> 'FeatureMatrix':
        """
        将矩阵移动到指定设备
        
        参数:
            device: 目标设备
            
        返回:
            新的 FeatureMatrix 实例
        """
        new_matrix = FeatureMatrix(self.n, device)
        if self._matrix is not None:
            new_matrix._matrix = self._matrix.to(device)
        new_matrix._statistics = self._statistics
        return new_matrix
    
    def clear_cache(self) -> None:
        """清除缓存的矩阵和统计信息"""
        self._matrix = None
        self._statistics = None
    
    def __repr__(self) -> str:
        return f"FeatureMatrix(n={self.n}, device='{self.device}', shape={self.shape})"


if __name__ == "__main__":
    # 可视化小规模
    fm = FeatureMatrix(3).build_sparse_matrix()
    