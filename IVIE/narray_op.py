import torch
import torch.nn as nn
from itertools import combinations


def _combo_to_bitmask(combo):
    """将组合元组转换为位掩码整数，用于排序"""
    mask = 0
    for idx in combo:
        mask |= (1 << idx)
    return mask


def _reorder_by_bitmask(cols_list, index_dict):
    """按位编码顺序重排列列表，返回排序后的张量"""
    # 按位掩码排序所有组合
    sorted_combos = sorted(index_dict.keys(), key=_combo_to_bitmask)
    sorted_indices = [index_dict[c] for c in sorted_combos]
    # 按排序后的索引重排列
    sorted_cols = [cols_list[i] for i in sorted_indices]
    return torch.cat(sorted_cols, dim=1)


class Min_interval(nn.Module):
    def __init__(self, add, alpha, beta):
        super().__init__()
        self.add = add
        self.alpha = alpha
        self.beta = beta

    def forward(self, xl, xu):
        columns_num = xl.size(1)
        items = list(range(columns_num))

        nodes_tnorml_cols = [xl[:, i : i + 1] for i in items]
        nodes_tnormu_cols = [xu[:, i : i + 1] for i in items]

        index_dict = {}
        for idx in items:
            index_dict[(idx,)] = idx

        max_order = min(self.add, columns_num)
        if max_order < 2:
            self.nodes_tnorml = torch.cat(nodes_tnorml_cols, dim=1)
            self.nodes_tnormu = torch.cat(nodes_tnormu_cols, dim=1)
            return self.nodes_tnorml, self.nodes_tnormu

        for length in range(2, max_order + 1):
            combos = list(combinations(items, length))
            if not combos:
                continue

            left_indices = [index_dict[c[1:]] for c in combos]
            right_indices = [index_dict[c[:-1]] for c in combos]

            left_lower = torch.cat([nodes_tnorml_cols[idx] for idx in left_indices], dim=1)
            left_upper = torch.cat([nodes_tnormu_cols[idx] for idx in left_indices], dim=1)
            right_lower = torch.cat([nodes_tnorml_cols[idx] for idx in right_indices], dim=1)
            right_upper = torch.cat([nodes_tnormu_cols[idx] for idx in right_indices], dim=1)

            result_lower, result_upper = self._select_pair(
                left_lower,
                left_upper,
                right_lower,
                right_upper,
            )

            for offset, combo in enumerate(combos):
                nodes_tnorml_cols.append(result_lower[:, offset : offset + 1])
                nodes_tnormu_cols.append(result_upper[:, offset : offset + 1])
                index_dict[combo] = len(nodes_tnorml_cols) - 1

        # 按位编码顺序重排输出
        self.nodes_tnorml = _reorder_by_bitmask(nodes_tnorml_cols, index_dict)
        self.nodes_tnormu = _reorder_by_bitmask(nodes_tnormu_cols, index_dict)
        return self.nodes_tnorml, self.nodes_tnormu

    def _select_pair(self, left_lower, left_upper, right_lower, right_upper):
        cur = (1 - self.alpha) * left_lower + self.alpha * left_upper
        nxt = (1 - self.alpha) * right_lower + self.alpha * right_upper
        choose_right = cur > nxt

        tie_mask = cur.eq(nxt)
        if tie_mask.any():
            beta_cur = (1 - self.beta) * left_lower + self.beta * left_upper
            beta_next = (1 - self.beta) * right_lower + self.beta * right_upper
            choose_right = torch.where(tie_mask, beta_cur > beta_next, choose_right)

        result_lower = torch.where(choose_right, right_lower, left_lower)
        result_upper = torch.where(choose_right, right_upper, left_upper)
        return result_lower, result_upper


class Algebraic_interval(nn.Module):
    """
    区间乘法运算: [a,b] * [c,d] = [a*c, b*d]
    对所有特征组合进行累积乘法
    """

    def __init__(self, add):
        super().__init__()
        self.add = add

    def forward(self, xl, xu):
        columns_num = xl.size(1)
        items = list(range(columns_num))

        nodes_tnorml_cols = [xl[:, i : i + 1] for i in items]
        nodes_tnormu_cols = [xu[:, i : i + 1] for i in items]

        index_dict = {}
        for idx in items:
            index_dict[(idx,)] = idx

        max_order = min(self.add, columns_num)
        if max_order < 2:
            self.nodes_tnorml = torch.cat(nodes_tnorml_cols, dim=1)
            self.nodes_tnormu = torch.cat(nodes_tnormu_cols, dim=1)
            return self.nodes_tnorml, self.nodes_tnormu

        for length in range(2, max_order + 1):
            combos = list(combinations(items, length))
            if not combos:
                continue

            left_indices = [index_dict[c[1:]] for c in combos]
            right_indices = [index_dict[c[:-1]] for c in combos]

            left_lower = torch.cat([nodes_tnorml_cols[idx] for idx in left_indices], dim=1)
            left_upper = torch.cat([nodes_tnormu_cols[idx] for idx in left_indices], dim=1)
            right_lower = torch.cat([nodes_tnorml_cols[idx] for idx in right_indices], dim=1)
            right_upper = torch.cat([nodes_tnormu_cols[idx] for idx in right_indices], dim=1)

            # 区间乘法: [a,b] * [c,d] = [a*c, b*d]
            result_lower = left_lower * right_lower
            result_upper = left_upper * right_upper

            for offset, combo in enumerate(combos):
                nodes_tnorml_cols.append(result_lower[:, offset : offset + 1])
                nodes_tnormu_cols.append(result_upper[:, offset : offset + 1])
                index_dict[combo] = len(nodes_tnorml_cols) - 1

        # 按位编码顺序重排输出
        self.nodes_tnorml = _reorder_by_bitmask(nodes_tnorml_cols, index_dict)
        self.nodes_tnormu = _reorder_by_bitmask(nodes_tnormu_cols, index_dict)
        return self.nodes_tnorml, self.nodes_tnormu
