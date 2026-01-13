import sys
from pathlib import Path
from itertools import combinations

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from IVIE.narray_op import Min_interval, Mul_interval


def _naive_select_min(left_l, left_u, right_l, right_u, alpha, beta):
    """Naive pairwise min selection for each sample"""
    batch = left_l.size(0)
    result_l = []
    result_u = []
    for i in range(batch):
        cur = (1 - alpha) * left_l[i] + alpha * left_u[i]
        nxt = (1 - alpha) * right_l[i] + alpha * right_u[i]
        if cur > nxt:
            result_l.append(right_l[i])
            result_u.append(right_u[i])
        elif cur == nxt:
            beta_cur = (1 - beta) * left_l[i] + beta * left_u[i]
            beta_nxt = (1 - beta) * right_l[i] + beta * right_u[i]
            if beta_cur > beta_nxt:
                result_l.append(right_l[i])
                result_u.append(right_u[i])
            else:
                result_l.append(left_l[i])
                result_u.append(left_u[i])
        else:
            result_l.append(left_l[i])
            result_u.append(left_u[i])
    return torch.stack(result_l).unsqueeze(1), torch.stack(result_u).unsqueeze(1)


def naive_min_interval(xl, xu, add, alpha, beta):
    columns_num = xl.size(1)
    items = list(range(columns_num))
    
    # Store columns in list
    nodes_l_cols = [xl[:, i:i+1] for i in items]
    nodes_u_cols = [xu[:, i:i+1] for i in items]
    
    index_dict = {}
    for idx in items:
        index_dict[(idx,)] = idx

    max_order = min(add, columns_num)
    for length in range(2, max_order + 1):
        for combo in combinations(items, length):
            index_l = index_dict[combo[1:]]
            index_r = index_dict[combo[:-1]]
            
            left_l = nodes_l_cols[index_l]
            left_u = nodes_u_cols[index_l]
            right_l = nodes_l_cols[index_r]
            right_u = nodes_u_cols[index_r]
            
            result_l, result_u = _naive_select_min(
                left_l.squeeze(1), left_u.squeeze(1),
                right_l.squeeze(1), right_u.squeeze(1),
                alpha, beta
            )
            nodes_l_cols.append(result_l)
            nodes_u_cols.append(result_u)
            index_dict[combo] = len(nodes_l_cols) - 1

    # Reorder by bitmask
    def combo_to_bitmask(c):
        mask = 0
        for idx in c:
            mask |= (1 << idx)
        return mask
    
    sorted_combos = sorted(index_dict.keys(), key=combo_to_bitmask)
    sorted_l = [nodes_l_cols[index_dict[c]] for c in sorted_combos]
    sorted_u = [nodes_u_cols[index_dict[c]] for c in sorted_combos]
    
    return torch.cat(sorted_l, dim=1), torch.cat(sorted_u, dim=1)


def test_min_interval_matches_naive():
    torch.manual_seed(0)
    batch = 8
    dims = 6
    add = 4
    alpha = 0.3
    beta = 0.7

    base = torch.rand(batch, dims)
    spread = torch.rand(batch, dims)
    xl = base
    xu = base + spread

    model = Min_interval(add=add, alpha=alpha, beta=beta)
    opt_lower, opt_upper = model(xl, xu)

    naive_lower, naive_upper = naive_min_interval(xl, xu, add, alpha, beta)

    assert torch.allclose(opt_lower, naive_lower)
    assert torch.allclose(opt_upper, naive_upper)


def test_min_interval_tie_breaking_prefers_beta():
    alpha = 0.5
    beta = 0.9
    add = 2

    xl = torch.tensor([[0.0, 0.2]])
    xu = torch.tensor([[1.0, 0.8]])

    model = Min_interval(add=add, alpha=alpha, beta=beta)
    opt_lower, opt_upper = model(xl, xu)

    naive_lower, naive_upper = naive_min_interval(xl, xu, add, alpha, beta)

    assert torch.allclose(opt_lower, naive_lower)
    assert torch.allclose(opt_upper, naive_upper)
