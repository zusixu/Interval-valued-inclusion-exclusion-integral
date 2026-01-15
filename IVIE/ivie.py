import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import numpy as np
from scipy.special import comb
from tqdm import tqdm
from . import narray_op,feature_layer


# Convert decimal to binary string
def sources_and_subsets_nodes(N):
    str1 = "{0:{fill}"+str(N)+"b}"
    a = []
    for i in range(1,2**N):
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))
    # find subset nodes of a node
    def node_subset(node, sourcesInNodes):
        return [node - 2**(i) for i in sourcesInNodes]
    
    # convert binary encoded string to integer list
    def string_to_integer_array(s, ch):
        N = len(s) 
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]
    
    for j in range(len(a)):
        # index from right to left
        idxLR = string_to_integer_array(a[j],'1')
        sourcesInNode.append(idxLR)  
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))
        subset.append(node_subset(j,idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices):
    return [i for i in indices]




class IE(nn.Module):
    def __init__(self, feature_size, additivity_order=None, op='Algebraic_interval', alpha=1, beta=0, device='cuda'):
        super(IE, self).__init__()
        self.add = additivity_order
        self.narray_op = op
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.error = torch.tensor((), device=device)
        self.columns_num = feature_size
        self.nVars = 2**self.columns_num - 2

        self.feature_matrix = feature_layer.FeatureMatrix(self.columns_num, device=device).build_sparse_matrix()
        
        # The FM is initialized with mean
        dummy = (1./self.columns_num) * torch.ones((self.nVars, 1), requires_grad=True)
#        self.vars = torch.nn.Parameter( torch.Tensor(self.nVars,N_out))
        self.vars = torch.nn.Parameter(dummy)
        
        # following function uses numpy vs pytorch
        self.sourcesInNode, self.subset = sources_and_subsets_nodes(self.columns_num)
        
        self.sourcesInNode = [torch.tensor(x) for x in self.sourcesInNode]
        self.subset = [torch.tensor(x) for x in self.subset]

        if self.add == None:
            self.add = self.columns_num

        if self.add > self.columns_num:
            raise IndexError('"additivity_order" must be less than the "number of features"')
        if self.narray_op not in ['Algebraic_interval', 'Min_interval']:  
            raise ValueError('narray_op / Algebraic_interval, Min_interval') 

        if self.narray_op == 'Min_interval':
            self.op = narray_op.Min_interval(self.add, self.alpha, self.beta)
        elif self.narray_op == 'Algebraic_interval':
            self.op = narray_op.Algebraic_interval(self.add)  
        

    def forward(self, x):
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
        
        # 区间减法
        left = torch.min(a - c, b - d)
        right = b - d

        # 转置为 (batch, 1) 格式
        return left.T, right.T
    
        # Converts NN-vars to FM vars
    def ivie_nn_vars(self, ivie_vars):
        ivie_vars = torch.abs(ivie_vars)
        
        # 当使用 additivity_order 时，只处理对应阶数的子集
        if self.add < self.columns_num:
            # 计算有效的掩码（阶数 <= additivity_order）
            valid_masks = []
            for mask in range(1, 2**self.columns_num):
                order = bin(mask).count('1')
                if order <= self.add:
                    valid_masks.append(mask - 1)  # 转换为0-based索引
            
            # 只使用有效掩码对应的vars
            num_valid = len(valid_masks)
            FM = ivie_vars[None, valid_masks[0], :]
            
            for idx in range(1, num_valid):
                var_idx = valid_masks[idx]
                indices = subset_to_indices(self.subset[var_idx])
                
                # 过滤indices，只保留在valid_masks中的
                valid_indices_in_FM = []
                for sub_idx in indices:
                    if sub_idx in valid_masks:
                        # 找到sub_idx在valid_masks中的位置
                        pos = valid_masks.index(sub_idx)
                        valid_indices_in_FM.append(pos)
                
                if len(valid_indices_in_FM) == 0:
                    # 没有有效的子集，直接使用当前值
                    FM = torch.cat((FM, ivie_vars[None, var_idx, :]), 0)
                elif len(valid_indices_in_FM) == 1:
                    FM = torch.cat((FM, ivie_vars[None, var_idx, :]), 0)
                else:
                    maxVal, _ = torch.max(FM[valid_indices_in_FM, :], 0)
                    temp = torch.add(maxVal, ivie_vars[var_idx, :])
                    FM = torch.cat((FM, temp[None, :]), 0)
            
            FM = torch.cat([FM, torch.ones((1, 1), device=self.device)], 0)
            FM = torch.min(FM, torch.ones(1, device=self.device))
            
        else:
            # 原始逻辑：使用所有vars
            FM = ivie_vars[None, 0, :]
            for i in range(1, self.nVars):
                indices = subset_to_indices(self.subset[i])
                if (len(indices) == 1):
                    FM = torch.cat((FM, ivie_vars[None, i, :]), 0)
                else:
                    maxVal, _ = torch.max(FM[indices, :], 0)
                    temp = torch.add(maxVal, ivie_vars[i, :])
                    FM = torch.cat((FM, temp[None, :]), 0)
                  
            FM = torch.cat([FM, torch.ones((1, 1), device=self.device)], 0)
            FM = torch.min(FM, torch.ones(1, device=self.device))
        
        return FM

    def fit_and_valid(self, train_Loader, test_Loader, criterion, optimizer, device='cuda', epochs=100):
        start = time.time()
        self.train_loss_list = []
        self.val_loss_list = []
        self.lrs_list = []

        for epoch in range(epochs):
            self.train_loss = 0
            self.val_loss = 0

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
                # Compute the its gradients
                loss.backward()
                # Adjust learning weights
                optimizer.step()
                self.lrs = optimizer.param_groups[0]["lr"]
                self.lrs_list.append(optimizer.param_groups[0]["lr"])

            avg_train_loss = self.train_loss / len(train_Loader.dataset)

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

            print('Epoch [{}/{}], train_loss: {loss:.8f} val_loss: {val_loss:.8f}'
                  .format(epoch + 1, epochs, i + 1, loss=avg_train_loss, val_loss=self.avg_val_loss))
            self.train_loss_list.append(avg_train_loss)
            self.val_loss_list.append(self.avg_val_loss)

        print("compute time")
        print(time.time() - start)

        return self.val_loss