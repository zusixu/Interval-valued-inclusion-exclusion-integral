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
        
        featuers_datal,featuers_datau = self.op(datal, datau)
        
        featuers_datal_mt = torch.matmul(featuers_datal, self.feature_matrix)
        featuers_datau_mt = torch.matmul(featuers_datau, self.feature_matrix)

        # 拆分左右端点
        
        a = torch.matmul(self.FM.T, featuers_datal_mt[:, ::2].T)  # 奇数列（左端点）
        b = torch.matmul(self.FM.T, featuers_datau_mt[:, ::2].T)  # 奇数列（左端点）
        c = torch.matmul(self.FM.T, featuers_datal_mt[:, 1::2].T) # 偶数列（右端点）
        d = torch.matmul(self.FM.T, featuers_datau_mt[:, 1::2].T) # 偶数列（右端点）
        
        # 区间减法
        left = torch.min(a - c, b - d)
        right = b - d

        # 转置为 (batch, 1) 格式
        return left.T, right.T
    
        # Converts NN-vars to FM vars
    def ivie_nn_vars(self, ivie_vars):
#        nVars,_ = ivie_vars.size()
        ivie_vars = torch.abs(ivie_vars)
        #        nInputs = inputs.get_shape().as_list()[1]
        
        FM = ivie_vars[None, 0,:]
        for i in range(1,self.nVars):
            indices = subset_to_indices(self.subset[i])
            if (len(indices) == 1):
                FM = torch.cat((FM,ivie_vars[None,i,:]),0)
            else:
                #         ss=tf.gather_nd(variables, [[1],[2]])
                maxVal,_ = torch.max(FM[indices,:],0)
                temp = torch.add(maxVal,ivie_vars[i,:])
                FM = torch.cat((FM,temp[None,:]),0)
              
        FM = torch.cat([FM, torch.ones((1,1), device=self.device)], 0)
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