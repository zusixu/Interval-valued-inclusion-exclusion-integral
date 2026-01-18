import torch
import torch.nn as nn


class OutputLayer_single(nn.Module):
    def __init__(self, columns_num, all_num):  #
        super().__init__()
        self.columns_num = int(columns_num)
        self.all_num = int(all_num)

        # definition of a matrix to store weights
        weight = torch.rand(1, self.all_num)
        weight[0, :self.columns_num] = 1 / self.columns_num
        self.weight = nn.Parameter(weight)
        # definition of a vector to store the bias
        bias = torch.FloatTensor([0.5])
        self.bias = nn.Parameter(bias)

    def forward(self, xl, xu):
        result_low = torch.nn.functional.linear(xl, self.weight, self.bias)
        result_up = torch.nn.functional.linear(xu, self.weight, self.bias)
        return result_low, result_up


class OutputLayer_interval(nn.Module):
    def __init__(self, columns_num, all_num):  #
        super().__init__()
        self.columns_num = int(columns_num)
        self.all_num = int(all_num)

        # definition of a matrix to store weights
        weight = torch.zeros(1, self.all_num)
        #weight = torch.rand(1, self.all_num)
        weight[0, :self.columns_num] = 1 / self.columns_num
        self.weight_left = nn.Parameter(weight)
        self.weight_right = nn.Parameter(weight)
        # definition of a vector to store the bias
        bias = torch.FloatTensor([0])
        self.bias_left = nn.Parameter(bias)
        self.bias_right = nn.Parameter(bias)

    def forward(self, xl, xu):
        result_low = torch.nn.functional.linear(xl, self.weight_left, self.bias_left)
        result_up = torch.nn.functional.linear(xu, self.weight_right, self.bias_right)
        return result_low, result_up

    def init_parameters(self):
        self.weight_left.data = torch.zeros_like(self.weight_left)
        self.weight_left.data[0, :self.columns_num] = 1 / self.columns_num
        self.weight_right.data = torch.zeros_like(self.weight_right)
        self.weight_right.data[0, :self.columns_num] = 1 / self.columns_num
        self.bias_left.data = torch.zeros_like(self.bias_left)
        self.bias_right.data = torch.zeros_like(self.bias_right)