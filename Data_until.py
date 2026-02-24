import numpy as np
import torch
from torch.utils.data import Dataset


class Config():
    window_size = 30 # 滑动窗口大小
    dropout = 0 # dropout概率
    kernel_size_tcn = 3 #tcn卷积核大小
    bidirectional_lstm = True #是否双向
    bidirectional_gru = True
    num_layer = 1 #lstm与gru层数
    ratio = 4 #通道注意力过渡层
    min_layer_output_dim = 64  #最小输出维度
    max_layer_output_dim = 256 #最大输出维度
    population_size = 50 # 种群大小
    max_generations = 20 # 最大繁殖迭代次数
    max_layers = 5 # 最大层数
    batch_size = 1024 # 批次大小
    num_epochs = 150 # 每个个体的epoch
    cross_rate = 0.8 # 交叉概率
    mutation_rate = 0.3 # 变异概率s
    tournament_size = 4 # 锦标赛选择个数
    lr = 0.001 # 学习率
    weight_decay = 0.00005 # l2正则化系数

config = Config()
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
def process_targets(data_length, early_rul=None):
    if early_rul is None:
        return torch.arange(data_length - 1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            return torch.arange(data_length - 1, -1, -1)
        else:
            return torch.cat((early_rul * torch.ones(early_rul_duration), torch.arange(early_rul - 1, -1, -1)))

def process_input_data_with_targets(input_data, target_data=None, window_length=1, shift=1):
    num_batches = int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = torch.empty(num_batches, window_length, num_features)

    if target_data is None:
        for batch in range(num_batches):
            output_data[batch] = input_data[batch * shift:batch * shift + window_length]
        return output_data
    else:
        output_targets = torch.empty(num_batches)
        for batch in range(num_batches):
            output_data[batch] = input_data[batch * shift:batch * shift + window_length]
            output_targets[batch] = target_data[batch * shift + (window_length - 1)]
        return output_data, output_targets

def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=1):
    max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, num_test_windows
