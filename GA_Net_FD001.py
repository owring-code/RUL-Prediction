import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from thop import profile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Config:
    window_size = 30 # 滑动窗口大小
    dropout = 0 # dropout概率
    kernel_size_tcn = 3 #tcn卷积核大小
    bidirectional_lstm = True #是否双向
    bidirectional_gru = True
    num_layer = 1 #lstm与gru层数
    ratio = 4 #通道注意力过渡层
    min_layer_output_dim = 64  #最小隐藏神经元
    max_layer_output_dim = 256 #最大隐藏神经元
    population_size = 50 # 种群大小
    max_generations = 20 # 最大繁殖迭代次数
    max_layers = 5 # 最大层数
    batch_size = 1024 # 批次大小
    num_epochs = 150 # 每个个体的epoch
    cross_rate = 0.8 # 交叉概率
    mutation_rate = 0.3 # 变异概率
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



# 读取数据
train_data = pd.read_csv("/openbayes/input/input0/train_FD001.txt", sep="\s+", header=None)
test_data = pd.read_csv("/openbayes/input/input0/test_FD001.txt", sep="\s+", header=None)
true_rul = pd.read_csv("/openbayes/input/input0/RUL_FD001.txt", sep='\s+', header=None).values.flatten()

# 定义窗口长度和移动步长
window_length = config.window_size
shift = 1
early_rul = 125

# 初始化处理后的训练和测试数据
processed_train_data = []
processed_train_targets = []
num_test_windows = 5
processed_test_data = []
num_test_targets = []  # 用于存储测试目标
num_test_windows_list = []
# 要删除的列索引
columns_to_be_dropped = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]

# 归一化数据
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_scaled = scaler.fit_transform(train_data.drop(columns=columns_to_be_dropped))
test_data_scaled = scaler.transform(test_data.drop(columns=columns_to_be_dropped))

# 将归一化后的数据与第一列合并
train_data = np.c_[train_data.iloc[:, 0], train_data_scaled]
test_data = np.c_[test_data.iloc[:, 0], test_data_scaled]

# 获取训练和测试机器的数量
num_train_machines = len(np.unique(train_data[:, 0]))
num_test_machines = len(np.unique(test_data[:, 0]))

# 处理训练数据
# 时间窗口
for i in range(1, num_train_machines + 1):
    temp_train_data = train_data[train_data[:, 0] == i][:, 1:]
    if len(temp_train_data) < window_length:
        print(f"Train engine {i} doesn't have enough data for window_length of {window_length}")
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    temp_train_targets = process_targets(data_length=temp_train_data.shape[0], early_rul=early_rul)
    data_for_a_machine, targets_for_a_machine = process_input_data_with_targets(
        torch.tensor(temp_train_data, dtype=torch.float32),
        temp_train_targets,
        window_length=window_length, shift=shift
    )

    processed_train_data.append(data_for_a_machine)
    processed_train_targets.append(targets_for_a_machine)

# 将训练数据和目标连接在一起
processed_train_data = torch.cat(processed_train_data)
processed_train_targets = torch.cat(processed_train_targets)

# 处理测试数据
for i in range(1, num_test_machines + 1):
    temp_test_data = test_data[test_data[:, 0] == i][:, 1:]
    if len(temp_test_data) < window_length:
        print(f"Test engine {i} doesn't have enough data for window_length of {window_length}")
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    test_data_for_an_engine, num_windows = process_test_data(
        torch.tensor(temp_test_data, dtype=torch.float32),
        window_length=window_length, shift=shift,
        num_test_windows=num_test_windows
    )

    processed_test_data.append(test_data_for_an_engine[-1].unsqueeze(0))
    num_test_windows_list.append(1)

    # 添加测试目标的处理
    if i <= len(true_rul):
        targets_for_test = process_targets(data_length=len(temp_test_data), early_rul=None)  # 或者根据需要调整early_rul
        targets_for_test += int(true_rul[i-1])
        # num_test_targets.append(targets_for_test[-num_windows:])  # 取最后的RUL目标
        num_test_targets.append(torch.tensor([true_rul[i-1]]))
    else:
        num_test_targets.append(torch.zeros(num_windows))  # 如果没有目标，则填充为零

# 将测试数据连接在一起
X_test = torch.cat(processed_test_data)
# 合并测试目标
Y_test = torch.cat(num_test_targets)

# 打乱训练数据
index = torch.randperm(len(processed_train_targets))
processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]

# 划分训练集和验证集
train_data, val_data, train_targets, val_targets = train_test_split(
    processed_train_data.numpy(),
    processed_train_targets.numpy(),
    test_size=0.2,
    random_state=42,
)

# 将训练数据和验证数据转换回PyTorch张量
X_train = torch.tensor(train_data, dtype=torch.float32)
X_val = torch.tensor(val_data, dtype=torch.float32)
Y_train = torch.tensor(train_targets, dtype=torch.float32)
Y_val = torch.tensor(val_targets, dtype=torch.float32)
# 创建 DataLoader
train_dataset = CustomDataset(X_train, Y_train)
val_dataset = CustomDataset(X_val, Y_val)
test_dataset = CustomDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last= True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last= True)

# 定义一些基本的神经网络层和注意力机制
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        #残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=config.dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = 1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs
            out_channels = num_channels
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=int((kernel_size-1)*dilation_size), dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out.transpose(1, 2)
        return out

class LSTMLayer(nn.Module):
    def __init__(self, input_size, output_size, num_layers=config.num_layer):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, output_size, num_layers=num_layers,
                            batch_first=True, bidirectional=config.bidirectional_lstm)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.dropout(x)
        return out

class GRULayer(nn.Module):
    def __init__(self, input_size, output_size, num_layer=config.num_layer):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_size, output_size, num_layers=num_layer, batch_first=True, bidirectional=config.bidirectional_gru)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x, _ = self.gru(x)
        out = self.dropout(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=config.ratio):
        super(ChannelAttention, self).__init__()
        assert in_planes > 0, "in_planes must be greater than 0"
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 共享MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )

        # self.relu1 = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2) 
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        out = x * out
        return out.transpose(1, 2)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (batch_size, time_steps, num_features)

        x = x.transpose(1, 2)  # (batch_size, num_features, time_steps)
        # print(x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #print(avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, time_steps)
        #print(max_out.shape)
        out = torch.cat([avg_out, max_out], dim=1)  # (batch_size, 2, time_steps)
        #print(out.shape)

        out = self.conv1(out)  # (batch_size, 1, time_steps)
        out = self.sigmoid(out)  # (batch_size, 1, time_steps)
        out = x * out # (batch_size, num_features, time_steps)
        out = out.transpose(1, 2)
        #print(out.shape)
        return out  # 返回加权后的输入

# 全连接
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# 构建多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError((hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 将“query”和“key”向量作相乘运算
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # 使用softmax进行归一化
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        hidden_layer = torch.matmul(attention_probs, value_layer)
        hidden_layer = hidden_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = hidden_layer.size()[:-2] + (self.all_head_size,)
        hidden_layer = hidden_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(hidden_layer)
        hidden_states = self.out_dropout(hidden_states)
        #print(hidden_states.shape)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的PE矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer，不参与反向传播更新，但在 state_dict 中保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # 截取与 x 序列长度对应的 PE 并相加
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x

# 可视化结果
def visualize(val_RMSE, predictions, true_rul):
    # 将 predictions 和 true_rul 转换为 CPU numpy 数组
    predictions = predictions.cpu().numpy()
    print(predictions.shape)
    true_rul = true_rul.cpu().numpy() if isinstance(true_rul, torch.Tensor) else true_rul
    print(true_rul.shape)
    plt.figure(figsize=(10, 6))
    plt.axvline(x=100, c='r', linestyle='--')
    plt.plot(true_rul, label='Actual Data')
    plt.plot(predictions, label='Predicted Data')
    plt.title('RUL Prediction on CMAPSS Data')
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")
    plt.savefig('result1/Transformer({}).png'.format(val_RMSE))
    plt.close()

def s_score(predictions, true_rul):
    predictions = predictions.cpu().numpy()
    true_rul = true_rul.cpu().numpy() if isinstance(true_rul, torch.Tensor) else true_rul

    diff = predictions - true_rul
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))


def evaluate_model_complexity(model, window_size=30, input_dim=14, device='cuda'):
    """
    计算模型的 FLOPs 和 参数量
    """
    model = model.to(device)
    model.eval()
    
    # 创建一个假的输入张量 (Batch_size=1)
    # 输入维度: [Batch, Window_Size, Features]
    # 注意：如果你的模型加了 Input Projection (64维)，这里的 input_dim 还是原始的 14
    dummy_input = torch.randn(1, window_size, input_dim).to(device)
    
    try:
        # thop.profile 会自动计算
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
        
        # 转换为更易读的单位 (M = 10^6, G = 10^9)
        flops_M = flops / 1e6
        params_K = params / 1e3
        
        return flops_M, params_K
    except Exception as e:
        print(f"Complexity calculation failed: {e}")
        return 0, 0


# 定义模型类
class DynamicModel(nn.Module):
    def __init__(self, chromosome):
        super(DynamicModel, self).__init__()

        self.input_proj_dim = 64 # 将原始14维特征投影到64维
        self.input_proj = nn.Linear(14, self.input_proj_dim)
        self.pos_encoder = PositionalEncoding(d_model=self.input_proj_dim, max_len=config.window_size + 10)

        # chromosome = [1,0,128]
        layers = ['TCN', 'LSTM', 'GRU', 'MultiHeadAttention', 'ChannelAttention', 'SpatialAttention']
        gene_size = chromosome[0]  # 第一个元素表示基因组的大小（层数）
        # 提取奇数索引的元素
        layers_name = [chromosome[i] for i in range(1, len(chromosome)) if i % 2 != 0]
        output_name = [chromosome[i] for i in range(1, len(chromosome)) if i % 2 == 0]

        input_dim = self.input_proj_dim
        self.dense_layers = nn.ModuleList()
        # print(chromosome)
        for i in range(0, gene_size):  # 根据层数取出相应的基因
            layer_type_index = layers_name[i]
            output_dim = output_name[i]
            # 获取基因信息
            layer_type = layers[layer_type_index]
            # 根据 layer_type 创建相应的层
            if layer_type == 'TCN':
                self.dense_layers.append(TemporalConvNet(input_dim, output_dim))
                input_dim = output_dim
            elif layer_type == 'LSTM':
                self.dense_layers.append(LSTMLayer(input_dim, output_dim))
                if config.bidirectional_lstm:
                    input_dim = output_dim * 2
                else:
                    input_dim = output_dim
            elif layer_type == 'GRU':
                self.dense_layers.append(GRULayer(input_dim, output_dim))
                if config.bidirectional_gru:
                    input_dim = output_dim * 2
                else:
                    input_dim = output_dim
            elif layer_type == 'MultiHeadAttention':
                num_head = 4 # 建议增加头数
                if output_dim % num_head != 0:
                    output_dim = (output_dim // num_head) * num_head
                self.dense_layers.append(MultiHeadAttention(num_head, input_dim, output_dim))
                input_dim = output_dim
            elif layer_type == 'ChannelAttention':
                self.dense_layers.append(ChannelAttention(input_dim))
            elif layer_type == 'SpatialAttention':
                self.dense_layers.append(SpatialAttention())


        self.fc = nn.Linear(input_dim, 1)  # 动态创建全连接层
        # 添加AdaptiveMaxPool1d层，将序列的时间维度减少到1
        self.pool = nn.AdaptiveMaxPool1d(1)


    # 初始化卷积权重
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # 使用Delta-正交初始化卷积层的权重
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # x shape: [batch, window, 14]
        x = self.input_proj(x) # [batch, window, 64]
        x = self.pos_encoder(x) # [batch, window, 64] with PE
        
        
        for layer in self.dense_layers:
            x = layer(x)
            if len(x.shape) == 3:
                l = nn.LayerNorm([x.shape[-2], x.shape[-1]]).to(x.device)
                x = l(x)
            # print(f"Layer {layer}: output shape {x.shape}")
        
        # 将输入 x 的时间维度池化为 1
        # print("x1_shape:",x.shape)
        # print(x)
        x = self.pool(x.transpose(1, 2))  # 需要将时间维度放到最后一维再池化
        # print("x2_shape:",x.shape)
        # print(x)
        x = x.squeeze(-1)  # 去掉时间维度
        # print("x3_shape:",x.shape)
        # print(x)
        # x = self.mlp(x)
        # 通过全连接层
        x = self.fc(x)
        # print("x4_shape:",x.shape)
        x = x.squeeze(-1)
        # print("x5_shape:",x.shape)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(learning_rate, learning_rate_decay, optimizer):
    """Sets the learning rate to the initial LR multiplied by learning_rate_decay(set 0.98, usually) every epoch"""
    learning_rate = learning_rate * learning_rate_decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

# 定义适应度函数（以 RMSE 为基础）
global_best = float('inf')
def fitness_function(Model, X_Train, y_Train, X_test, y_test, num_epoch, counter, chromosome):
    
    start_time = time.time() # 记录开始时间

    global global_best
    Model = Model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Model.parameters(), lr=config.lr)
    best_RMSE = float('inf')
    # 训练模型
    Model.init_weight()
    with tqdm(total=num_epoch, desc='Processing', colour='#00FFFF') as pbar:
        for _ in range(num_epoch):  # 可根据需要调整epoch数
            Model.train()
            train_loss = []
            val_loss = []
            for X_Train, y_Train in train_loader:
                X_Train = X_Train.to(device)
                y_Train = y_Train.to(device)
                
                # training model
                optimizer.zero_grad()
                outputs = Model(X_Train)
                loss = criterion(outputs, y_Train.float())
                counter += 1
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                
            Model.eval()
            with torch.no_grad():
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                predictions = Model(X_test)
                val_loss_cal = criterion(predictions, y_test.float())
                val_loss.append(val_loss_cal.item())
            val_RMSE = torch.sqrt(torch.tensor(np.mean(val_loss),dtype=torch.float32)).item()
            
            if best_RMSE > val_RMSE:
                best_RMSE = val_RMSE
                if global_best > best_RMSE :
                    global_best = best_RMSE
                    if global_best <= 12.5:
                        torch.save(Model.state_dict(), f"/openbayes/home/pth1/Model_1.pth")
                        struct = np.array(chromosome)
                        np.save('/openbayes/home/model_struct.npy', struct)
            pbar.set_postfix(**{"Loss": f"{np.mean(val_loss):.3f}",
                                "RMSE": f"{best_RMSE:.3f}",
                                "Best_RMSE": f"{global_best:.3f}"})
            pbar.update(1)
    pbar.close()

    end_time = time.time() # 记录结束时间
    cost_time = end_time - start_time
    print(f"Chromosome: {chromosome} | Best Validation RMSE: {best_RMSE:.4f} | Time taken: {cost_time:.2f} seconds")

    fitness_result = 1/best_RMSE

    return fitness_result, cost_time


# 定义遗传算法的关键步骤
def initialize_population(pop_size, max_layers):
    population = []
    # 计算每层的基础数量和余数分配
    base = pop_size // max_layers
    remainder = pop_size % max_layers
    
    # 生成层数列表，确保每层至少有base个，前remainder层多1个
    layer_counts = [base + 1 if i < remainder else base for i in range(max_layers)]
    layers = [i+1 for i, count in enumerate(layer_counts) for _ in range(count)]
    
    # 打乱层数顺序以避免连续生成相同层数
    random.shuffle(layers)
    
    # 为每个层数生成对应的染色体
    for num_layers in layers:
        chromosome = [num_layers]
        for _ in range(num_layers):
            layer_type = random.randint(0, 5)
            layer_output_dim = random.randint(config.min_layer_output_dim, config.max_layer_output_dim)
            chromosome.append(layer_type)
            chromosome.append(layer_output_dim)
        population.append(chromosome)
    
    return population


def crossover(parent1, parent2):
    # parent结构: [层数, 类型1, 维度1, 类型2, 维度2, ...]
    
    # 1. 提取基因对 (类型, 维度)
    def get_genes(parent):
        genes = []
        num_layers = parent[0]
        for i in range(num_layers):
            idx = 1 + i * 2
            genes.append(parent[idx : idx+2]) # [Type, Dim]
        return genes

    genes1 = get_genes(parent1)
    genes2 = get_genes(parent2)
    
    # 2. 确定交叉点 (基于较短的父代层数)
    min_layers = min(len(genes1), len(genes2))
    if min_layers < 2:
        # 如果层数太少，直接不交叉或随机返回一个
        return list(parent1), list(parent2)
        
    crossover_point = random.randint(1, min_layers - 1)
    
    # 3. 交换基因块
    # Child 1: P1的前半部分 + P2的后半部分
    child1_genes = genes1[:crossover_point] + genes2[crossover_point:]
    # Child 2: P2的前半部分 + P1的后半部分
    child2_genes = genes2[:crossover_point] + genes1[crossover_point:]
    
    # 4. 重组为染色体格式 [层数, flat_genes...]
    def reconstruct(genes):
        chromosome = [len(genes)]
        for gene in genes:
            chromosome.extend(gene)
        return chromosome
        
    child1 = reconstruct(child1_genes)
    child2 = reconstruct(child2_genes)
    
    return child1, child2


def mutate(chromosome):
    # 遍历染色体的每个基因（跳过第一个表示层数的基因）
    for i in range(1, len(chromosome)):
        if i % 2 != 0:
            if random.random() < mutation_rate:
                chromosome[i] = random.randint(0, 5)
        if i % 2 == 0:
            if random.random() < mutation_rate:
                chromosome[i] = random.randint(config.min_layer_output_dim, config.max_layer_output_dim)

    return chromosome


# 遗传算法参数
population_size = config.population_size
max_generations = config.max_generations
max_layers = config.max_layers
batch_size = config.batch_size
num_epochs = config.num_epochs
layer_choices = ['TCN', 'LSTM', 'GRU', 'MultiHeadAttention', 'ChannelAttention', 'SpatialAttention']
cross_rate = config.cross_rate
mutation_rate = config.mutation_rate

# 锦标赛选择
def tournament_selection(population_fitness, tournament_size):
    # 从种群中随机选择 tournament_size 个个体进行比赛
    tournament = random.sample(population_fitness, tournament_size)
    
    # 比赛中适应度最好的个体胜出
    winner = max(tournament, key=lambda x: x[0])  # 比较适应度（fitness），适应度越大越好
    return winner[1]  # 返回胜出个体的染色体

def genetic_algorithm(X_train, y_train, X_val, y_val):

    total_search_start = time.time()  # <--- 记录GA开始总时间
    total_compute_time = 0            # <--- 累计纯GPU训练时间

    population = initialize_population(population_size, max_layers)
    top_5_chormosomes = []
    with open("fitness_log1.txt", "a") as log_file:

        # 记录开始时间
        log_file.write(f"Search Started at: {time.ctime(total_search_start)}\n")

        for generation in range(max_generations):

            generation_start = time.time()

            population_fitness = []
            counter = 0
            for chromosome in population:
                # chromosome = [1, 2, 256, 1, 1]
                if len(chromosome)>1:
                    counter += 1
                    print("第{}个个体:".format(counter))
                    print(chromosome)
                    model = DynamicModel(chromosome)
                    fitness, cost_time = fitness_function(model, X_train, y_train, X_val, y_val, num_epochs, counter, chromosome)
                    total_compute_time += cost_time # 累加时间
                
                # 记录详细日志：个体结构 | RMSE | 耗时
                log_entry = (f"Gen {generation+1} - ID {counter}: "
                             f"Time={cost_time:.2f}s, "
                             f"RMSE={1/fitness:.4f}, "
                             f"Struct={chromosome}\n")
                print(log_entry)
                log_file.write(log_entry)
                
                population_fitness.append((fitness, chromosome))

            # 精英保留策略：找到当前种群中的最优的五个个体
            top_5 = sorted(population_fitness, key=lambda x: x[0], reverse=True)[:5]
            # print(top_5)
            for i in top_5:
                top_5_chormosomes.append(i)
            # print(top_5_chormosomes)
            top_5_chormosomes = sorted(top_5_chormosomes, key=lambda x: x[0], reverse=True)[:5]
            best_chromosome = max(population_fitness, key=lambda x: x[0])[1]
            best_fitness = max(population_fitness, key=lambda x: x[0])[0]

            new_population = []
            for _ in range(population_size // 2):
                selected_chromosome = tournament_selection(population_fitness, tournament_size=config.tournament_size)
                new_population.append(selected_chromosome)
            population = new_population

            # 交叉和变异
            next_generation = []
            while len(next_generation) < population_size :  
                parents = random.sample(population, 2)
                if random.random() < cross_rate:
                    offspring1, offspring2 = crossover(parents[0], parents[1])
                else:
                    offspring1, offspring2 = parents[0], parents[1]

                if random.random() < mutation_rate:
                    offspring1 = mutate(offspring1)
                if random.random() < mutation_rate:
                    offspring2 = mutate(offspring2)

                next_generation.extend([offspring1, offspring2])

            # 加入精英个体到下一代
            # next_generation.append(best_chromosome)
            # print(next_generation)
            # 更新种群
            generation_end = time.time()
            print(f"Generation {generation+1} Finished. Duration: {(generation_end - generation_start)/60:.2f} min")
            population = next_generation
            print(f"Generation {generation + 1}, Best fitness: {best_fitness},min_RMSE: {1/best_fitness}, Best chromosome: {best_chromosome}")
            # 输出这代最好的五个个体
            print(f"这代最好的五个个体:", top_5)
            # 输出目前最好的五个个体
            print(f"目前最好的五个个体:", top_5_chormosomes)
        
        print("搜索完成，开始评估最优模型的计算资源消耗...")

        # 重建最优模型
        best_model = DynamicModel(best_chromosome)

        # 计算 FLOPs 和 Params
        # 假设你的 window_size 是 30，特征数是 14
        flops, params = evaluate_model_complexity(best_model, 
                                          window_size=config.window_size, 
                                          input_dim=14, 
                                          device=device)

        print(f"=== Best Model Complexity ===")
        print(f"Structure: {best_chromosome}")
        print(f"Parameters: {params:.2f} K")
        print(f"FLOPs: {flops:.2f} M")

    total_search_end = time.time()
    total_wall_clock_time = total_search_end - total_search_start
    
    print(f"Total Search Time (Wall Clock): {total_wall_clock_time/3600:.2f} hours")
    print(f"Total GPU Compute Time: {total_compute_time/3600:.2f} hours")

    return population

# 调用主函数运行遗传算法
if __name__ == '__main__': 
    final_population = genetic_algorithm(X_train, Y_train, X_test, Y_test)