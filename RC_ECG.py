

#执行ECG任务

import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import resample
from collections import defaultdict
import random

def z_norm_normalization(data):
    """Normalize data using Z-normalization."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data  # 防止除以零
    z_norm_data = (data - mean) / std
    return z_norm_data

# 设置随机数种子
np.random.seed(0)
random.seed(0)

# 本地数据路径
local_path = 'mitdb'

# 记录的列表
records = ['232', '222', '209', '208', '223', '200', '213', '205', '109']
# 心跳类型和目标数量
types = {'N': 750, 'L': 750, 'V': 750, 'A': 750}

# 存储每种类型的心跳及其标签
heartbeats_by_type = defaultdict(list)
heartbeat_labels = defaultdict(list)

# 加载数据和提取心跳
for record_id in records:
    record_path = f'{local_path}/{record_id}'
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    # 获取R峰和符号
    r_peaks = annotation.sample
    symbols = annotation.symbol
    
    # 根据类型收集心跳
    for i, symbol in enumerate(symbols):
        if symbol in types and len(heartbeats_by_type[symbol]) < types[symbol]:
            start = (r_peaks[i-1] + r_peaks[i]) // 2 if i != 0 else 0
            end = (r_peaks[i] + r_peaks[i+1]) // 2 if i < len(r_peaks)-1 else len(record.p_signal)
            heartbeat = record.p_signal[start:end, 0]  # 假设只有一个通道
            # 重新采样
            target_samples = int(len(heartbeat) * 80 / 360)  # 计算目标采样点数
            resampled_heartbeat = resample(heartbeat, target_samples)
            # 应用Z-Norm标准化
            normalized_heartbeat = z_norm_normalization(resampled_heartbeat)
            heartbeats_by_type[symbol].append(normalized_heartbeat)
            # 计算重新采样后的R峰相对位置
            relative_peak_position = int((r_peaks[i] - start) * 80 / 360)
            heartbeat_labels[symbol].append((symbol, relative_peak_position))

# 连接和随机排列心跳
all_heartbeats = []
all_labels = []
for type_key in types:
    all_heartbeats.extend(heartbeats_by_type[type_key])
    all_labels.extend(heartbeat_labels[type_key])

# 随机打乱心跳和标签
combined = list(zip(all_heartbeats, all_labels))
random.shuffle(combined)
all_heartbeats, all_labels = zip(*combined)  # 解压回两个列表

# 计算连接后的心跳的R峰位置
concatenated_heartbeat = np.concatenate(all_heartbeats)
current_length = 0
beat_positions = []
for heartbeat, label_info in zip(all_heartbeats, all_labels):
    symbol, relative_peak_position = label_info
    absolute_peak_position = current_length + relative_peak_position
    beat_positions.append((symbol, absolute_peak_position))
    current_length += len(heartbeat)

# 绘图
plt.figure(figsize=(15, 4))
plt.plot(concatenated_heartbeat[-1000:], label="Concatenated Heartbeats of N, F, V, A Types")
plt.title("Randomly Concatenated Heartbeats with R-Peak Positions")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()


# In[43]:


import numpy as np

def create_vector_matrix(beat_positions, total_length):
    """基于AAMI标签创建向量矩阵。"""
    matrix = np.zeros((total_length, 4))  # 默认所有位置为 [0, 0, 0, 0]
    label_vectors = {
        'A': np.array([1, 0, 0, 0]),
        'L': np.array([0, 1, 0, 0]),
        'V': np.array([0, 0, 1, 0]),
        'N': np.array([0, 0, 0, 1])
    }
    for label, pos in beat_positions:
        if label in label_vectors:  # 只处理映射过的标签
            start = max(0, pos - 4)  # 定义每个标签影响的前4个点
            end = min(total_length, pos + 9)  # 定义每个标签影响的6个点
            matrix[start:end] = label_vectors[label]
    return matrix

total_length = concatenated_heartbeat.size  # 假设concatenated_heartbeat已经创建

# 创建向量矩阵
vector_matrix = create_vector_matrix(beat_positions, total_length)


fig, axes = plt.subplots(4, 1, figsize=(20, 8), sharex=True)
labels = ['A', 'L', 'V', 'N']
colors = ['blue', 'green', 'red', 'purple']

for i in range(4):
    axes[i].plot(vector_matrix[4000:5000, i], color=colors[i], label=f'Label {labels[i]}')
    axes[i].set_ylabel(labels[i])
    axes[i].legend(loc='upper right')

plt.xlabel('Data Point Index')
plt.show()


# In[29]:


#数据编码（方法一：先加mask，后time encoding).
# 加mask

import numpy as np
import matplotlib.pyplot as plt

def generate_random_vectors(N, L):
    """生成N个随机向量，每个向量长度为L，元素为-1或1"""
    np.random.seed(42)  # 设置固定的随机种子
    return np.random.choice([-1, 1], size=(N, L))

def expand_signal_with_vectors(signal, random_vectors):
    """将信号的每个数据点与对应的随机向量相乘，并扩展为L个长度为N的列向量"""
    # 初始化一个空数组，大小为 N x (信号长度 * L)
    expanded_signal = np.zeros((N, len(signal) * L))
    for index, value in enumerate(signal):
        # 生成每个数据点的扩展向量
        expanded_matrix = value * random_vectors  # 使用未转置的随机向量
        # 将扩展向量放入结果数组中的适当位置
        expanded_signal[:, index * L:(index + 1) * L] = expanded_matrix
    return expanded_signal

# 示例参数
N = 8  # 例如，生成5个随机向量
L = 5 # 每个向量的长度为5
random_vectors = generate_random_vectors(N, L)

# 假设resampled_ecg_signal已经由之前的代码块得到

# 扩展信号
expanded_ecg_signal = expand_signal_with_vectors(concatenated_heartbeat, random_vectors)


# In[25]:

#PWM转换，time encoding
target_fs = 80 

T = 1/target_fs/L  # 将每行映射到区间 [0, T]
UL = np.max(expanded_ecg_signal)
DL = np.min(expanded_ecg_signal)
mapped_ecg_signal = (expanded_ecg_signal-DL)/(UL-DL)*(T-0)+0

# In[26]:

# 参数定义
vi = 3.3

RC_list = [
    {'R': 1, 'C': 0.4E-3},
    {'R': 1, 'C': 1.0E-3},
    {'R': 1, 'C': 1.6E-3},
    {'R': 1, 'C': 2.2E-3},
    {'R': 1, 'C': 2.8E-3},
    {'R': 1, 'C': 3.4E-3},
    {'R': 1, 'C': 4.0E-3},
    {'R': 1, 'C': 4.6E-3},     
]

# 初始化 s 数组，长度为 mapped_ecg_signal 中每一行的长度加1
s = np.zeros((mapped_ecg_signal.shape[0], mapped_ecg_signal.shape[1] + 1))

# 计算 s[i+1] 对每一行和对应的 RC 配对
for index, rc in enumerate(RC_list):
    tau = rc['R'] * rc['C']
    for i in range(mapped_ecg_signal.shape[1]):
        s[index, i+1] = vi * (1 - np.exp(-mapped_ecg_signal[index, i] / tau)) * np.exp(-(T - mapped_ecg_signal[index, i]) / tau) + s[index, i] * np.exp(-T / tau)


# In[30]:

#整理输出矩阵

# 从第二列开始提取每五列数据
extracted_cols = s[:, 1:].reshape(s.shape[0], -1, 5)                       #经过RC
#extracted_cols = mapped_ecg_signal[:, 0:].reshape(s.shape[0], -1, 5)        #不经过RC

# 将提取的每五列数据组合成新的一列
new_s= extracted_cols.transpose(0, 2, 1).reshape(-1, extracted_cols.shape[1])


# 创建一行全为1的数组，长度与 new_s_trimmed 的列数相同
new_row = np.ones((1, new_s.shape[1]))

# 将新行添加到矩阵的底部
new_s_expanded = np.vstack((new_s, new_row))

# In[44]:

#线性回归

# 计算权重向量
y_target = vector_matrix.T[:,1000:int(len(vector_matrix)*0.8)] #清洗掉前1000个点，训练前80%

x_train = new_s_expanded[:,1000:int(len(vector_matrix)*0.8)]

w = y_target @ x_train.T @ np.linalg.pinv(x_train  @ x_train.T) 


# In[45]:

#输出训练结果

y_train = w @ x_train

y_train.shape


# In[46]:

# 设置绘图
fig, axs = plt.subplots(4, 1, figsize=(20, 8))  # 创建一个图形和4个子图，每个子图一行

for i in range(4):
    axs[i].plot(y_target[i, 0:1000])  
    axs[i].plot(y_train[i, 0:1000])  # 绘制每行
    axs[i].set_title(f'Row {i+1}')  # 设置每个子图的标题
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('Value')

plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图形


# In[47]:

#输出测试结果

x_test = new_s_expanded[:,int(len(vector_matrix)*0.8):len(vector_matrix)]

y_test = w @ x_test

y_test_target = vector_matrix.T[:,int(len(vector_matrix)*0.8):len(vector_matrix)]


# In[50]:


# 设置绘图
fig, axs = plt.subplots(4, 1, figsize=(20, 8))  # 创建一个图形和4个子图，每个子图一行

for i in range(4):
    axs[i].plot(y_test_target[i,-1000:])  
    axs[i].plot(y_test[i,-1000:])  # 绘制每行
    axs[i].set_title(f'Row {i+1}')  # 设置每个子图的标题
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('Value')

plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图形

# In[51]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 初始化所有计数器
N = np.zeros((4, 4))  # 用于保存每一行（A, F, V, N）的计数结果

# 遍历每一行和每一列
for i in range(y_test_target.shape[0]):  # 遍历每一行A, F, V, N
    in_staircase = False
    start_index = 0

    for j in range(y_test_target.shape[1]):  # 遍历每一列
        if y_test_target[i, j] == 1 and not in_staircase:
            # 开始新的台阶
            start_index = j
            in_staircase = True
        elif y_test_target[i, j] == 0 and in_staircase:
            # 结束当前的台阶
            end_index = j
            in_staircase = False

            # 处理台阶
            segment = y_test[:, start_index:end_index]
            max_values = np.max(segment, axis=1)
            max_index = np.argmax(max_values)

            # 根据最大值的索引更新计数器
            N[i, max_index] += 1

        # 如果到达最后还在台阶中，处理这最后一个台阶
        if j == y_test_target.shape[1] - 1 and in_staircase:
            end_index = j + 1
            segment = y_test[:, start_index:end_index]
            max_values = np.max(segment, axis=1)
            max_index = np.argmax(max_values)
            N[i, max_index] += 1

# 输出结果
for row_label, counts in zip(['A', 'L', 'V', 'N'], N):
    print(f'Counts for {row_label}: {row_label}A={counts[0]}, {row_label}L={counts[1]}, {row_label}V={counts[2]}, {row_label}N={counts[3]}')

#画出混淆矩阵    

# 设置标签名
labels = ['A', 'L', 'V', 'N']

# 创建一个heatmap来显示混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(N, annot=True, fmt=".0f", cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
    
# 计算每一类的总次数
total_counts = np.sum(N, axis=1)

# 计算每一类的准确率
accuracies = np.diag(N) / total_counts

# 计算平均准确率
average_accuracy = np.mean(accuracies)

# 输出每一类的准确率和平均准确率
print("Individual Accuracies:", accuracies)
print("Average Accuracy:", average_accuracy)

#origin画图数据导出
aa = concatenated_heartbeat[-1000:]
bb = y_test_target[:,-1000:]
cc = y_test[:,-1000:]

#功耗计算只计算test的
energy1 = mapped_ecg_signal[:,int(mapped_ecg_signal.shape[1]*0.8):]
energy = np.sum(0.1 * 0.1 * mapped_ecg_signal[:,int(mapped_ecg_signal.shape[1]*0.8):] / 40E6)
