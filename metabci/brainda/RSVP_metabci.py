import numpy as np
import mne
from mne.preprocessing import ICA
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from algorithms.deep_learning.Attn_EEGNet import EEGNetAttn
from metabci.brainda.algorithms.manifold.riemann import Alignment
from metabci.brainda.algorithms.manifold.riemann import MDRM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold




cdt_file = r'C:\Users\TAIIC\Desktop\meta-rsvp\MetaBCI\metabci\brainda\subject_1\Acquisition 01.cdt'

# 读取 .cdt 文件
raw = mne.io.read_raw_curry(cdt_file, preload=True, verbose=None)

# 输出信息查看
print(raw.info)


raw.pick_channels(['O2', 'O1', 'OZ', 'PO8', 'PO7', 'PO6', 'PO5', 'PO4', 'PO3', 'POZ', 'P8', 'P7', 'P6', 'P5', 'P4', 'P3', 'PZ', 'TP8', 'TP7', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'])

# 设置采样率一致
raw.resample(250)

raw.notch_filter(freqs=50, method='spectrum_fit')  
raw.filter(l_freq=1, h_freq=30, method='fir', fir_window='hamming')  
# raw.plot_psd()

# 重参考到通道7和通道8的平均值
raw.set_eeg_reference(ref_channels=['TP7', 'TP8'])  # 替换为实际通道名称
raw.drop_channels(['TP7', 'TP8'])

ica = ICA(n_components=23, method='fastica', random_state=42)

# 2. 拟合数据
ica.fit(raw)


# 5. 应用 ICA 去除伪迹
ica.apply(raw)
# raw.plot_psd(fmax=120, average=False, spatial_colors=True)


# 从标注中提取事件及事件ID
events, event_id = mne.events_from_annotations(raw)

# 定义处理窗口并初始化结果变量
labels_for_targ = []
train_trials = []

#  ！！事件与标签自动映射 记得改标签！！！


# 遍历事件
for event in events:
    latency = event[0]  # 获取事件的时间点（样本）
    event_type = event[2]  # 获取事件类型（与 event_id 字典对应）

    # 只处理标签为 '1' 和 '2' 的事件
    if event_type not in [1, 3]:
        continue

    # 定义处理窗口
    sfreq = raw.info['sfreq']  # 采样率
    tou = int(latency - sfreq * 0.2)  # 起始时间点（事件前200ms）
    wei = int(latency + sfreq * 0.8)  # 结束时间点（事件后800ms）

    # 确保索引不越界
    if tou < 0 or wei > len(raw.times):
        print(f"Skipping event at sample {latency} due to boundary conditions.")
        continue

    # 提取时间窗口数据
    waves = raw.get_data(start=tou, stop=wei)  # 获取数据窗口

    # 去均值处理（基线校正）
    baseline_mean = waves[:, :int(sfreq * 0.2)].mean(axis=1, keepdims=True)
    waves = waves - baseline_mean
    waves = waves[:,50:250]

    # 存储数据和标签
    labels_for_targ.append(event_type)
    train_trials.append(waves)

# 转换为 NumPy 数组

labels_for_targ = np.array(labels_for_targ)
labels_for_targ[labels_for_targ == 1] = 0
labels_for_targ[labels_for_targ == 3] = 1
y = labels_for_targ = labels_for_targ.reshape(-1, 1)

X = np.array(train_trials)            # (n_trials, n_channels, n_times)
          # (n_trials, 1, n_channels, n_times)
y = np.array(labels_for_targ).squeeze()  # (n_trials,)




# 5折交叉

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
acc_list = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ----------- 对训练数据做Alignment，参数 align_method 可选 'riemann' 或 'euclid'
    aligner = Alignment(align_method='riemann')
    X_train_aligned = aligner.fit(X_train).transform(X_train)
    X_test_aligned = aligner.transform(X_test)   # 测试集用训练集中心对齐

    # 这里举例用MDRM分类
    
    clf = MDRM()
    clf.fit(X_train_aligned, y_train)
    y_pred = clf.predict(X_test_aligned)

    
    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    print(f"Fold {fold+1} acc: {acc:.4f}")

print(f"\n5-Fold CV Average Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")










