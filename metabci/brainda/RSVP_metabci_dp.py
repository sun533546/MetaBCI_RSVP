import numpy as np
import mne
from mne.preprocessing import ICA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from algorithms.deep_learning.Attn_EEGNet import EEGNetAttn


def scale_data(data):
    for i in range(data.shape[0]):
        avg = np.mean(data[i,:,:], axis=-1, keepdims=True)
        std = np.std(data[i,:,:], axis=-1, keepdims=True)
        data[i,:,:] = (data[i,:,:] - avg) / std
    return data


# =============== 1. 数据读取与预处理 ==============
cdt_file = r'C:\Users\TAIIC\Desktop\meta-rsvp\MetaBCI\metabci\brainda\subject_1\Acquisition 01.cdt'
raw = mne.io.read_raw_curry(cdt_file, preload=True, verbose=None)
print(raw.info)

# 选取相关通道
raw.pick_channels(['O2', 'O1', 'OZ', 'PO8', 'PO7', 'PO6', 'PO5', 'PO4', 'PO3', 'POZ', 
                   'P8', 'P7', 'P6', 'P5', 'P4', 'P3', 'PZ', 'TP8', 'TP7', 
                   'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'])

raw.resample(250)
raw.notch_filter(freqs=50, method='spectrum_fit')
raw.filter(l_freq=1, h_freq=30, method='fir', fir_window='hamming')
raw.set_eeg_reference(ref_channels=['TP7', 'TP8'])
raw.drop_channels(['TP7', 'TP8'])

ica = ICA(n_components=23, method='fastica', random_state=42)
ica.fit(raw)
ica.apply(raw)

# 提取事件和标签
events, event_id = mne.events_from_annotations(raw)
labels_for_targ = []
train_trials = []

for event in events:
    latency = event[0]
    event_type = event[2]
    # 只取目标与非目标事件（按你的event_id实际映射调整）
    if event_type not in [1, 3]:
        continue
    sfreq = raw.info['sfreq']
    tou = int(latency - sfreq * 0.2)
    wei = int(latency + sfreq * 0.8)
    if tou < 0 or wei > len(raw.times):
        print(f"Skipping event at sample {latency} due to boundary conditions.")
        continue
    waves = raw.get_data(start=tou, stop=wei)
    baseline_mean = waves[:, :int(sfreq * 0.2)].mean(axis=1, keepdims=True)
    waves = waves - baseline_mean
    waves = waves[:, 50:250]  # 只取某时间窗
    labels_for_targ.append(event_type)
    train_trials.append(waves)

labels_for_targ = np.array(labels_for_targ)
labels_for_targ[labels_for_targ == 1] = 0
labels_for_targ[labels_for_targ == 3] = 1
y = labels_for_targ.reshape(-1)
X = np.array(train_trials)             # (n_trials, n_channels, n_times)
X = X[:, np.newaxis, :, :]             # (n_trials, 1, n_channels, n_times)

X = scale_data(X)

print("X shape:", X.shape, "y shape:", y.shape)

# =============== 2. 加载模型与预训练权重 ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # 二分类

# 构建模型，注意参数需与训练时一致
model = EEGNetAttn(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(r'C:\Users\TAIIC\Desktop\meta-rsvp\model_EEGNETATTN.pth', map_location=device))
model.eval()

# =============== 3. 模型推理（批量预测） ==============
with torch.no_grad():
    X_tensor = torch.tensor(X).float().to(device)  # shape: (n_samples, 1, n_channels, n_times)
    outputs = model(X_tensor)                      # shape: (n_samples, num_classes)
    y_pred = outputs.argmax(dim=1).cpu().numpy()

print("预测标签:", y_pred)
if y is not None:
    acc = accuracy_score(y, y_pred)
    print("Test accuracy:", acc)

    # 加权平均准确率（balanced accuracy）
    mask_target = (y == 1)
    mask_nontarget = (y == 0)
    acc_target = (y_pred[mask_target] == y[mask_target]).mean() if mask_target.sum() > 0 else np.nan
    acc_nontarget = (y_pred[mask_nontarget] == y[mask_nontarget]).mean() if mask_nontarget.sum() > 0 else np.nan
    acc_balanced = 0.5 * acc_target + 0.5 * acc_nontarget
    print(f"Target acc: {acc_target:.4f}, Nontarget acc: {acc_nontarget:.4f}, Balanced acc: {acc_balanced:.4f}")











