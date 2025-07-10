import numpy as np
import mne
from mne.preprocessing import ICA
from mne.io import read_raw_curry
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from brainflow.workers import ProcessWorker
from brainda.algorithms.deep_learning.Attn_EEGNet import EEGNetAttn

def read_data(run_files, chs, interval, labels):
    """
    数据加载与预处理
    """
    all_trials = []
    all_labels = []
    for run_file in run_files:
        raw = read_raw_curry(run_file, preload=True, verbose=False)
        raw.pick_channels(chs)
        raw.resample(250)
        raw.notch_filter(freqs=50, method='spectrum_fit')
        raw.filter(l_freq=1, h_freq=30, method='fir', fir_window='hamming')
        ch_names = raw.info['ch_names']
        ica = ICA(n_components=len(chs), method='fastica', random_state=42)
        ica.fit(raw)
        ica.apply(raw)

        events, event_id = mne.events_from_annotations(raw)
        for event in events:
            latency = event[0]
            event_type = event[2]
            # 只处理1和3，后续转换为0/1
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
            waves = waves[:, 50:250]   # 取刺激后800ms内的数据 (共200采样点)
            all_trials.append(waves)
            all_labels.append(event_type)
    labels_arr = np.array(all_labels)
    labels_arr[labels_arr == 1] = 0
    labels_arr[labels_arr == 3] = 1
    X = np.array(all_trials)           # (n_trials, n_channels, n_times)
    y = labels_arr.astype(np.int64)    # (n_trials,)
    return X, y, ch_names

def train_attn_eegnet(X, y, ch_names, n_epochs=100, batch_size=32, lr=0.001, device='cpu'):
    """
    用Attn_EEGNet训练，并保存模型和通道顺序
    """
    X = (X - np.mean(X, axis=-1, keepdims=True)) / (np.std(X, axis=-1, keepdims=True) + 1e-6)
    X = X[:, np.newaxis, :, :]   # (n_trials, 1, n_channels, n_times)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    model = EEGNetAttn(num_classes=2, in_channels=1, samples=X.shape[-1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # y 的标签必须是 0=非目标，1=目标
    weights = torch.tensor([1.0, 11.5], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    n_trials = X.shape[0]
    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(n_trials)
        total_loss = 0
        for i in range(0, n_trials, batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X_tensor[idx].to(device)
            batch_y = y_tensor[idx].to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/n_trials:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "attn_eegnet_rsvp.pth")
    with open("attn_eegnet_ch_names.pkl", "wb") as f:
        pickle.dump(ch_names, f)
    print("模型和通道顺序保存为 attn_eegnet_rsvp.pth")
    return model

# worker框架：仅用于兼容调用

class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, pick_chs, stim_interval, stim_labels,
                 srate, lsl_source_id, timeout, worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        # 读取数据
        X, y, ch_names = read_data(run_files=self.run_files,
                                   chs=self.pick_chs,
                                   interval=self.stim_interval,
                                   labels=self.stim_labels)
        print("数据读取完成，开始训练Attn_EEGNet")
        train_attn_eegnet(X, y, ch_names)
        print("训练完成，模型已保存！")

    def consume(self, data):
        pass
    def post(self):
        pass

if __name__ == '__main__':
    srate = 1000
    stim_interval = [0, 0.8]
    stim_labels = [101, 103]
    run_files = [r'C:\Users\TAIIC\Desktop\meta-rsvp\MetaBCI\metabci\brainda\subject_1\Acquisition 01.cdt']
    pick_chs = ['O2', 'O1', 'OZ', 'PO8', 'PO7', 'PO6', 'PO5', 'PO4', 'PO3', 'POZ', 'P8', 'P7', 'P6', 'P5', 'P4', 'P3', 'PZ', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6']
    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=20,
        worker_name=feedback_worker_name)
    worker.pre()
    print('Attn_EEGNet模型和通道顺序已保存！')

