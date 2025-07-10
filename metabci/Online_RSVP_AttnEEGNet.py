import sys
sys.path.insert(0, r"C:\Users\TAIIC\Desktop\meta-rsvp\MetaBCI")
import numpy as np
import torch
import pickle
from scipy import signal
from metabci.brainflow.workers import ProcessWorker
from metabci.brainflow.amplifiers import Marker, Curry8
import time

from brainda.algorithms.deep_learning.Attn_EEGNet import EEGNetAttn

# 带通滤波函数
def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2 * freq0 / srate
    wn2 = 2 * freq1 / srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new

def resample(data, up, down):
    """ 简单的重采样方法（升采样和降采样） """
    return signal.resample(data, int(data.shape[-1] * up / down), axis=-1)

class FeedbackWorker(ProcessWorker):
    def __init__(self,
                 model,
                 channel_names,
                 stim_interval,
                 stim_labels,
                 srate,
                 lsl_source_id,
                 timeout,
                 worker_name):
        self.model = model
        self.channel_names = channel_names
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def consume(self, data):
        if data is None:
            print("No data received.")
            return None
        else:
            print(f"Received data.")

        # 转成 numpy，并保证 shape = (n_channels, n_times)
        data = np.array(data, dtype=np.float32).T

        # 选择特定通道
        ch_indices = [self.channel_names.index(ch) for ch in self.channel_names]
        data_selected = data[ch_indices, :]

        # 降采样到250Hz（假设原始是1000Hz）
        data_selected = resample(data_selected, up=250, down=1000)

        # 带通滤波1-30Hz
        data_selected = bandpass(data_selected, 1, 30, 250)

        # 标准化（和离线一致，按每个通道减均值除标准差）
        data_selected = (data_selected - np.mean(data_selected, axis=-1, keepdims=True)) / (np.std(data_selected, axis=-1, keepdims=True) + 1e-6)

        # 加入batch和channel维度，shape = (1, 1, n_channels, n_times)
        data_tensor = torch.tensor(data_selected[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self.model(data_tensor)
            pred = torch.argmax(output, dim=1).item()
        print(f"Predicted label: {pred}")
        return pred

    def post(self):
        pass

if __name__ == '__main__':
    # === 参数区 ===
    pick_chs = ['O2', 'O1', 'OZ', 'PO8', 'PO7', 'PO6', 'PO5', 'PO4', 'PO3', 'POZ', 'P8', 'P7', 'P6', 'P5', 'P4', 'P3', 'PZ', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6']
    srate = 1000
    stim_interval = [0, 0.8]
    stim_labels = [101, 103]
    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    # === 加载模型和通道顺序 ===
    model = EEGNetAttn(num_classes=2, in_channels=1, samples=200)  # 200是窗口长度（800ms@250Hz）
    model.load_state_dict(torch.load(r'attn_eegnet_rsvp.pth', map_location='cpu'))
    with open('attn_eegnet_ch_names.pkl', 'rb') as f:
        channel_names = pickle.load(f)

    # === Worker ===
    worker = FeedbackWorker(model=model,
                            channel_names=channel_names,
                            stim_interval=stim_interval,
                            stim_labels=stim_labels,
                            srate=srate,
                            lsl_source_id=lsl_source_id,
                            timeout=20,
                            worker_name=feedback_worker_name)

    marker = Marker(interval=stim_interval, srate=srate, events=stim_labels)
    worker.pre()

    # method = 'Curry8' 
    ns = Curry8(device_address=('192.168.31.14', 4455), srate=srate, num_chans=69)

    ns.connect_tcp()
    ns.start_acq()
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(2)
    ns.start_trans()
    print('Online processing started, waiting for data...')

    input('press any key to close\n')
    ns.down_worker('feedback_worker')
    time.sleep(1)
    ns.stop_trans()
    ns.stop_acq()
    ns.close_connection()
    ns.clear()
    print('bye')







