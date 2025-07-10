import sys
sys.path.insert(0, r"C:\Users\TAIIC\Desktop\meta-rsvp\MetaBCI")
import joblib
import numpy as np
from scipy import signal
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.manifold.riemann import Alignment
from metabci.brainflow.amplifiers import Marker, Curry8
import time
import socket

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
                 stim_interval,
                 stim_labels,
                 srate,
                 lsl_source_id,
                 timeout,
                 worker_name,
                 channel_names):  # 增加 channel_names 参数
        self.model = model
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.channel_names = channel_names  # 保存传入的通道名称
        super().__init__(timeout=timeout, name=worker_name)

        # 在线加载训练好的对齐器
        self.aligner = joblib.load(r'C:\Users\TAIIC\Desktop\meta-rsvp\riemann_aligner.pkl')  # 加载保存的对齐器

    def consume(self, data):

        if data is None:
            print("No data received.")
        else:
            print(f"Received data ")

        # 转换为np.float64类型并转置数据（确保数据维度为样本 x 通道 x 时间）
        data = np.array(data, dtype=np.float64).T  # 转置数据

        # 选择特定的通道（这里的 `channel_names` 是你需要传递的通道名称列表）
        ch_indices = [self.channel_names.index(ch) for ch in self.channel_names]  # 获取通道的索引
        data_selected = data[ch_indices, :]  # 提取选择的通道数据

        # 降采样（假设原始采样率较高，这里将其降到250Hz）
        data_selected = resample(data_selected, up=250, down=1000)  # 根据目标采样率调整

        # 带通滤波，去除50Hz工频噪声以及30Hz以上的高频噪声
        data_selected = bandpass(data_selected, 1, 30, self.srate)  # 带通滤波1-30Hz

        # 标准化：每个试次的每个通道减去均值，除以标准差
        X = (data_selected - np.mean(data_selected, axis=-1, keepdims=True)) / np.std(data_selected, axis=(-1, -2), keepdims=True)

        # 对数据进行Riemannian对齐
        X_aligned = self.aligner.transform(X)  # 使用已训练的对齐方法

        # 使用训练好的模型进行预测
        y_pred = self.model.predict(X_aligned)
        
        # 标签转换：将101转换为0，103转换为1
        if y_pred == 101:
            y_pred = 0
        elif y_pred == 103:
            y_pred = 1

        print(f"Predicted label: {y_pred}")
        # real_label = self.stim_labels[0]  # 获取实际标签
        # print(f"Real label: {real_label}")

        return y_pred

    def post(self):
        pass


# 主程序，初始化 FeedbackWorker 并开始数据采集
if __name__ == '__main__':
    # 假设这次训练使用的是 23 个通道名称
    pick_chs = ['O2', 'O1', 'OZ', 'PO8', 'PO7', 'PO6', 'PO5', 'PO4', 'PO3', 'POZ', 'P8', 'P7', 'P6', 'P5', 'P4', 'P3', 'PZ', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6']
    
    srate = 1000
    stim_interval = [0, 0.8]
    stim_labels = [101, 103]  # 假设你发送的标签是 101 和 103
    run_files = r'C:\Users\TAIIC\Desktop\meta-rsvp\manifold_model.pkl'
    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'
    
    # 加载训练好的模型
    model = joblib.load(run_files)
    
    # 创建 FeedbackWorker 实例时传入 channel_names
    worker = FeedbackWorker(model=model,
                            stim_interval=stim_interval,
                            stim_labels=stim_labels,
                            srate=srate,
                            lsl_source_id=lsl_source_id,
                            timeout=20,
                            worker_name=feedback_worker_name,
                            channel_names=pick_chs)  # 传递通道名称
    
    marker = Marker(interval=stim_interval, srate=srate, events=stim_labels) 
    worker.pre()
    
    # method = 'LSL' 
    method = 'Curry8' 

    ns = Curry8(device_address=('192.168.31.14', 4455), srate=srate, num_chans=69)  # 设置采样率和通道数

    # 与ns建立tcp连接
    ns.connect_tcp()
    # 开始采集波形数据
    ns.start_acq()
    # 注册 worker 来实现在线处理
    ns.register_worker(feedback_worker_name, worker, marker)
    # 开启在线处理进程
    ns.up_worker(feedback_worker_name)
    # 等待 0.5s
    time.sleep(0.5)
    # ns开始截取数据线程，并把数据传递数据给处理进程
    ns.start_trans()  # 启动数据传输
    print('Online processing started, waiting for data...')
  
    # 任意键关闭处理进程
    input('press any key to close\n')
    
    ns.down_worker('feedback_worker')
    time.sleep(1)
    ns.stop_trans()
    ns.stop_acq()
    ns.close_connection()
    ns.clear()
    print('bye')






