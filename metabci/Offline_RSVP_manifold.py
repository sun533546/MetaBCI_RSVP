import numpy as np
import mne
from mne.preprocessing import ICA
from brainflow.amplifiers import Marker
from brainflow.workers import ProcessWorker
from mne.io import read_raw_curry
import joblib
from metabci.brainda.algorithms.manifold.riemann import Alignment
from metabci.brainda.algorithms.manifold.riemann import MDRM
from sklearn.metrics import accuracy_score


def read_data(run_files, chs, interval, labels):
    """
    数据加载与预处理
    """
    
    for run_file in run_files:
        raw = read_raw_curry(run_file, preload=True, verbose=False)
        raw.pick_channels(chs)
        
        raw.resample(250)

        raw.notch_filter(freqs=50, method='spectrum_fit')  
        raw.filter(l_freq=1, h_freq=30, method='fir', fir_window='hamming')  


        ch = raw.info['ch_names']
        ica = ICA(n_components=23, method='fastica', random_state=42)
        
        # 2. 拟合数据
        ica.fit(raw)

        # 5. 应用 ICA 去除伪迹
        ica.apply(raw)

        # 从标注中提取事件及事件ID
        events, event_id = mne.events_from_annotations(raw)

        # 定义处理窗口并初始化结果变量
        labels_for_targ = []
        train_trials = []

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
        y = np.array(labels_for_targ).squeeze()  # (n_trials,)
    
    return X, y, ch



def train_model(X, y, srate=250):
    """
    训练模型并保存为.pkl文件
    """
    # 数据对齐
    aligner = Alignment(align_method='riemann')
    X_aligned = aligner.fit(X).transform(X)  # 对齐训练数据

    # 使用MDRM分类器
    clf = MDRM()
    clf.fit(X_aligned, y)

    # 保存对齐器和模型
    joblib.dump(aligner, 'riemann_aligner.pkl')  # 保存对齐器
    joblib.dump(clf, 'manifold_model.pkl')  # 保存模型

    print("Model trained and saved successfully!")
    return clf, aligner  # 返回训练好的模型和对齐器


def model_predict(X, srate=250, model=None):
    """
    用训练好的模型进行预测
    """

    for i in range(X.shape[0]):
        avg = np.mean(X[i,:,:],axis=-1,keepdims = True)
        std = np.std(X[i,:,:],axis=-1,keepdims=True)
        X[i,:,: ]=(X[i,:,:] - avg)/std

    # 对数据进行Riemannian对齐
    aligner = Alignment(align_method='riemann')
    X_aligned = aligner.fit(X).transform(X)

    # 预测
    y_pred = model.predict(X_aligned)

    return y_pred

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
        # 读取数据并训练模型
        X, y, ch_ind = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 labels=self.stim_labels)           # 读取数据
        print("Loading training data successfully")
        
        # 训练并保存模型
        self.estimator = train_model(X, y, srate=250)
        self.ch_ind = ch_ind
        print('Connected')

    def consume(self, data):
        pass

    def post(self):
        pass


if __name__ == '__main__':
    # Sample rate EEG amplifier
    srate = 1000
    # Data epoch duration
    acc_list = []
    
    stim_interval = [0, 0.8]
    # Label
    stim_labels = [101, 103]
    # Data path
    run_files = [r'C:\Users\TAIIC\Desktop\meta-rsvp\MetaBCI\metabci\brainda\subject_1\Acquisition 01.cdt']
    pick_chs = ['O2', 'O1', 'OZ', 'PO8', 'PO7', 'PO6', 'PO5', 'PO4', 'PO3', 'POZ', 'P8', 'P7', 'P6', 'P5', 'P4', 'P3', 'PZ', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6']
    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    # 训练模型
    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=20,
        worker_name=feedback_worker_name)

    # 在线数据环形缓冲区
    marker = Marker(interval=stim_interval, srate=srate, events=stim_labels)  # 获取epoch索引

    acc = worker.pre()


    
    print('模型已保存为 manifold_model.pkl')
