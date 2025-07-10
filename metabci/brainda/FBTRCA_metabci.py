import numpy as np
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices,
    match_kfold_indices)
from metabci.brainda.algorithms.decomposition import FBTRCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank

import mne
import numpy as np
import pandas as pd

cdt_file_path = r"C:\Users\HP\Desktop\8\Acquisition 01.cdt"

# 读取 .cdt 文件
raw = mne.io.read_raw_curry(cdt_file_path, preload=True)

# 输出信息查看
print(raw.info)

# 选择与SSVEP相关的通道
raw.pick_channels(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'PZ', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'POZ', 'O1', 'O2', 'OZ'])

# 设置采样率一致
raw.resample(250)

# 带通滤波
raw.filter(5, 90, l_trans_bandwidth=2, h_trans_bandwidth=5, phase='zero-double')

# 从注释中提取事件


events, event_id = mne.events_from_annotations(raw)

# 查看提取到的事件和事件ID
print(events)
print(event_id)

# 创建epochs，提取你感兴趣的时间段
epochs = mne.Epochs(raw, events, event_id, tmin=0.2, tmax=1.15, baseline=None, preload=True)

# 提取数据和标签
X = epochs.get_data()  # (n_epochs, n_channels, n_times)
y = epochs.events[:, -1]  # 标签

# 生成滤波器组（与之前一致）
wp = [(5, 90), (14, 90), (22, 90), (30, 90), (38, 90)]
ws = [(3, 92), (12, 92), (20, 92), (28, 92), (36, 92)]
filterbank = generate_filterbank(wp, ws, srate=250, order=3, rp=0.5)

# 设置6折交叉验证
kfold = 6
# indices = generate_kfold_indices(y, kfold=kfold)

indices = np.array([x for x in range(y.shape[0])]) # 

# 设置FBTRCA分类器
filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
estimator = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=np.array(filterweights), n_jobs=-1)

# 进行交叉验证并计算准确率
accs = []
for k in range(kfold):
    # train_ind, validate_ind, test_ind = match_kfold_indices(k, y, indices)
    # train_ind = np.concatenate((train_ind, validate_ind))
    
    np.random.shuffle(indices)
    train_ind = indices[:80]
    test_ind = indices[80:]

    p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
    accs.append(np.mean(p_labels == y[test_ind]))

# 输出平均准确率
print("Average accuracy: ", np.mean(accs))
print("accuracies: ", accs)









# import numpy as np
# from metabci.brainda.datasets import Wang2016
# from metabci.brainda.paradigms import SSVEP
# from metabci.brainda.algorithms.utils.model_selection import (
#     set_random_seeds,
#     generate_kfold_indices,
#     match_kfold_indices)
# from metabci.brainda.algorithms.decomposition import FBTRCA
# from metabci.brainda.algorithms.decomposition.base import generate_filterbank
# import mne

# wp=[(5,90),(14,90),(22,90),(30,90),(38,90)]
# ws=[(3,92),(12,92),(20,92),(28,92),(36,92)]

# filterbank = generate_filterbank(wp,ws,srate=250,order=15,rp=0.5)



# # cdt_file_path = r"F:\VR-SSVEP\BS3\ex1\10\Acquisition 01.cdt"

# # # 读取 .cdt 文件
# # raw = mne.io.read_raw_curry(cdt_file_path, preload=True)

# dataset = Wang2016()

# paradigm = SSVEP(
#     channels=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'PZ', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'POZ', 'O1', 'O2', 'OZ'],
#     intervals=[(0.14, 0.64)],
#     srate=250
# )


# # add 5-90Hz bandpass filter in raw hook
# def raw_hook(raw, caches):
#     # do something with raw object
#     raw.filter(5, 90, l_trans_bandwidth=2,h_trans_bandwidth=5,
#         phase='zero-double')
#     caches['raw_stage'] = caches.get('raw_stage', -1) + 1
#     return raw, caches


# def epochs_hook(epochs, caches):
#     # do something with epochs object
#     # print(epochs.event_id)
#     caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
#     return epochs, caches


# def data_hook(X, y, meta, caches):
#     # retrive caches from the last stage
#     # print("Raw stage:{},Epochs stage:{}".format(caches['raw_stage'], caches['epoch_stage']))
#     # do something with X, y, and meta
#     caches['data_stage'] = caches.get('data_stage', -1) + 1
#     return X, y, meta, caches


# paradigm.register_raw_hook(raw_hook)
# paradigm.register_epochs_hook(epochs_hook)
# paradigm.register_data_hook(data_hook)

# X, y, meta = paradigm.get_data(
#     dataset,
#     subjects=[1],
#     return_concat=True,
#     n_jobs=None,
#     verbose=False)
# # X, y, meta = paradigm.get_data(raw)
# # 6-fold cross validation
# set_random_seeds(38)
# kfold = 6
# indices = generate_kfold_indices(meta, kfold=kfold)
# accs = []
# # classifier
# filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
# estimator = FBTRCA(filterbank=filterbank,n_components = 1, ensemble = True,filterweights=np.array(filterweights), n_jobs=-1)
# for k in range(kfold):
#     train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
#     # merge train and validate set
#     train_ind = np.concatenate((train_ind, validate_ind))
#     p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])

#     accs.append(np.mean(p_labels==y[test_ind]))
# print(np.mean(accs))
# # If everything is fine, you will get the accuracy about 0.9417.
