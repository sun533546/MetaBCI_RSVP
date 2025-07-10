import sys
sys.path.insert(0, r"C:\Users\HP\Desktop\meta-rsvp\MetaBCI")
import math

from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (
    RSVP,
    paradigm,
    pix2height,
    code_sequence_generate,

)
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix

if __name__ == "__main__":
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器宽度尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix([1920, 1080])  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([1920,1080])
    # esc/q退出开始选择界面    
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口；False非全窗口显示，此时显示win_size参数默认屏幕分辨率
        record_frames=False, # 不记录实验每一帧
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    
    win = ex.get_window()

    # q退出范式界面

    """
    RSVP
    """
   
    image_dir = r"C:\Users\HP\Desktop\meta-rsvp\images"  # 顶层目录，plane/non_plane作为子文件夹
    image_size = (500, 500)
    nrep = 3
    bg_color = np.array([-1, -1, -1])
    display_time = 1.0     # block开始提示时间
    rest_time = 2.0        # block结束休息时间
    port_addr = "COM5"                  #   无端口：None；  NeuroScan :"COM5"; 
    online = False         # True/False，根据需求切换
    lsl_source_id = "meta_online_worker"   # 或"你的LSL-ID"，仅在线需要

    # ==== 3. RSVP 类实例化与图片配置 ====
    basic_RSVP = RSVP(win=win)
    basic_RSVP.config_image(
        image_dir=image_dir,
        image_size=image_size,
    )

    # ==== 4. 注册 RSVP 范式到实验对象 ====
    ex.register_paradigm(
        "basic_RSVP",               # 范式名称
        paradigm,                   # 范式主控函数（保持不变）
        VSObject=basic_RSVP,
        bg_color=bg_color,
        display_time=display_time,
        rest_time=rest_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="rsvp",
        online=online,
        lsl_source_id=lsl_source_id,
        device_type="NeuroScan"
        
    )

    # ==== 5. 启动主实验流程 ====
    ex.run()


