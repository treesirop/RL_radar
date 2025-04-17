# -*- coding: utf-8 -*-
# !/usr/bin/python
import os
class DefaultConfigure(object):
    # data path
    radar_train_data_root = '/home/wkl/Pixel/dataset_path/path_train_radar.txt'  # 修改
    radar_val_data_root = '/home/wkl/Pixel/dataset_path/path_test_radar.txt'
    radar_test_data_root = '/home/wkl/Pixel/dataset_path/path_test_radar.txt'


    # checkpoint path
    model = 'full'
    load_model_path = model +'/checkpoints'
    os.makedirs(load_model_path,exist_ok=True)
    checkpoint_model = None  # 'xxx.pth'
    # optimizer_state
    load_optimizer_path =  model + '/optimizer_state'
    os.makedirs(load_optimizer_path,exist_ok=True)
    
    optimizer = None  # 'xxx.pth'
    load_lossfunc_path = model + '/loss_func_state'
    os.makedirs(load_lossfunc_path,exist_ok=True)
    
    loss_func = None  # 'xxx.pth'

    use_gpu = True
    device = '1'

    batch_size = 4
    num_workers = 4
    display = 100
    snapshot = 4000
    load_file_count = 20  # 每次加载文件个数

    max_epoch = 10
    # max_iter = 100000
    lr = 0.01 
    momentum = 0.9
    weight_decay = 1e-4
    eps = 1e-5  # 防止分母为0

    result_file = model + '/result'
    os.makedirs(result_file, exist_ok=True)
    
    log_name = 'train.log'


    try:
        # 递归创建结果目录（包含所有父目录）
        os.makedirs(result_file, exist_ok=True)  # [3,6](@ref)
        print(f"目录创建成功：{result_file}")
        
        # 创建日志文件路径
        log_path = os.path.join(result_file, log_name)  # [8](@ref)
        
        # 自动创建日志文件（如果不存在）
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:  # [5](@ref)
                f.write(f"Training log created at {os.path.getctime(log_path)}\n")
            print(f"日志文件已创建：{log_path}")
        else:
            print(f"日志文件已存在：{log_path}")

    except OSError as e:
        print(f"操作失败：{e.strerror}")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")
    prop = 1  # Sample proportion

    time_steps = [30, 60, 90, 120]
    thresholds = (25, 35, 45, 55)
    balancing_weights = (1, 2, 5, 10, 40)
    
