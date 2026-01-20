# 使用cuda做信号处理算法验证
import time
import numpy as np
import torch
import torch.fft as fft
import cupy as cp

dim = 1024          # 数据维度
cycle_times = 20    # 循环次数
signal = np.random.randn(4096, 4096).astype(np.float32)

# 第1部分，使用numpy做fft
def fft_numpy(signal):
    fft_2d = np.fft.fft2(signal)
    fft_2d_shift = np.fft.fftshift(fft_2d)
    rd = np.abs(fft_2d_shift)
    return rd
start = time.time()
for i in range(0, cycle_times):
    rd_numpy = fft_numpy(signal)
end = time.time()
print('cpu time:{}s'.format(end-start))

# 第2部分，使用torch做fft
def fft_torch(signal):
    # 转移到GPU
    signal_torch = torch.tensor(signal).cuda()
    # 执行二维FFT
    fft_2d = fft.fft2(signal_torch)
    # 频域中心化 (PyTorch旧版本可用torch.roll实现)
    fft_2d_shift = fft.fftshift(fft_2d)
    # 计算幅度谱
    rd = torch.abs(fft_2d_shift)
    return rd

start = time.time()
for i in range(0, cycle_times):
    rd_torch = fft_torch(signal)
end = time.time()
print('torch time:{}s'.format(end-start))

# 第3部分，使用cupy做fft
def fft_cupy(signal):
    # 将数据转移到GPU
    signal_cupy = cp.asarray(signal)
    # 在GPU上计算FFT
    fft_2d = cp.fft.fft2(signal_cupy)
    rd = cp.abs(fft_2d)
    # 将结果传回CPU
    # rd_cupy = cp.asnumpy(rd)
    return rd
start = time.time()
for i in range(0, cycle_times):
    rd_cupy = fft_cupy(signal)
end = time.time()
print('cupy time:{}s'.format(end-start))




# def radar_fft_processing(radar_signal):
#     # radar_signal: (batch_size, height, width)
#     # 转移到GPU
#     signal_tensor = radar_signal.cuda()
    
#     # 执行二维FFT
#     fft_result = fft.fft2(signal_tensor)
    
#     # 频域中心化 (PyTorch旧版本可用torch.roll实现)
#     fft_shifted = fft.fftshift(fft_result)
    
#     # 计算幅度谱
#     magnitude_spectrum = torch.abs(fft_shifted)
    
#     return magnitude_spectrum, fft_shifted

# # 使用示例
# batch_size, H, W = 4, 256, 256
# simulated_radar = torch.randn(batch_size, H, W)
# magnitude, complex_spectrum = radar_fft_processing(simulated_radar)

