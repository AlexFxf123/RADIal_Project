# 读取文件的测试代码
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import timeit
from SignalProcessing.rpl import RadarSignalProcessing
import sys
# 导入DBReader的目录
sys.path.append('D:/RADIal_Project/DBReader')
from DBReader.DBReader import SyncReader

root_folder = 'D:/RADIal_Project/RADIal/raw_sequences/RECORD@2020-11-22_12.28.47'
db = SyncReader(root_folder,tolerance=40000)

sample = db.GetSensorData(68)


# 显示图片
plt.figure(figsize=(10,10))
plt.imshow(sample['camera']['data'])

# Get the laser point cloud， 显示激光点云
pts = sample['scala']['data']
plt.figure(figsize=(10,10))
plt.plot(-pts[:,1],pts[:,0],'.')
plt.xlim(-15,15)
plt.ylim(0,100)
plt.grid(True)


RSP = RadarSignalProcessing('D:/RADIal_Project/SignalProcessing/CalibrationTable.npy',method='PC')

pc=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],sample['radar_ch2']['data'],sample['radar_ch3']['data'])

# PC = [Range,Doppler,Azimuth,Elevation] m, m/s, rad, rad
Az = pc[:,2]
R = pc[:,0]
El = pc[:,3]
z = R*np.sin(El)
El_cos = np.cos(El)
x = R*El_cos*np.cos(Az)
y = R*El_cos*np.sin(Az)
v = pc[:,1]

plt.plot(-y,x,'.',color='red')
plt.xlim(-15,15)
plt.ylim(0,100)
plt.grid(True)

# choose move point
# choose_x = []
# choose_y = []
# choose_v = []
# len = np.shape(R)[0]
# for i in range(0,len):
#     if (abs(v[i]) > 1):
#         choose_x.append(x[i])
#         choose_y.append(y[i])
#         choose_v.append(v[i])

# plt.figure(figsize=(10,10))
# plt.plot(y,x,'o',color='red')
# plt.grid(True)

# SteeringWheel,YawRate,VehSpd = 


plt.show()

test = 1