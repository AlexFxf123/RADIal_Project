# 读取文件的测试代码
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import timeit
import cv2
from rpl import RadarSignalProcessing
import sys
# 导入DBReader的目录
sys.path.append('D:/RADIal_Project/DBReader')
from DBReader.DBReader import SyncReader
def ConpensateLayerAngle(pcl,index,sensor_height):
    
    offset=0
    if(index%2==0):
        offset = np.deg2rad(.6)

    x = pcl[:,4] * np.cos(pcl[:,5]+offset) * np.cos(pcl[:,6])
    y = pcl[:,4] * np.cos(pcl[:,5]+offset) * np.sin(pcl[:,6])
    z = pcl[:,4] * np.sin(pcl[:,5]+offset) + sensor_height
    
    pcl[:,0] = x
    pcl[:,1] = y
    pcl[:,2] = z
    
    return pcl

radar_height = 0.8
lidar_height = 0.42

root_folder = 'D:/RADIal_Project/RADIal/raw_sequences/RECORD@2020-11-22_12.28.47'
db = SyncReader(root_folder,tolerance=40000)

sample = db.GetSensorData(68)
image = sample['camera']['data']

# 显示图片
plt.figure(figsize=(10,10))
plt.imshow(image)




RSP = RadarSignalProcessing('D:/RADIal_Project/SignalProcessing/CalibrationTable.npy',method='PC')

pc=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],sample['radar_ch2']['data'],sample['radar_ch3']['data'])

# PC = [Range,Doppler,Azimuth,Elevation] m, m/s, rad, rad
radar_pts = pc
Az = pc[:,2]
R = pc[:,0]
El = pc[:,3]
El_cos = np.cos(El)
radar_pts[:,0] = R*El_cos*np.cos(Az)    # x
radar_pts[:,1] = R*El_cos*np.sin(Az)    # y
radar_pts[:,2] = R*np.sin(El) + radar_height         # Z
radar_pts[:,3] = pc[:,1]                # V


# Get the laser point cloud
pts = sample['scala']['data']
# Load the camera calibration parameters
calib = np.load('D:/RADIal_Project/DBReader/examples/camera_calib.npy',allow_pickle=True).item()
pts = ConpensateLayerAngle(pts,sample['scala']['sample_number'],lidar_height)[:,:3]
pts[:,[0, 1, 2]] = pts[:,[1, 0,2]] # Swap the order
pts[:,0]*=-1 # Left is positive
plt.figure(figsize=(10,10))
plt.plot(pts[:,0],pts[:,1],'.')
plt.xlim(-15,15)
plt.ylim(0,100)
plt.grid()


imgpts, _ = cv2.projectPoints(np.array(pts), 
                              calib['extrinsic']['rotation_vector'], 
                              calib['extrinsic']['translation_vector'],
                              calib['intrinsic']['camera_matrix'],
                              calib['intrinsic']['distortion_coefficients'])

imgpts=imgpts.squeeze(1).astype('int')

# Keep only points inside the image size
idx = np.where( (imgpts[:,0]>=0) & (imgpts[:,0]<image.shape[1]) & (imgpts[:,1]>=0) & (imgpts[:,1]<image.shape[0]))[0]


# 用雷达点云替代激光点云
radar_pts = radar_pts[:,[0,1,2]]
radar_pts[:,[0,1,2]] = radar_pts[:,[1,0,2]] # Swap the order
radar_pts[:,0]*=-1 # Left is positive
plt.plot(radar_pts[:,0],radar_pts[:,1],'.',color='red')
plt.xlim(-15,15)
plt.ylim(0,100)
plt.grid()

imgpts, _ = cv2.projectPoints(np.array(radar_pts), 
                              calib['extrinsic']['rotation_vector'], 
                              calib['extrinsic']['translation_vector'],
                              calib['intrinsic']['camera_matrix'],
                              calib['intrinsic']['distortion_coefficients'])

imgpts=imgpts.squeeze(1).astype('int')

# Keep only points inside the image size
idx = np.where( (imgpts[:,0]>=0) & (imgpts[:,0]<image.shape[1]) & (imgpts[:,1]>=0) & (imgpts[:,1]<image.shape[0]))[0]

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.plot(imgpts[idx,0],imgpts[idx,1],'r.')



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