import numpy as np
from dataset import RADIal
from loader import CreateDataLoaders
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset, DataLoader, random_split,Subset

dataset = RADIal(root_dir = './RADIal',difficult=True)

# pick-up randomly any sample
data = dataset.__getitem__(np.random.randint(len(dataset)))

image = data[0]
boxes = data[5]

fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(image)

for box in boxes:
    if(box[0]==-1):
        break # -1 means no object
    rect = Rectangle(box[:2]/2,(box[2]-box[0])/2,(box[3]-box[1])/2,linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)