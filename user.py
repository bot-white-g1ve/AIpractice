import torch
import torchvision
from torch.utils.data import DataLoader
from vgg16 import Tudui

tudui = torch.load("tudui25.pth", map_location=torch.device('cpu'))

datas = torchvision.datasets.CIFAR10("CIFAR10Data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
label_names = datas.classes
toPIL = torchvision.transforms.ToPILImage()
'''
#查看前十张图片
for idx, data in enumerate(datas):
    dataImg = toPIL(data[0])
    dataImg.show()
    print(label_names[data[1]])
    if idx == 9:
        break
'''

#验证第n张图片的输出和实际
data = datas[675]
img = toPIL(data[0])
img.show()
tensor = torch.reshape(data[0], (1,3,32,32))
out = tudui(tensor)
idx = torch.argmax(out, dim=1).item()
print(label_names[idx])
#print(label_names[data[1]])
