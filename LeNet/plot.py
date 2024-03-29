from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

''' torchvision是一个包含常用数据集、模型架构和图像转换工具的库，用于计算机视觉任务。
FashionMNIST是一个流行的时尚物品数据集，包含了10个类别的70,000张灰度图像，
每张图像的尺寸是28x28像素。这个数据集常用于测试机器学习模型的性能。'''

'''transforms模块提供了一系列用于图像预处理的工具。这些工具可以调整图像的尺寸、标准化图像数据等，
帮助提高模型的性能。在加载数据集时，可以通过transform参数将这些预处理步骤应用到每个图像上'''

'''torch.utils.data模块包含了数据加载和批处理功能，如DataLoader类，它可以从一个数据集中批量加载数据，支持多线程加载和数据打乱等功能'''

train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

'''显示数据'''
# 获得一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
batch_y = b_y.numpy()  # 将张量转换成Numpy数组
class_label = train_data.classes  # 训练集的标签
# print(class_label)
print("The size of batch in train data:", batch_x.shape)  # 每个mini-batch的维度是64*224*224

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()
