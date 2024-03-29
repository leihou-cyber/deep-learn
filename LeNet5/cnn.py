# 导入PyTorch库
import torch
import torch.nn as nn
import torch.optim as optim  # 包含各种优化算法
import torchvision  # 视觉库，提供一系列计算机视觉任务中常用的数据集，模型和转换
import torchvision.transforms as transforms  # 用于对图像进行预处理和转换模块
from torch.utils.data import DataLoader  # 加载数据集


# 定义LeNet-5架构神经网络
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层进行卷积运算，输出通道为6，有6个卷积核，提取了6个不同的特征
        # 卷积核（滤波器）大小为5
        # LeNet-5 中的卷积层帮助网络学习图像的低级特征，如边缘和纹理
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 最大池化层 2*2大小，步幅为2
        # 主要作用：压缩图像，减小数据尺寸，降低计算量，保留主要特征
        # 操作：取2*2的图像区域，取其中的最大值，平均池化则取平均值
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一层输出为6，第二层的输入为6，输出为16
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层 卷积层16*4*4
        # 前面两个最大池化层使图像尺寸减小到原来的 1/4
        # 全连接层通过学习权重将输入特征映射到输出类别
        # 并且在训练完成后将输出转换为概率分布，从而实现对输入图像的分类预测
        # 通过反向传播调整权重
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 120降维到84
        self.fc2 = nn.Linear(120, 84)
        # 最后输出10个类别，则为10
        self.fc3 = nn.Linear(84, 10)

    # 前向传播函数定义网络数据流向
    def forward(self, x):
        # relu执行了非线性映射 pool1执行最大池化
        # 输入执行第一次卷积运算
        x = self.pool1(torch.relu(self.conv1(x)))
        # 第二次输入为第一次输出 开始第二次卷积运算，输出16个不同的
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # 展平为一维
        # 全连接运算
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义数据变换和加载MNIST数据集，数据预处理
# 1.数据转换成tensor
# 2.Normalize把像素值进行了归一化，(0.5,) 是均值，(0.5,) 是标准差
# 这个操作将每个通道的像素值从 [0, 1] 的范围归一化到均值为 0、标准差为 1 的范围内。
# 归一化可以帮助模型更快地收敛，并且有助于避免梯度消失或梯度爆炸的问题
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 训练数据集
# 1.root数据集存放地址
# 2.train为True表示为训练集
# 3.download会自动下载训练集放在data文件中
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# 每个批次包含64个样本；shuffle=true,每个epoch开始，对数据进行随机打乱
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试数据集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 创建实例
net = LeNet5()
# 交叉熵损失函数，用于分类问题，PyTorch自带
criterion = nn.CrossEntropyLoss()
# Adam优化器，学习率为0.001，学习率越大，优化速度越快，但过大学习率，会使得在谷底反复横跳
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 在训练循环中添加计算测试集准确率的代码段
# 每个epoch代表一次前向传播与反向传播，即训练一次
for epoch in range(10):
    running_loss = 0.0
    # 启动循环，迭代训练集中的数据批次，i为索引，data为当前批次数据
    for i, data in enumerate(train_loader, 0):
        # 将当前数据解包为
        # inputs 包含神经网络的输入数据，labels 包含对应的真实标签。
        inputs, labels = data
        # 清除优化张量的梯度，不使用之前的梯度
        optimizer.zero_grad()
        # 通过神经网络训练得到outputs
        outputs = net(inputs)
        # 利用损失函数求损失值
        # 预测输出 outputs 和真实标签 labels 之间的损失值。
        loss = criterion(outputs, labels)
        # 再进行反向传播，优化内部参数
        loss.backward()
        # 根据前面计算的梯度更新模型的参数
        # 它执行一次优化步骤，根据优化器的更新规则进行参数更新。
        optimizer.step()
        # 计算epoch总损失
        running_loss += loss.item()
print("Finished Training")

# 测试
correct = 0  # 记录模型在测试集上预测正确的样本数量
total = 0  # 记录测试集中样本的总数量
# 上下文管理器，不会追踪梯度
with torch.no_grad():
    for data in test_loader:
        # inputs 包含神经网络的输入数据，labels 包含对应的真实标签。
        inputs, labels = data
        # 前向传播
        outputs = net(inputs)
        # torch.max() 函数找到每个样本预测值中的最大值及其索引
        # 取输出值的最大值的索引
        # torch.max(outputs.data, 1) 返回一个元组，第一个元素是最大值，第二个元素是最大值的索引
        # 用不需要的最大值 _ 进行占位，而将索引值赋给 predicted
        _, predicted = torch.max(outputs.data, 1)
        # 跟踪测试集中的总样本数量
        total += labels.size(0)
        # 比较了预测值 predicted 和真实标签 labels 是否相等
        # 将相等的数量累加到 correct 变量上
        correct += (predicted == labels).sum().item()

# 得出准确率
accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy}%")
print("Finished Testing")
