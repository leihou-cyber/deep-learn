import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 特征提取模块：将输入数据进行一系列的卷积、激活函数和池化操作，用于提取输入数据的特征
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            # 非线性激活 inplace=True表示直接在原地进行操作，节省内存消耗
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        # 全连接层 分类器模块
        # 序列容器，按顺序组合了网络的各个层。
        # 容器中的层将按照顺序依次执行。
        self.classifier = nn.Sequential(
            # Dropout 层用于防止过拟合
            # p 参数是丢弃概率，表示在训练过程中丢弃输入单元的概率
            nn.Dropout(p=0.5),
            # 输入为上一层maxPooling3的输出
            nn.Linear(128 * 6 * 6, 2048),
            # 增加了网络的非线性
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # 将2048维的特征映射到分类的类别数量，即 num_classes
            # 输出的表示每个类别的预测概率，类别的个数
            nn.Linear(2048, num_classes),
        )
        # 初始化网络的权重
        if init_weights:
            self._initialize_weights()

    # self:实例对象，x:输入数据
    def forward(self, x):
        # 用于提取输入数据的特征信息
        x = self.features(x)
        # 特征数据 x 展平成一维张量
        # dim=0:batch;dim=1:channel;dim=2:height;dim=3:weight
        x = torch.flatten(x, start_dim=1)
        # 进行分类或回归等任务，用于最终的分类或回归输出
        x = self.classifier(x)
        return x

    # 初始化神经网络模型的权重
    def _initialize_weights(self):
        # 遍历所有层结构
        for m in self.modules():
            # 判断是否为卷积层
            if isinstance(m, nn.Conv2d):
                # 对权重w进行 kaiming正态分布初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 检查当前卷积层是否有偏置项
                if m.bias is not None:
                    # 有则初始化为常数项0
                    nn.init.constant_(m.bias, 0)
            # 全连接层
            elif isinstance(m, nn.Linear):
                # m.weight：表示需要初始化的权重参数
                # 0：表示正态分布的均值
                # 0.01：表示正态分布的标准差
                nn.init.normal_(m.weight, 0, 0.01)
                # 将线性层的偏置项初始化为常数值0
                nn.init.constant_(m.bias, 0)
