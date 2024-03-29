import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 预处理数据
    # RandomResizedCrop随机裁剪；RandomHorizontalFlip随机翻转
    # Resize：重新标定尺寸
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # get data root path
    # 数据集根目录的绝对路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # flower data set path
    # 包含花卉数据集的路径
    image_path = os.path.join(data_root, "data_set", "flower_data")
    # 一个断言，用于检查 image_path 是否存在
    # 如果该路径不存在，会抛出一个 AssertionError，显示路径不存在的消息
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 训练集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # 训练数据集中的样本数量
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # 获取了训练数据集中每个类别对应的索引字典
    flower_list = train_dataset.class_to_idx
    # 将键值对对调后存储到新字典中去
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    # 将字典 cla_dict 转换为 JSON 格式的字符串，并且设置了缩进为 4 个空格
    json_str = json.dumps(cla_dict, indent=4)
    # 打开一个名为 'class_indices.json' 的文件以写入数据
    # 如果文件不存在，将会被创建
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 每一批为32个图片
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    # num_classes=5 指定了模型输出的类别数量为 5
    # init_weights=True 表示使用预训练的权重进行初始化
    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    # 交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './AlexNet_classicl.pth'
    # 最佳准确率为 0.0
    best_acc = 0.0
    # 每个 epoch 中训练数据的批次数
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train：使用dropout方法 管理dropout方法
        # 将模型设置为训练模式，这样模型中的 Dropout 层等会起作用
        net.train()
        running_loss = 0.0
        # 创建了一个 tqdm 进度条，用于可视化训练进度
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

            # eval：禁用dropout方法
            # 将模型设置为评估模式，这样会关闭 Dropout 层等，以确保在验证过程中使用整个模型
            net.eval()
            # 正确预测数量
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                # 创建了一个 tqdm 进度条，用于可视化验证进度
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    # 选择每个样本中概率最大的类别作为预测结果
                    predict_y = torch.max(outputs, dim=1)[1]
                    # 模型预测正确的样本数量累加到 acc 变量中
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
