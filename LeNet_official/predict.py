import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    # 对图像预处理 将输入的图像处理成合法规则大小
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         # 将PIL图像转化成tensor(H x W x C)
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('2.jpg')
    im = transform(im)  # [C, H, W]
    # 最前面添加一个参数
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
