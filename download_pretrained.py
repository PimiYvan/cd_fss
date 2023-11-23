
from torchvision.models import resnet
from torchvision.models import vgg


if __name__ == '__main__':
    backbone1 = vgg.vgg16(pretrained=True)
    backbone2 = resnet.resnet50(pretrained=True)
    print('pretrained donwloaded')

