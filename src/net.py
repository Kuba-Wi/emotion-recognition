from abc import ABC, abstractmethod
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152


class BaseNet(ABC, nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass


class BasicNet(BaseNet):
    def __init__(self, classes_num: int):
        # initialize the network
        super(BasicNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, classes_num)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)  # Flatten the feature map
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicNetWithDropout(BaseNet):
    def __init__(self, classes_num: int):
        # initialize the network
        super(BasicNetWithDropout, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, classes_num)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)  # Flatten the feature map
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EmotionRecognitionModel(BaseNet):
    def __init__(self):
        super(EmotionRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(512 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv4(x))
        x = self.bn3(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv5(x))
        x = self.bn4(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.bn5(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.bn6(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class NetResnet152(BaseNet):
    def __init__(self, num_classes):
        super(NetResnet152, self).__init__()
        self.resnet = resnet152(num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)
