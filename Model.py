
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 40, 3, stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(40, 89, 3, stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(89, 112, 3, stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(112, 142, 3, stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(5112, 1024)
        self.fc2 = nn.Linear(1024, 200)
        # self.fc3 = nn.Linear(256,64)
        self.fc4 = nn.Linear(200, 10)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc4(x)
        return x


























































# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 24, 3)
#         self.conv2 = nn.Conv2d(24, 36, 3, stride=(1, 1), padding=(1, 1))
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(36, 42, 3, stride=(1, 1), padding=(1, 1))
#         self.conv4 = nn.Conv2d(42, 60, 3, stride=(1, 1), padding=(1, 1))
#         self.fc1 = nn.Linear(2160, 512)
#         self.fc2 = nn.Linear(512, 64)
#         self.fc3 = nn.Linear(64, 10)
#         self.drop1 = nn.Dropout(0.3)
#         self.drop2 = nn.Dropout(0.2)
#
#     def forward(self, x):
#         x = self.pool(self.conv1(x))
#         x = self.pool(self.conv2(x))
#         x = self.pool(self.conv3(x))
#         x = self.conv4(x)
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = self.drop2(x)
#         x = F.relu(self.fc2(x))
#         x = self.drop2(x)
#         x = self.fc3(x)
#         return x
#
#





