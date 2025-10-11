import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNetDQN(nn.Module):
    def __init__(self, num_classes=4, channel=4, init_weights=True):
        super(AtariNetDQN, self).__init__()
        self.cnn = nn.Sequential( # (B, 3, 84, 84)
            nn.Conv2d(channel, 32, kernel_size=8, stride=4), # (B, 32, 20, 20)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (B, 64, 9, 9)
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (B, 64, 7, 7)
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print("before: ", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()))
        # x = x.float() / 255.
        # print("after: ", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()))
        x = x.float()
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

