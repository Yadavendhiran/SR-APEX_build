import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            # layer 1
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            # layer 2
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            # layer 3
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            # layer 4
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(720,1280),mode="bilinear",align_corners=False),
            # layer 5
            nn.Conv2d(in_channels=8,out_channels=3,kernel_size=3,stride=1,padding=1),
        )

    def forward(self,x):
        x=self.encoder(x)
        return x
