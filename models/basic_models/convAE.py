import torch.nn as nn
#####################################################################
# Encoder Architecture: in_dims = (3, 224, 224)                     #
#   - Conv2d, kernel: 16, stride: 4, pad=0, out_dims: (32, 56, 56)  #
#   - LeakyReLU                                                     #
#   - Conv2d, kernel: 8, stride: 3, pad=0, out_dims: (64, 16, 16)   #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - Conv2d, kernel: 4, stride: 2, pad=0, out_dims: (128, 7, 7)    #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - Conv2d, kernel: 3, stride: 2, pad=0, out_dims: (256, 3, 3)    #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - Conv2d, kernel: 3, stride: 1, pad=0, out_dims: (512, 1, 1)    #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - Flatten                                                       #
#   - Linear, output units: nz (= representation dimensionality)    #
#####################################################################
class Encoder(nn.Module):
  def __init__(self, nz):
    super().__init__()
    self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=16,
                  stride=4, padding=0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8,
                      stride=3, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                      stride=2, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                      stride=2, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                        stride=1, padding=0 , bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(512, nz, bias=True)
        )
  
  def forward(self, x):
    return self.net(x)

    
#####################################################################
# Decoder Architecture (mirrors encoder architecture):              #
#   - Linear, output units: 512                                     #
#   - Reshape, output shape: (512, 1, 1)                            #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - ConvT2d, out_dim = (256, 3, 3), stride: 2, kernel: 3          #
#   - BatchNorm2d                                                   #
#   - LeakyReLU                                                     #
#   - ConvT2d, out_dim = (128, 7, 7), stride: 4, kernel: 3          #
#   - ...                                                           #
#   - ...                                                           #
#   - ConvT2d, out_dim = (3, 224, 224), stride: 4, kernel: 16       #
#   - Sigmoid (to limit output in range [0...1])                    #
#####################################################################
class Decoder(nn.Module):
  def __init__(self, nz):
    super().__init__()
    self.map = nn.Linear(nz, 512, bias=True)   # for initial Linear layer
    self.net = nn.Sequential(
      nn.BatchNorm2d(512),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,
                stride=1, padding=0, bias=True),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                stride=2, padding=0, bias=True),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                stride=2, padding=0, bias=True),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=8,
                stride=3, padding=0, bias=True),
      nn.BatchNorm2d(32), 
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=16,
                stride=4, padding=0, bias=True),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    return self.net(self.map(x).reshape(-1, 512, 1, 1))