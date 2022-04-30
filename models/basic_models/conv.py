import torch
import torch.nn as nn
from numpy import sqrt


# TODO: convert hidden channels into params
class ConvDecoder512(nn.Module):
  """
  Convoluted Decoder for LCMVAE. (Temporary)
  in_dim:  
  """
  def __init__(self, embed_dim):
    super().__init__()
    self.map = nn.Linear(embed_dim, 512, bias=True)   # for initial Linear layer
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
                stride=4, padding=0, bias=True)
    )
  
  def forward(self, x):
    return self.net(self.map(x).reshape(-1, 512, 1, 1))


class PreConvLayer(nn.Module):
  def __init__(self, embed_dim, device=None):
    super().__init__()
    self.checkpoint_file = "pre_conv_layer"
    self.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.embed_dim = embed_dim
    self.net = nn.Sequential(
        nn.Conv2d(self.embed_dim, 256,
                  kernel_size=2,
                  stride=2,
                  padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.Conv2d(256, 256,
                  kernel_size=1,
                  stride=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.Conv2d(256, 12,
                  kernel_size=1,
                  stride=1),
    ).to(self.device)

  def forward(self, x):
    language_embedding = x[:, -self.embed_dim:]
    image_embedding_temp = x[:, :-self.embed_dim]
    image_embedding = torch.zeros((image_embedding_temp.shape[0], 196*768))
    image_embedding[:, :image_embedding_temp.shape[1]] = image_embedding_temp
    image_embedding = image_embedding.reshape(
        image_embedding.shape[0], -1, self.embed_dim).permute(0, 2, 1)
    batch, channels, vector_size = image_embedding.shape
    width = int(sqrt(vector_size))
    image_embedding = self.net(
        image_embedding.view(batch, channels, width, width)).reshape(-1, self.embed_dim)
    return torch.cat((image_embedding, language_embedding), dim=-1)


class ConvDecoder768(nn.Module):
  """
  Convoluted Decoder for LCMVAE
  in_dim:
  """

  def __init__(self, embed_dim, out_channels, device=None):
    super().__init__()
    # for initial Linear layer
    self.checkpoint_file = "conv_decoder_768"
    self.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    self.map = nn.Linear(embed_dim, 768, bias=True).to(device)
    self.upsample_1 = nn.Upsample(size=(9, 9)).to(device)
    self.upsample_2 = nn.Upsample(size=(17, 17)).to(device)
    self.upsample_3 = nn.Upsample(size=(33, 33)).to(device)
    self.upsample_4 = nn.Upsample(size=(49, 49)).to(device)
    self.upsample_5 = nn.Upsample(size=(224, 224)).to(device)
    self.net_1 = nn.Sequential(
        nn.BatchNorm2d(768),
        nn.ConvTranspose2d(in_channels=768, out_channels=128, kernel_size=3,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                           stride=1, padding=0, bias=True),
        nn.LeakyReLU(),
    ).to(device)
    self.net_2 = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                           stride=1, padding=0, bias=True),
        nn.LeakyReLU(),
    ).to(device)
    self.net_3 = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                           stride=1, padding=0, bias=True),
        nn.LeakyReLU(),
    ).to(device)
    self.net_4 = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                           stride=1, padding=0, bias=True),
        nn.LeakyReLU(),
    ).to(device)
    self.net_5 = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5,
                           stride=1, padding=0, bias=True),
        nn.LeakyReLU(),
    ).to(device)
    self.net_6 = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5,
                           stride=1, padding=0, bias=True),
        nn.LeakyReLU(),
    ).to(device)
    self.net_7 = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5,
                           stride=1, padding=0, bias=True),
        nn.LeakyReLU(),
    ).to(device)
    self.net_8 = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5,
                           stride=1, padding=0, bias=True),
        nn.LeakyReLU(),
    ).to(device)
    self.net_9 = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5,
                           stride=1, padding=0, bias=True),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=5,
                           stride=1, padding=0, bias=True),
    ).to(device)

  def forward(self, x):
    out_1 = self.net_1(self.map(x).reshape(-1, 768, 1, 1))
    out_2 = self.upsample_1(out_1) + self.net_2(out_1)
    out_3 = self.net_3(out_2)
    out_4 = self.upsample_2(out_3) + self.net_4(out_3)
    out_5 = self.net_5(out_4)
    out_6 = self.upsample_3(out_5) + self.net_6(out_5)
    out_7 = self.net_7(out_6)
    out_8 = self.upsample_4(out_7) + self.net_8(out_7)
    return self.upsample_5(self.net_9(out_8))


# TODO: TESTING NEEDED
class DirectDeconvDecoder(nn.Module):
  """
  Convoluted Decoder for LCMVAE.
  in_dim:  
  """
  def __init__(self, embed_dim):
    super().__init__()
    self.map = nn.Linear(embed_dim, 1536, bias=True)   # for initial Linear layer
    self.net = nn.Sequential(
      nn.BatchNorm2d(1536),
      nn.LeakyReLU(),      
      nn.ConvTranspose2d(in_channels=1536, out_channels=768, kernel_size=3,
                stride=1, padding=0, bias=True),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=768, out_channels=256, kernel_size=3,
                stride=1, padding=0, bias=True),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=256, out_channels=84, kernel_size=3,
                stride=2, padding=0, bias=True),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=84, out_channels=42, kernel_size=4,
                stride=1, padding=0, bias=True),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=42, out_channels=21, kernel_size=4,
                stride=2, padding=0, bias=True),
      nn.BatchNorm2d(16), 
      nn.LeakyReLU(),
      nn.ConvTranspose2d(in_channels=21, out_channels=3, kernel_size=6,
                stride=2, padding=0, bias=True)
    )
  
  def forward(self, x):
    return self.net(self.map(x).reshape(-1, 1536, 1, 1))
