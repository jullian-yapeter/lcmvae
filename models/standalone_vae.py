import torch
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
#   - Linear: out_dim: embed_dim                                    #
#####################################################################
class Encoder(nn.Module):
  def __init__(self, embed_dim):
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
            nn.Linear(512, embed_dim, bias=True)
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
                stride=4, padding=0, bias=True),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    return self.net(self.map(x).reshape(-1, 512, 1, 1))


class StandaloneVAE(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.checkpoint_file = config.checkpoint_file

        self.encoder = Encoder(2*config.embed_dim).to(self.device)
        self.decoder = Decoder(config.embed_dim).to(self.device)
                
        self.mse_criterion = nn.MSELoss(reduction="sum")
        self.prior = {
            "mean": torch.zeros(config.embed_dim, device=self.device),
            "log_sigma": torch.zeros(config.embed_dim, device=self.device)
        }
        self.encoder.apply(StandaloneVAE._init_vae_weights)
        self.decoder.apply(StandaloneVAE._init_vae_weights)




    def forward(self, x, caiption=None, use_epsilon=False):
        if use_epsilon is None:
            use_epsilon = self.config.use_epsilon 
        encoder_out = self.encoder(x)
        mean = encoder_out[:, :self.config.embed_dim]
        log_sigma = encoder_out[:, self.config.embed_dim:]
        z = mean
        if use_epsilon:
            epsilon = torch.randn(
                x.shape[0], self.config.embed_dim, device=self.device)
            z = mean + torch.exp(log_sigma) * epsilon
        decoder_out = self.decoder(z)
        return {
            "reconstruction": decoder_out,
            "mean": mean,
            "log_sigma": log_sigma
        }

    def loss(self, vae_outputs, target_images, beta):
        reconstruction_images = vae_outputs["reconstruction"]
        vae_mean = vae_outputs["mean"]
        vae_log_sigma = vae_outputs["log_sigma"]

        rec_loss = self.mse_criterion(
            target_images, reconstruction_images) / target_images.shape[0]
        kl_loss = torch.mean(torch.sum(
            StandaloneVAE.kl_divergence(
                vae_mean, vae_log_sigma, self.prior["mean"], self.prior["log_sigma"]), dim=1), dtype=torch.float)

        return (rec_loss + beta * kl_loss).type(torch.float), rec_loss, kl_loss

    def reconstruct(self, x):
        encoder_out = self.encoder(x)
        mean = encoder_out[:, :self.config.embed_dim]
        log_sigma = encoder_out[:, self.config.embed_dim:]
        decoder_out = self.decoder(mean)
        return {
            "reconstruction": decoder_out.view(-1, *self.config.im_dims),
            "mean": mean,
            "log_sigma": log_sigma
        }

    @staticmethod
    def kl_divergence(mu1, log_sigma1, mu2, log_sigma2):
        return (log_sigma2 - log_sigma1) + (torch.exp(log_sigma1) ** 2 + (mu1 - mu2) ** 2) \
            / (2 * torch.exp(log_sigma2) ** 2) - 0.5

    
    @staticmethod
    def _init_vae_weights(m):
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        except:
            pass

