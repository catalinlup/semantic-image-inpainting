import torch.nn as nn

## Generator
class Generator(nn.Module):
    """
    Class implementing the generator network of the GAN system.
    """
    def __init__(self, ngpu=1, ngf=64, nz=100, nc=3):
        """
        Initializes a generator network.

        Keyword arguments:
        ngpu -- the number of gpus the network should run on.
        ngf -- the size of the generator's feature map (also the size of the generated image)
        nz -- the size of the input latent vector used to generate the image.
        nc -- the number of color channels for the generated image.
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 8 * ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        return self.main(input)