import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
from utils import load_inpainting_parameters
from Generator import Generator
from Discriminator import Discriminator
import matplotlib.pyplot as plt
from PIL import Image
from UnNormalize import UnNormalize




netG = torch.load(f'models/celeb4/celeb4_generator.pt')
netD = torch.load(f'models/celeb_200/celeb_200_discriminator.pt')

fixed_noise = torch.randn(1, 100, 1, 1, device='cuda')
res = netG(fixed_noise)
print(res * 0.5 + 0.5)

plt.imshow(res.cpu().detach()[0].permute(1, 2, 0) * 0.5 + 0.5)
plt.show()


# img_transforms = transforms.Compose([
#                             transforms.Resize(64),
#                             transforms.CenterCrop(64),
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# INPUT_FOLDER = 'image_inpainting_input'
# img = Image.open(f'{INPUT_FOLDER}/cool_picture.png')
# img_transformed = img_transforms(img).to('cuda').reshape(1, 3, 64, 64)

# print(img_transformed.shape)
# print(netD(img_transformed))
# print(netD(res))

# # plt.imshow(netG(fixed_noise).cpu().detach()[0].permute(1, 2, 0))
# unorm = lambda x: x * 0.5 + 0.5
# img_transformed = unorm(img_transformed)
# plt.imshow(img_transformed.cpu().detach()[0].permute(1, 2, 0))
# plt.show()
