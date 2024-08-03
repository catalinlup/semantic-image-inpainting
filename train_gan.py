import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from utils import load_training_parameters, weights_init
from Generator import Generator
from Discriminator import Discriminator
import time
import json

# load the parameters used for training the GAN
params = load_training_parameters()

# configure the dataset used for training the GAN, as well as the transforms to be applied to each of the image in the dataset
dataset = dset.ImageFolder(root=params['dataroot'],
                           transform=transforms.Compose([
                            transforms.Resize(params['image_size']),
                            transforms.CenterCrop(params['image_size']),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

# configure a data loader to load the images in batches
dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'],
                                         shuffle=True, num_workers=params['workers'])

# configure the device to do the training on.
device = torch.device("cuda:0" if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")


if not params['resume_from_last']:

    # Initialize the generator network
    netG = Generator(params['ngpu'], params['ngf'], params['nz'], params['nc']).to(device)
    # apply gaussian initialization of weights (as specified in the GAN paper)
    netG.apply(weights_init)

    # Initialize the discriminator network
    netD = Discriminator(params['ngpu'], params['ndf'], params['nc']).to(device)
    # apply gaussian initialization of weights (as specified in the GAN paper)
    netD.apply(weights_init)

else:
    print('Resuming from ' + params['last_model_name'])
    folder_name = f'{params["last_model_name"]}'

    netG = Generator(params['ngpu'], params['ngf'], params['nz'], params['nc']).to(device)
    netD = Discriminator(params['ngpu'], params['ndf'], params['nc']).to(device)


    netG.load_state_dict(torch.load(f'models/{folder_name}/{params["last_model_name"]}_generator.pt'))
    netD.load_state_dict(torch.load(f'models/{folder_name}/{params["last_model_name"]}_discriminator.pt'))


# Initialize the loss function
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

# Initialize the optimzers used to train the GAN
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))


## Training Loop for the GAN system

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(params['num_epochs']):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        output = netD(real_cpu).view(-1)
    
        
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item
        
        # train with all-fake batch
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        
        # Update D
        optimizerD.step()
        
        ## Update G network: maximize log(D(G(z)))
        
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        
        optimizerG.step()
        
        if i % 50 == 0:
            print(f'ErrD {errD}, ErrG {errG}, iter: {i}')
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        if (iters % 500 == 0) or ((epoch == params['num_epochs'] - 1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1
        
        
# Save the trained generator and discriminator

folder_name = f'{params["output_model_name"]}'
os.mkdir(f'models/{folder_name}')
# save the generator
torch.save(netG.state_dict(), f'models/{folder_name}/{params["output_model_name"]}_generator.pt')

# save the discriminator
torch.save(netD.state_dict(), f'models/{folder_name}/{params["output_model_name"]}_discriminator.pt')

# save the parameters used for training
# json.dump(params, open(f'models/{folder_name}/params.json'))
