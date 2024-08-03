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
from UnNormalize import UnNormalize

# load the parameters 
params = load_inpainting_parameters()


# configure the device to do the training on.
device = torch.device("cuda:0" if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")


# Setup Normalization Transform

# Setup Denormalization Transform

unorm = lambda x: x * 0.5 + 0.5


img_transforms = transforms.Compose([
                            transforms.Resize(params['image_size']),
                            transforms.CenterCrop(params['image_size']),
                            transforms.ToTensor()])


# load the networks
folder_name = f'{params["gan_model_name"]}'


netG = Generator(params['ngpu'], params['ngf'], params['nz'], params['nc']).to(device)
netD = Discriminator(params['ngpu'], params['ndf'], params['nc']).to(device)
  
netG.load_state_dict(torch.load(f'models/{folder_name}/{params["gan_model_name"]}_generator.pt'))
netD.load_state_dict(torch.load(f'models/{folder_name}/{params["gan_model_name"]}_discriminator.pt'))




## CRITICAL CODE (Searching for the Optimal Latent Vector)

# Computes the pixel weights to for context loss
# The operation is computed using a pytorch convolution operation for efficiency reasons
# The weight is computed by counting the number of unknown pixels with a certain window around each pixel and then dividing by the size of the window
# The operation is implemented by computing the convolution between a padded version of (1 - mask) and a filter of size window_size * window_size with all elements 1.
# Since unknown pixels are marked with 0 and known pixels with 1, (1 - mask) convoluted with the filter will compute the number of uknown neighbors for each pixel.
# The result of the convolution is then divided by the size of the mask to obtain the actual weights
def compute_weights_convolution(mask, window_size=7):
    # setup the convolution operation
    conv = nn.Conv2d(1, 1, window_size, padding=3).to(device)

    # set the weights of the convolution to 1 and the bias to 0 (essentially setting up an all 1 filter)
    with torch.no_grad():
        conv.weight.copy_(torch.ones(window_size * window_size).reshape((window_size, window_size)))
        conv.bias = nn.Parameter(torch.zeros((1,)).to(device))

    # compute the inverse of the mask
    inv_mask = 1 - mask

    # compute the convolution
    squeezed_mask = inv_mask.unsqueeze(0)
    res = conv(squeezed_mask)
    
    # return the result of the convolution divided by the window size. We also multiply by the mask to make sure that the unknown
    # pixels are being assigned a weight of 0 (we cannot compute the contextual loss on unknown pixels)
    return mask * res / (window_size * window_size)

# Computes the context loss as described in the paper. 
# The context loss is essentially the L1 norm between
# the predicted pixel values and the known pixel values of the inpainting target.
# Pixels positions that have more unknown neighbors in the inpainting target are assigned a higher weight,
# since predicting them right makes it more likely to predict the unknown pixels right.
def context_loss(z, y, W):
    return torch.mean(torch.abs(W * (unorm(netG(z)[0]) - y)))

# Computes the prior loss, as described in the paper.
# The prior loss is proportional to the probability that the image created by the generator is fake, according to the discriminator.
# This loss being low, represents that the image created by the generator is realistic
def prior_loss(z, lm):
    res = netD(netG(z))
    # criterion = nn.BCELoss()
    # return lm * criterion(res.view(-1), torch.full((1,), 0, dtype=torch.float, device=device))
    return lm * torch.log(1 - res)


# Combines context and prior loss, as described in the paper
def compute_loss(z, y, mask, lm):
    W = compute_weights_convolution(mask)
    return context_loss(z, y, W) + prior_loss(z, lm)





# function that generates a random vector to be used as the starting point
# for the optimization process
def generate_random_latent_vector():
    return torch.rand(1, params['nz'], 1, 1, device=device)


# Method that implements the process of searching for the optimal latent vector z
# The method returns the output image created by the generator based on the found optimal latent vector z
def optimize_latent_vector(target_image, target_image_mask, lm, num_iterations, lr):
  
    # start with a random latent vector
    z = generate_random_latent_vector()
    # configure the latent vector as a paramter to be optimzed
    z = torch.nn.Parameter(z, requires_grad=True)


    # use the adam optimizer to optimize z with respect to the loss between
    # the generated image and the target image
    optimizer = optim.Adam([z], lr=lr)

    # run the optimization sttep for multiple multiple operations
    for i in range(num_iterations):
        def closure():
            optimizer.zero_grad()
            
            with torch.no_grad():
                # constrain z to always be between -1 and 1
                z.clamp(-1, 1)
            
            # compute the combined contextual and prior loss based on the latent vector, the inpainting target and the mask
            loss = compute_loss(z, target_image, target_image_mask, lm)
            # perform backpropagation
            loss.backward() 
            
            # print the loss each iteration
            print(f'Iter {i}, Loss: {loss.item()}')
            
            return loss
        
        # perform the next optimization step based on the computed loss
        optimizer.step(closure)
    
    # return the denormalized (rgb pixel values between 0 and 1)
    return unorm(netG(z)[0])

## END CRITICAL CODE


def perform_inpainting(target_image, mask, lm, lr_inpainting, num_iterations_inpainting):
    """
    Perform image inpainting.
    """
    # compute the denormalized version of 
    target_image_denormalized = unorm(target_image)

    corrupted_img = target_image * mask
    corrupted_img_denormalozed = target_image_denormalized * mask

   
    predicted_image = optimize_latent_vector(corrupted_img, mask, lm, num_iterations_inpainting, lr_inpainting)

    return predicted_image, corrupted_img_denormalozed



