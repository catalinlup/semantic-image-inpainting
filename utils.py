import torch.nn as nn
import torchvision.datasets
import os
from dotenv import load_dotenv
from dotenv import find_dotenv

def weights_init(m):
    """
    Initializes the weights of the neural network using a normal distribution,
    as specified in the paper.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def download_dataset():
    """
    Download the dataset used to train the GAN
    """
    torchvision.datasets.CelebA('celeb', download=True, split='train')


def load_training_parameters():
    """
    Loads the parameters used for training the GAN system.
    """
    load_dotenv(find_dotenv('training.env'))

    return {
        'dataroot': os.environ['dataroot'],
        'workers': int(os.environ['workers']),
        'batch_size': int(os.environ['batch_size']),
        'image_size': int(os.environ['image_size']),
        'nc': int(os.environ['nc']),
        'nz': int(os.environ['nz']),
        'ngf': int(os.environ['ngf']),
        'ndf': int(os.environ['ndf']),
        'num_epochs': int(os.environ['num_epochs']),
        'lr': float(os.environ['lr']),
        'beta1': float(os.environ['beta1']),
        'ngpu': int(os.environ['ngpu']),
        'output_model_name': os.environ['output_model_name'],
        'last_model_name': os.environ['last_model_name'],
        'resume_from_last': True if os.environ['resume_from_last'] == 'true' else False
    }


def load_inpainting_parameters():
    """
    Load the paramters used for image in-painting
    """
    load_dotenv(find_dotenv('inpainting.env'))

    return {
        'gan_model_name': os.environ['gan_model_name'],
        'image_size': int(os.environ['image_size']),
        'nc': int(os.environ['nc']),
        'nz': int(os.environ['nz']),
        'ngf': int(os.environ['ngf']),
        'ndf': int(os.environ['ndf']),
        'ngpu': int(os.environ['ngpu']),
        'lr_inpainting': float(os.environ['lr_inpainting']),
        'lm': float(os.environ['lm']),
        'num_iterations_inpainting': int(os.environ['num_iterations_inpainting'])
    }