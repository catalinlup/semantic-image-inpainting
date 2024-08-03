import torchvision.datasets

def download_dataset():
    """
    Download the dataset used to train the GAN
    """
    torchvision.datasets.CelebA('celeb', download=True, split='train')

download_dataset()