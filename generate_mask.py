import torch
from torchvision.utils import save_image
import random


device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")


def generate_random_mask(mask_size, ratio):
    """
    Generates random noise to be used as a mask
    """

    mask = torch.ones(mask_size * mask_size).reshape((mask_size, mask_size))

    for i in range(mask_size):
        for j in range(mask_size):
            if random.random() < ratio:
                mask[i, j] = 0.0

    
    return mask


def generate_half_vertical_mask(mask_size, ratio):
    """
    Creates a mask that removes a vertical patch from the image
    """
    mask = torch.ones(mask_size * mask_size).reshape((mask_size, mask_size))

    mask[:, 0 : int(ratio * mask_size)] = 0

    return mask

def generate_half_horizontal_mask(mask_size, ratio):
    """
    Creates a mask that removes a horizontal patch from the image
    """
    mask = torch.ones(mask_size * mask_size).reshape((mask_size, mask_size))

    mask[0: int(ratio * mask_size), :] = 0

    return mask

def generate_rect_mask(mask_size, start_lin, start_col, w, h):
    """
    Creates a mask for a set size that represents a rect starting from (start_lin, start_col) and having a width 'w' and a height 'h'
    """

    mask = torch.ones(mask_size * mask_size).reshape((mask_size, mask_size))

    mask[start_lin : start_lin + h, start_col : start_col + w] = 0

    return mask


msk = generate_rect_mask(64, 10, 15, 40, 40)
save_image(msk, 'masks/mask_square.png')


half_mask_vertical = generate_rect_mask(64, 2, 2, 30, 60)
save_image(half_mask_vertical, 'masks/msk_vertical.png')

hald_mask_horizontal = generate_rect_mask(64, 2, 2, 60, 30)
save_image(hald_mask_horizontal, 'masks/msk_horizontal.png')