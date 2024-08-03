from image_inpainting import perform_inpainting, compute_weights_convolution
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import load_inpainting_parameters
import torch
from poisson_blend.poisson_blend import poisson_blend

from PIL import Image


params = load_inpainting_parameters()

img_transforms = transforms.Compose([
                            transforms.Resize(params['image_size']),
                            transforms.CenterCrop(params['image_size']),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda:0" if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")

INPUT_FOLDER = 'image_inpainting_input'
MASK_INPUT_FOLDER = 'masks'
OUTPUT_FOLDER = 'image_inpainting_output'

MASKED_IMAGES_TO_BE_PROCESSED = [
    # ('img1.png', 'mask_square.png'),
    # ('img1.png', 'msk_horizontal.png'),
    # ('img1.png', 'msk_vertical.png'),
    # ('img1.png', 'msk_noise_25.png'),
    # ('img1.png', 'msk_noise_80.png'),

    ('000001.jpg', 'mask_square.png'),
    ('000001.jpg', 'msk_horizontal.png'),
    ('000001.jpg', 'msk_vertical.png'),

    ('000002.jpg', 'mask_square.png'),
    ('000002.jpg', 'msk_horizontal.png'),
    ('000002.jpg', 'msk_vertical.png'),

    ('000003.jpg', 'mask_square.png'),
    ('000003.jpg', 'msk_horizontal.png'),
    ('000003.jpg', 'msk_vertical.png'),

    ('000004.jpg', 'mask_square.png'),
    ('000004.jpg', 'msk_horizontal.png'),
    ('000004.jpg', 'msk_vertical.png'),

    ('000005.jpg', 'mask_square.png'),
    ('000005.jpg', 'msk_horizontal.png'),
    ('000005.jpg', 'msk_vertical.png'),

    ('000006.jpg', 'mask_square.png'),
    ('000006.jpg', 'msk_horizontal.png'),
    ('000006.jpg', 'msk_vertical.png'),

    ('000007.jpg', 'mask_square.png'),
    ('000007.jpg', 'msk_horizontal.png'),
    ('000007.jpg', 'msk_vertical.png'),

    ('000008.jpg', 'mask_square.png'),
    ('000008.jpg', 'msk_horizontal.png'),
    ('000008.jpg', 'msk_vertical.png'),


    ('000009.jpg', 'mask_square.png'),
    ('000009.jpg', 'msk_horizontal.png'),
    ('000009.jpg', 'msk_vertical.png'),

    # ('057540.jpg', 'mask_square.png'),
    # ('057540.jpg', 'msk_horizontal.png'),
    # ('057540.jpg', 'msk_vertical.png'),

    # ('135053.jpg', 'mask_square.png'),
    # ('135053.jpg', 'msk_horizontal.png'),
    # ('135053.jpg', 'msk_vertical.png'),

    # ('135062.jpg', 'mask_square.png'),
    # ('135062.jpg', 'msk_horizontal.png'),
    # ('135062.jpg', 'msk_vertical.png'),

    # ('img2.png', 'mask_square.png'),
    # ('img2.png', 'msk_horizontal.png'),
    # ('img2.png', 'msk_vertical.png'),
    # ('img2.png', 'msk_noise_25.png'),
    # ('img2.png', 'msk_noise_80.png'),

    # ('img3.png', 'mask_square.png'),
    # ('img3.png', 'msk_horizontal.png'),
    # ('img3.png', 'msk_vertical.png'),
    # ('img3.png', 'msk_noise_25.png'),
    # ('img3.png', 'msk_noise_80.png'),

    # ('img4.png', 'mask_square.png'),
    # ('img4.png', 'msk_horizontal.png'),
    # ('img4.png', 'msk_vertical.png'),
    # ('img4.png', 'msk_noise_25.png'),
    # ('img4.png', 'msk_noise_80.png'),

    # ('img5.png', 'mask_square.png'),
    # ('img5.png', 'msk_horizontal.png'),
    # ('img5.png', 'msk_vertical.png'),
    # ('img5.png', 'msk_noise_25.png'),
    # ('img5.png', 'msk_noise_80.png'),

    # ('img6.png', 'mask_square.png'),
    # ('img6.png', 'msk_horizontal.png'),
    # ('img6.png', 'msk_vertical.png'),
    # ('img6.png', 'msk_noise_25.png'),
    # ('img6.png', 'msk_noise_80.png'),

    # ('img7.png', 'mask_square.png'),
    # ('img7.png', 'msk_horizontal.png'),
    # ('img7.png', 'msk_vertical.png'),
    # ('img7.png', 'msk_noise_25.png'),
    # ('img7.png', 'msk_noise_80.png'),

    # ('img8.png', 'mask_square.png'),
    # ('img8.png', 'msk_horizontal.png'),
    # ('img8.png', 'msk_vertical.png'),
    # ('img8.png', 'msk_noise_25.png'),
    # ('img8.png', 'msk_noise_80.png'),

]


for img_path, mask_path in MASKED_IMAGES_TO_BE_PROCESSED:

    base_name = img_path.split('.')[0] + '_' + mask_path.split('.')[0]

    img = Image.open(f'{INPUT_FOLDER}/{img_path}').convert('RGB')
    img_transformed = img_transforms(img).to(device)
    

    mask = Image.open(f'{MASK_INPUT_FOLDER}/{mask_path}')
    mask_transformed = transforms.ToTensor()(mask).to(device)[0]

    predicted_image, masked_image = perform_inpainting(img_transformed, mask_transformed, params['lm'], params['lr_inpainting'], params['num_iterations_inpainting'])

    save_image(predicted_image, f'{OUTPUT_FOLDER}/{base_name}_predicted.png')
    save_image(masked_image, f'{OUTPUT_FOLDER}/{base_name}_masked.png')
    save_image(img_transformed * 0.5 + 0.5, f'{OUTPUT_FOLDER}/{base_name}_expected.png')
    poisson_blend(f'{OUTPUT_FOLDER}/{base_name}_masked.png', f'{OUTPUT_FOLDER}/{base_name}_predicted.png', f'{MASK_INPUT_FOLDER}/{mask_path}', f'{OUTPUT_FOLDER}/{base_name}_blended.png')

