import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from predict_walls.UNet_Pytorch_Customdataset.utils.data_loading import BasicDataset
from predict_walls.UNet_Pytorch_Customdataset.unet import UNet
from predict_walls.UNet_Pytorch_Customdataset.utils.utils import plot_img_and_mask

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
#                         help='Specify the file in which the model is stored')
#     parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
#     parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
#     parser.add_argument('--viz', '-v', action='store_true',
#                         help='Visualize the images as they are processed')
#     parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
#     parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
#                         help='Minimum probability value to consider a mask pixel white')
#     parser.add_argument('--scale', '-s', type=float, default=0.5,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
#     return parser.parse_args()

# def get_output_filenames(input, output):
#     def _generate_name(fn):
#         return f'{os.path.splitext(fn)[0]}_OUT.png'
#     return output or list(map(_generate_name, input))

def mask_to_image(mask: np.ndarray, mask_values):
    # Define a color map for each class
    color_map = {
        0: [0, 0, 0],        # Background: black
        1: [255, 0, 0],      # Class 1: red
        2: [0, 255, 0],      # Class 2: green
        3: [0, 0, 255],      # Class 3: blue
        4: [255, 255, 0],    # Class 4: yellow
        5: [0, 255, 255],    # Class 5: cyan
        6: [255, 0, 255],    # Class 6: magenta
        7: [192, 192, 192],  # Class 7: silver
        8: [128, 128, 128],  # Class 8: gray
        9: [128, 0, 0],      # Class 9: maroon
        10: [128, 128, 0],   # Class 10: olive
        11: [0, 128, 0],     # Class 11: dark green
        12: [128, 0, 128],   # Class 12: purple
        13: [0, 128, 128],   # Class 13: teal
        14: [0, 0, 128],     # Class 14: navy
        15: [255, 165, 0],   # Class 15: orange
        16: [255, 105, 180], # Class 16: hot pink
        17: [255, 20, 147],  # Class 17: deep pink
        18: [255, 99, 71],   # Class 18: tomato
        19: [255, 69, 0],    # Class 19: orange red
        20: [255, 215, 0],   # Class 20: gold
    }
    # Initialize an empty image
    out = np.zeros((mask.shape[-2], mask.shape[-1], 3), dtype=np.uint8)
    # Assign colors to each class
    for i, v in enumerate(mask_values):
        if v in color_map:
            out[mask == i] = color_map[v]
    return Image.fromarray(out)

def overlay_masks_on_image(image: Image, mask: np.ndarray, mask_values):
    # Convert the image to a numpy array
    img_array = np.array(image)
    # Convert the mask to an image with colors
    mask_image = mask_to_image(mask, mask_values)
    mask_array = np.array(mask_image)
    # Overlay the mask on the original image
    overlay = np.where(mask_array == 0, img_array, mask_array)
    return Image.fromarray(overlay)

# if __name__ == '__main__':
#     args = get_args()
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#     in_files = args.input
#     out_files = get_output_filenames(args)

#     net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Loading model {args.model}')
#     logging.info(f'Using device {device}')

#     net.to(device=device)
#     state_dict = torch.load(args.model, map_location=device)
#     mask_values = state_dict.pop('mask_values', [0, 1])
#     net.load_state_dict(state_dict)

#     logging.info('Model loaded!')

#     for i, filename in enumerate(in_files):
#         logging.info(f'Predicting image {filename} ...')
#         img = Image.open(filename)
#         if img.mode == 'L':  # 'L' mode indicates a single-channel grayscale image
#             img = img.convert('RGB')

#         mask = predict_img(net=net,
#                            full_img=img,
#                            scale_factor=args.scale,
#                            out_threshold=args.mask_threshold,
#                            device=device)
#         print(mask)
#         if not args.no_save:
#             out_filename = out_files[i]
#             result = mask_to_image(mask, mask_values)
#             result.save(out_filename)
#             logging.info(f'Mask saved to {out_filename}')


def predict_wall_mask(input, output, classes, model, bilinear=False, scale=0.5, mask_threshold=0.5, no_save=False):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    input = input

    net = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')


    logging.info(f'Predicting image {input} ...')
    img = Image.open(input)
    if img.mode == 'L':  # 'L' mode indicates a single-channel grayscale image
        img = img.convert('RGB')

    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=scale,
                        out_threshold=mask_threshold,
                        device=device)
    if not no_save:
        result = mask_to_image(mask, mask_values)
        result.save(output)
        logging.info(f'Mask saved to {output}')
    return mask