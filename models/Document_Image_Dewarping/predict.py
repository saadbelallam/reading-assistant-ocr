# """
# Code references:
# # >>> https://github.com/cvlab-stonybrook/DewarpNet
# # >>> https://github.com/fh2019ustc/DocGeoNet
# """
# import argparse
# import time

# import cv2
# import glob
# import numpy as np
# import os

# import torch
# import torch.nn.functional as F

# from PIL import Image
# from model import DewarpTextlineMaskGuide


# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_size', type=int, default=224, help='image size')
#     parser.add_argument('--model_path', type=str, default='model/30.pt', help='model path')
#     parser.add_argument('--img_path', type=str, default='dataset/Dewarp/DocUNet_dataset/crop',
#                         help='image path or path to folder containing images')

#     parser.add_argument('--save_path', type=str, default='infer/', help='save path')

#     return parser.parse_args()


# def predict(img_path, save_path, filename, recti_model):
#     assert os.path.exists(img_path), 'Incorrect Image Path'
#     assert os.path.exists(save_path), 'Incorrect Save Path'

#     img_size = parser.input_size

#     img = np.array(Image.open(img_path))[:, :, :3] / 255.
#     img_h, img_w, _ = img.shape
#     input_img = cv2.resize(img, (img_size, img_size))

#     with torch.no_grad():
#         recti_model.eval()
#         input_ = torch.from_numpy(input_img).permute(2, 0, 1).cuda()
#         input_ = input_.unsqueeze(0)
#         start = time.time()

#         bm = recti_model(input_.float())
#         bm = (2 * (bm / 223.) - 1) * 0.99
#         ps_time = time.time() - start

#     bm = bm.detach().cpu()
#     bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))  # x flow
#     bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))  # y flow
#     bm0 = cv2.blur(bm0, (3, 3))
#     bm1 = cv2.blur(bm1, (3, 3))
#     lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).float()  # h * w * 2

#     out = F.grid_sample(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
#     img_geo = ((out[0] * 255.).permute(1, 2, 0).numpy()).astype(np.uint8)

#     cv2.imwrite(filename, img_geo[:, :, ::-1])  # save

#     return ps_time


# if __name__ == '__main__':
#     parser = get_args()

#     recti_model = DewarpTextlineMaskGuide(image_size=parser.input_size)
#     recti_model = torch.nn.DataParallel(recti_model)
#     state_dict = torch.load(parser.model_path, map_location='cpu')

#     recti_model.load_state_dict(state_dict)
#     recti_model.cuda()
#     print(f'model loaded')

#     img_path = parser.img_path
#     save_path = parser.save_path
#     total_time = 0.0

#     start = time.time()
#     img_num = 0.0
#     for file in glob.glob(img_path + "/*"):  # img_names:  #
#         print("file: ", file)
#         filename = (save_path + "/" + file[file.rindex("/") + 1:file.rindex(".")] + ".png")

#         total_time += predict(file, save_path, filename, recti_model)
#         print("Written ", file[file.rindex("/") + 1:file.rindex(".")])
#         img_num += 1
#     print('FPS: %.1f' % (1.0 / (total_time / img_num)))


"""
Code references:
# >>> https://github.com/cvlab-stonybrook/DewarpNet
# >>> https://github.com/fh2019ustc/DocGeoNet
"""
import time
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
from models.Document_Image_Dewarping.model import DewarpTextlineMaskGuide


def predict_single_image(img_path, output_path=None, 
                         input_size=224, 
                         model_path='models/Document_Image_Dewarping/model/30.pt'):
    """
    Process a single image through the document dewarping model
    
    Args:
        img_path (str): Path to the input image
        output_path (str, optional): Path to save the output image. If None, constructs from input path
        input_size (int): Size of images for the model input
        model_path (str): Path to the model weights file
        
    Returns:
        tuple: (dewarped_image, processing_time)
    """
    # Check if input file exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'Input image not found: {img_path}')
    
    # Determine output path if not provided
    if output_path is None:
        base_dir = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(base_dir, f"{name}_dewarped.png")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    recti_model = DewarpTextlineMaskGuide(image_size=input_size)
    recti_model = torch.nn.DataParallel(recti_model)
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        recti_model.load_state_dict(state_dict)
        recti_model.cuda()
        print(f'Model loaded from {model_path}')
    except Exception as e:
        raise RuntimeError(f'Failed to load model: {e}')
    
    # Process the image
    start_time = time.time()
    
    # Load and preprocess image
    img = np.array(Image.open(img_path))[:, :, :3] / 255.
    img_h, img_w, _ = img.shape
    input_img = cv2.resize(img, (input_size, input_size))
    
    # Run inference
    with torch.no_grad():
        recti_model.eval()
        input_ = torch.from_numpy(input_img).permute(2, 0, 1).cuda()
        input_ = input_.unsqueeze(0)
        
        inference_start = time.time()
        bm = recti_model(input_.float())
        bm = (2 * (bm / 223.) - 1) * 0.99
        inference_time = time.time() - inference_start
    
    # Process output
    bm = bm.detach().cpu()
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))  # x flow
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))  # y flow
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).float()  # h * w * 2
    
    out = F.grid_sample(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
    img_geo = ((out[0] * 255.).permute(1, 2, 0).numpy()).astype(np.uint8)
    
    # Save result
    cv2.imwrite(output_path, img_geo[:, :, ::-1])  # save
    
    total_time = time.time() - start_time
    print(f"Image processed in {inference_time:.3f}s (total: {total_time:.3f}s)")
    print(f"Dewarped image saved to: {output_path}")
    
    return img_geo, inference_time


def dewarp_image(img_path, output_path=None, input_size=224, model_path='models/Document_Image_Dewarping/model/30.pt'):
    """
    Simple wrapper function for dewarping a single image.
    Returns just the dewarped image.
    
    Args:
        img_path (str): Path to the input image
        output_path (str, optional): Path to save the output image. If None, constructs from input path
        input_size (int): Size of images for the model input
        model_path (str): Path to the model weights file
        
    Returns:
        numpy.ndarray: The dewarped image
    """
    try:
        dewarped_img, _ = predict_single_image(
            img_path=img_path,
            output_path=output_path,
            input_size=input_size,
            model_path=model_path
        )
        return dewarped_img
    except Exception as e:
        print(f"Error dewarping image {img_path}: {e}")
        return None


# if __name__ == '__main__':
#     # Example usage when run directly
#     import sys
    
#     if len(sys.argv) < 2:
#         print("Usage: python predict.py <input_image_path> [output_image_path]")
#         sys.exit(1)
    
#     input_path = sys.argv[1]
#     output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
#     try:
#         dewarp_image(input_path, output_path)
#     except Exception as e:
#         print(f"Error: {e}")
#         sys.exit(1)