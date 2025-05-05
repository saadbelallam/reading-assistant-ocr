################
#    Step 1    #
################

import os
from src.utils.statistical_clustering import hybrid_blocks_detection_2
from src.pipeline.lines_detection import apply_craft_detection
from models.Document_Image_Dewarping.predict import dewarp_image

from deskew import determine_skew

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate


def text_zones(input_image_path, output_dir) :
    print(f"Traitement de l'image : {input_image_path}")
    quads = apply_craft_detection(input_image_path)["boxes"]
    clusters = hybrid_blocks_detection_2(quads, output_dir, prox_thr=1.0, overlap_thr=0.75,
                                visualize=True, img_path=input_image_path)

def deskew(input_image_path, output_dir) :
    basename = input_image_path.split("/")[-1].replace("1_text_zones.png", "")
    image = io.imread(input_image_path)
    # rgba to rgb
    if image.shape[2] == 4 :
        image = image[:, :, :3]
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, resize=True) * 255
    # make sure the output directory exists
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    # save the rotated image
    io.imsave(f"{output_dir}/2_deskewed.png", rotated.astype(np.uint8))



def geometric_correction(input_image_path, output_dir) :
    # Step 1: Detect text zones
    text_zones(input_image_path, output_dir)

    # Step 2: Deskew the image
    filename = os.path.basename(input_image_path)
    basename, _ = os.path.splitext(filename)
    input_image_path = os.path.join(output_dir, "1_text_zones.png")
    deskew(input_image_path, output_dir)
    
    # # Step 3: Geometric correction
    input_image_path = os.path.join(output_dir, "2_deskewed.png")
    output_dir = output_dir + "/3_dewarped.png"
    dewarp_image(input_image_path, output_dir)

    return output_dir

# if __name__ == '__main__' :
#     file = "/home/yassine/ab.png"
#     geometric_correction(file, "outputs")