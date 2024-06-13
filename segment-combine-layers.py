import os
import numpy as np
import imageio
import cv2

COLLAGES_DIRECTORY = "collages"
OUTPUT_DIRECTORY = "final_collage"
OUTPUT_NAME = "final_collage.png"

if not os.path.exists(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)

# Initialize a base image with an alpha channel
base_image = np.zeros((1280, 1600, 4), dtype=np.uint8)  # 4 channels for RGBA
base_dims = base_image.shape[:2]

# Loop through all PNG files in the collages directory
for fname in os.listdir(COLLAGES_DIRECTORY):
    if fname.endswith('.png'):
        img = imageio.imread(os.path.join(COLLAGES_DIRECTORY, fname))

    # normalize alpha channels from 0-255 to 0-1
    alpha_background = base_image[:,:,3] / 255.0
    alpha_foreground = img[:,:,3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        base_image[:,:,color] = alpha_foreground * img[:,:,color] + \
            alpha_background * base_image[:,:,color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    base_image[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

# display the image
cv2.imshow("Composited image", base_image)

# Save the final composed image
cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, OUTPUT_NAME), base_image)

