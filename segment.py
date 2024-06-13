#Pretrained model and code to use pretrained model from:
#https://github.com/hiwad-aziz/kitti_deeplab/blob/master/inference.py

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import imageio
import numpy as np
import cv2
import random

IMAGE_HEIGHT = 512
IMAGE_HEIGHT_CROPPED = 300
IMAGE_WIDTH = 512
TF_RECORD_PATH = "output.tfrecord"
NUM_IMGS = 8
OUTPUT_PATH = "output_images"
NUM_CLASSES = 19
IMAGE_DIRECTORY = "segmented_images_colored"
OUTPUT_DIRECTORY = "collages/"


def load_inference_graph(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")

    return graph

def logits2image(logits):
    image_classes = 19
    logits = logits.astype(np.uint8)
    images = [np.zeros([IMAGE_HEIGHT_CROPPED,IMAGE_WIDTH,3],dtype=float) for _ in range(image_classes)]

    for i in range(IMAGE_HEIGHT_CROPPED):
        for j in range(IMAGE_WIDTH):
            img_class = logits[i,j]
            images[img_class][i,j,:] = [255, 255, 255]
    
    images = [img.astype(np.uint8) for img in images]
    return images

def slice_imgs(n_samples):
    print("slicing images")
    print("loading inference graph")
    graph = load_inference_graph('frozen_inference_graph.pb')
    image_input = graph.get_tensor_by_name('prefix/ImageTensor:0')
    print("image input shape: ", image_input.shape)
    softmax = graph.get_tensor_by_name('prefix/SemanticPredictions:0')
    print("softmax input shape: ", softmax.shape)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    # Create output directories
    if not os.path.exists('segmented_images_colored/'):
        os.mkdir('segmented_images_colored/') 
    
    print("Iterating through images")

    image_dir = "images/"
    with tf.compat.v1.Session(graph=graph) as sess:
        for i in range(n_samples):
            fname=random.choice(os.listdir(image_dir))
            original_img = imageio.imread(os.path.join(image_dir, fname))[IMAGE_HEIGHT - IMAGE_HEIGHT_CROPPED:, :] 
            img = np.expand_dims(original_img, axis=0)
            probs = sess.run(softmax, {image_input: img})
            img = tf.squeeze(probs).eval()
            img_colored = logits2image(img)
            for i, sorted_img in enumerate(img_colored):
                sorted_img = cv2.cvtColor(sorted_img, cv2.COLOR_BGR2RGB)
                masked_img = cv2.bitwise_and(original_img, sorted_img, mask=None)

                flattened_img = masked_img.flatten()
                if (np.count_nonzero(flattened_img) < IMAGE_HEIGHT_CROPPED * IMAGE_WIDTH * 3 / 12 or np.count_nonzero(flattened_img) > IMAGE_HEIGHT_CROPPED * IMAGE_WIDTH * 3 * .80):  # certain percent of image must be colored
                    continue
                
                if masked_img.shape[2] == 3:
                    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2RGBA)
                print("masked_img shape: ", masked_img.shape)
                mask = np.all(masked_img[:, :, :3] == [0, 0, 0], axis=-1)
                masked_img[mask, 3] = 0

                if not os.path.exists('segmented_images_colored/class_'+str(i)):
                    os.mkdir('segmented_images_colored/class_'+str(i))
                cv2.imwrite(os.path.join('segmented_images_colored/class_'+str(i), fname[:-4]+".png"), masked_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                print(fname)
        


# Generating collages
def parabola(x):
    return int((-0.0002*(x**2)) + (0.55*x) + 300)

def place_imgs(img_class=int, num_imgs=int, output_name=str):
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)
    base_image = np.zeros((1280, 1600, 3), dtype=np.uint8)
    base_image = np.concatenate((base_image, np.ones((1280, 1600, 1), dtype=np.uint8)), axis=2)
    base_dims = base_image.shape[:2]
    for _ in range(num_imgs):
        fname = random.choice(os.listdir(IMAGE_DIRECTORY+'/class_'+str(img_class)))
        img = imageio.imread(os.path.join(IMAGE_DIRECTORY+'/class_'+str(img_class)+"/"+ fname)) 
        img = np.pad(img, ((0, base_dims[0] - img.shape[2]), (0,  base_dims[1] - img.shape[1]), (0, 0)), mode='constant', constant_values=0)

        #random scatter
        x = random.uniform(-300, base_dims[1]-300)
        y = random.uniform(-300, base_dims[0]-300)

        # #follow parabola
        # x = random.uniform(-300, base_dims[1]-300)
        # y = parabola(x) + int(random.uniform(-1.0, 1.0)*(base_dims[0]/8))

        # #center
        # x_gutter = base_dims[1]/8
        # y_gutter = base_dims[0]/8
        # x = random.uniform(x_gutter, base_dims[1] - x_gutter)
        # y = random.uniform(y_gutter, base_dims[0]- y_gutter)

        M = np.float32([
            [1, 0, x],
            [0, 1, y]
        ])

        img = cv2.warpAffine(img, M, (base_dims[1], base_dims[0]))
        crop_type = np.random.choice([0, 1])
        if crop_type == 0:
            img = cv2.bitwise_not(img)
            base_image = cv2.bitwise_and(img, base_image)
        else:
            base_image = cv2.bitwise_or(img, base_image)
    cv2.imwrite(output_name, base_image)
        

if __name__ == "__main__":
    if sys.argv[1] == "slice":
        tf.compat.v1.enable_eager_execution()
        slice_imgs(int(sys.argv[2]))
    if sys.argv[1] == "collage":
        classes = [int(i) for i in sys.argv[4:]]
        repetitions = int(sys.argv[2])
        samples = int(sys.argv[3])
        for i in range(len(classes)): 
            for j in range(repetitions):
                place_imgs(classes[i], samples, OUTPUT_DIRECTORY+'collage_'+str((repetitions*i) + j)+'.png')
    else:
        print("To slice images: python segment.py slice number_of_images")
        print("To slice images: python segment.py collage number_of_repetitions num_samples class_numbers")
