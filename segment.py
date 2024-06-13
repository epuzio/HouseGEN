#https://keras.io/examples/vision/deeplabv3_plus/
#https://github.com/hiwad-aziz/kitti_deeplab/blob/master/inference.py
#https://github.com/rishizek/tensorflow-deeplab-v3-plus?tab=readme-ov-file
#https://github.com/hellochick/ICNet-tensorflow
#https://www.kaggle.com/code/kerneler/image-segmentation-cityscapes-dataset

import tensorflow as tf
from zipfile import ZipFile
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchdata.datapipes.iter import FileLister, FileOpener
import matplotlib.pyplot as plt
import os
import sys
import tempfile
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


def load_inference_graph(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")

    return graph

def logits2image(logits):
    print('logits called')
    image_classes = 19
    logits = logits.astype(np.uint8)
    images = [np.zeros([IMAGE_HEIGHT_CROPPED,IMAGE_WIDTH,3],dtype=float) for _ in range(image_classes)]

    for i in range(IMAGE_HEIGHT_CROPPED):
        for j in range(IMAGE_WIDTH):
            img_class = logits[i,j]
            images[img_class][i,j,:] = [255, 255, 255]
    
    images = [img.astype(np.uint8) for img in images]
    return images

def slice_imgs():
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
    # image_dir = "test_img/"
    with tf.compat.v1.Session(graph=graph) as sess:
        for i in range(5):
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
        
if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    slice_imgs()
