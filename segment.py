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

IMAGE_HEIGHT = 512
IMAGE_HEIGHT_CROPPED = 300
IMAGE_WIDTH = 512
TF_RECORD_PATH = "output.tfrecord"
NUM_IMGS = 8
OUTPUT_PATH = "output_images"
NUM_CLASSES = 19

def decode_image(image):
    '''
    Decode the image from the tfrecord file.
    '''
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    image = tf.cast(image, tf.uint8)
    image = tf.io.encode_png(image)
    return image

def read_tfrecord(example):
    '''
    Decode the image from the tfrecord file using decode_image
    '''
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image

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
    # dataset = tf.data.TFRecordDataset(TF_RECORD_PATH)
    # dataset = dataset.map(read_tfrecord)

    print("loading inference graph")
    graph = load_inference_graph('frozen_inference_graph.pb')
    image_input = graph.get_tensor_by_name('prefix/ImageTensor:0')
    print("image input shape: ", image_input.shape)
    softmax = graph.get_tensor_by_name('prefix/SemanticPredictions:0')
    print("softmax input shape: ", softmax.shape)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

        
    # Create output directories in the image folder
    if not os.path.exists('segmented_images_colored/'):
        os.mkdir('segmented_images_colored/') 
        # for n in range(NUM_CLASSES):
        #     os.mkdir('segmented_images_colored/class_' + str(n))
    
    print("Iterating through images")

    ctr = 0
    image_dir = "images/"
    # image_dir = "test_img/"
    with tf.compat.v1.Session(graph=graph) as sess:
        for fname in sorted(os.listdir(image_dir)):
            if ctr > 5:
                break
            original_img = imageio.imread(os.path.join(image_dir, fname))[:IMAGE_HEIGHT_CROPPED, :] 
            img = np.expand_dims(original_img, axis=0)
            print("img type: ", type(img))
            print("img shape: ", img.shape)
            print("JPEG Image Data Type: ", img.dtype)
            print("JPEG Image Shape: ", img.shape)
            print("JPEG Image Min: ", np.min(img))
            print("JPEG Image Max: ", np.max(img))
           
            probs = sess.run(softmax, {image_input: img})
            img = tf.squeeze(probs).eval()
            img_colored = logits2image(img)
            for i, sorted_img in enumerate(img_colored):
                sorted_img = cv2.cvtColor(sorted_img, cv2.COLOR_BGR2RGB)
                print("sorted_img shape: ", sorted_img.shape)
                print("img shape: ", original_img.shape)
                masked_img = cv2.bitwise_and(original_img, sorted_img, mask=None)
                if (np.all(masked_img > 20)):
                    if not os.path.exists('segmented_images_colored/class_'+str(i)):
                        os.mkdir('segmented_images_colored/class_'+str(i))
                    cv2.imwrite(os.path.join('segmented_images_colored/class_'+str(i)+"/"+fname),cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))   
                    print(fname)
            ctr += 1
        
if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    slice_imgs()
