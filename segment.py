#https://keras.io/examples/vision/deeplabv3_plus/
#https://github.com/hiwad-aziz/kitti_deeplab/blob/master/inference.py

import tensorflow as tf
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchdata.datapipes.iter import FileLister, FileOpener
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import cv2

tf.compat.v1.enable_eager_execution()
# tf.disable_v2_behavior()

# IMAGE_HEIGHT = 512
# IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1382 #change these
TF_RECORD_PATH = "output.tfrecord"
NUM_IMGS = 8
OUTPUT_PATH = "output_images"

def decode_image(image):
    '''
    Map the image to the [-1, 1] range.
    '''
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [512, 1382]) #CHANGE THIS.
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    image = (tf.cast(image, tf.float32) / 127.5) - 1 #Normalize images to [-1, 1]
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
  print(image.shape)
  print("type: ", type(image))
  return image


def load_inference_graph(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')  
    return graph

def logits2image(logits):
    logits = logits.astype(np.uint8)
    images = [np.zeros([IMAGE_HEIGHT,IMAGE_WIDTH,3],dtype=float)]
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            img_class = logits[i,j]
            images[img_class][i,j,:] = [255, 255, 255]
    
    images = images.astype(np.uint8)
    return images

def slice_imgs():
    print("slicing images")
    dataset = tf.data.TFRecordDataset(TF_RECORD_PATH)
    dataset = dataset.map(read_tfrecord) #Unpacks from string to a float32 with correct dims under shape
    dataset_size = sum(1 for _ in dataset)
    # dataset = dataset.shuffle(dataset_size).batch(NUM_IMGS)

    
    print("loading inference graph")
    graph = load_inference_graph('/Users/esmepuzio/MyHouseGAN/frozen_inference_graph.pb')
    image_input = graph.get_tensor_by_name('prefix/ImageTensor:0')
    print("image input shape: ", image_input.shape)
    softmax = graph.get_tensor_by_name('prefix/SemanticPredictions:0')
    print("softmax input shape: ", softmax.shape)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    print("Iterating through images")

    ctr = 0
    for i, tf_image in enumerate(dataset):
        if ctr > 5:
            break
        with tf.compat.v1.Session(graph=graph) as sess:
            img = tf_image.numpy()
            plt.imshow(img)
            plt.show()
            print("shape: ", img.shape)
            print("type: ", type(img))
            print("ok we start fr")
            img = np.expand_dims(img, axis=0)
            print("shape: ", img.shape)
            probs = sess.run(softmax, {image_input: img})
            img = tf.squeeze(probs).eval()
            image_masks = logits2image(img)
            for mask in image_masks:
                masked_image = (cv2.bitwise_and(img, mask=mask))
                cv2.imwrite(os.path.join(f"{OUTPUT_PATH}/img_{i}"),masked_image)
        ctr += 1
        
if __name__ == "__main__":
    slice_imgs()