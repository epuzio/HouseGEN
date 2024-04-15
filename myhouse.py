#imports
import tensorflow as tf
import sys
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import losses
import numpy as np
from matplotlib import pyplot as plt
import os
import time
from IPython import display
import subprocess


## Arguments for GAN
EPOCHS = 300
EXAMPLES_TO_GENERATE = 8
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 8 
TF_RECORD_PATH = "output.tfrecord"
strides = [(1, 1), (2, 2), (2, 2)] 
noise_dim = 100
seed = tf.random.normal([EXAMPLES_TO_GENERATE, noise_dim])

alpha_Discriminator = 0.2
alpha_Generator = 0.2
momentum_BatchNormalization = 0.8

################# Building Generator #################

def add_generator_layer(model, num_filters, kernel, num_strides): #from rectgan
  '''
  Adding Conv2DTranspose -> BatchNormalization -> LeakyReLU layers to the model
  '''
  model.add(layers.Conv2DTranspose(num_filters, kernel_size=kernel, strides=num_strides, padding='same', use_bias=False))
  model.add(layers.BatchNormalization(momentum=momentum_BatchNormalization)) #instance normalization
  model.add(layers.ReLU())

def make_generator():
  '''
  Creating noise for the generator.
  '''
  model = tf.keras.Sequential() #add layers to generator model
  model.add(layers.Dense(16*16*256, input_shape=(100,), use_bias=False))
  model.add(layers.LeakyReLU(alpha=0.2))
  model.add(layers.Reshape((16, 16, 256)))

  add_generator_layer(model, 256, (5, 5), num_strides=(2, 2))
  add_generator_layer(model, 128, (5, 5), num_strides=(2, 2))
  add_generator_layer(model, 64, (5, 5), num_strides=(2, 2))
  add_generator_layer(model, 32, (5, 5), num_strides=(2, 2))
  model.add(layers.Conv2D(3,kernel_size=3,strides=1,padding='same',use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
  model.add(layers.Activation('tanh'))
  assert model.output_shape == (None, 256, 256, 3), "Generator output dimensions should be (256, 256, 3), aborting."
  return model

def add_discriminator_layer(model, filters, kernel, num_strides):
  model.add(layers.GaussianNoise(0.1)) #tip from reddit: https://github.com/ShivamShrirao/GANs_TF_2.0/blob/main/celeb_face_GAN.ipynb
  model.add(layers.Conv2D(filters, kernel_size=kernel, strides=num_strides, padding='same', use_bias=False))
  model.add(layers.BatchNormalization(momentum=momentum_BatchNormalization))
  model.add(layers.LeakyReLU(alpha_Discriminator))
    


def make_discriminator():
  """
  Building the discriminator.
  """
  model = tf.keras.Sequential()
  add_discriminator_layer(model, 8, (3, 3), num_strides=(2,2))
  add_discriminator_layer(model, 16, (3, 3), num_strides=(2,2))
  add_discriminator_layer(model, 32, (3, 3), num_strides=(2,2))
  add_discriminator_layer(model, 64, (3, 3), num_strides=(2,2))
  add_discriminator_layer(model, 128, (3, 3), num_strides=(2,2))
  add_discriminator_layer(model, 256, (3, 3), num_strides=(2,2))
 
  model.add(layers.Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(128))
  model.add(layers.Dense(1))
  return model

def discriminator_loss(real_output, fake_output):
  real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output) #real_output = 1
  fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output) #fake_output = 0
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output) #array of 1s

######## Running the model ########


@tf.function #turns into a graph, for faster execution
def train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    #compute gradients for discriminator and generator using two different GradientTapes
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: 
      generated_images = generator(noise, training=True)

      real_output = discriminator(image_batch, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)
  for i in range(predictions.shape[0]): #for each image in the batch
    plt.imshow(predictions[i, :, :, :]*0.5 + 0.5)
    plt.axis('off')
    plt.savefig(f"GAN_output_images/img{i}/image{i}_epoch_{epoch:04d}.png")
    plt.clf()  #clear the current figure to prevent overlapping plots
  
def make_output_directories(examples_to_generate):
  '''
  Make one image output directory for each example to generate.
  This saves all work-in-progress images to each directory.
  '''
  if not os.path.exists('GAN_output_images'):
    os.makedirs('GAN_output_images')
  for i in range(examples_to_generate):
    new_dir = 'GAN_output_images/img' + str(i)
    if not os.path.exists(new_dir):
      os.makedirs(new_dir)

def decode_image(image):
  '''
  Map the image to the [-1, 1] range.
  '''
  image = tf.image.decode_jpeg(image, channels=3)
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
  return image
  
def train(): 
  #Set up dataset
  print("BEGINNING SETUP...")
  print("Creating dataset from TFRecord...")
  dataset = tf.data.TFRecordDataset(TF_RECORD_PATH)
  dataset = dataset.map(read_tfrecord) #Unpacks from string to a float32 with correct dims under shape
  
  dataset_size = sum(1 for _ in dataset)

  print("Shuffling dataset, creating batches...")
  train_dataset = dataset.shuffle(dataset_size).batch(BATCH_SIZE)
  
  #Create generator and discriminator models:
  print("Creating generator model...")
  generator = make_generator()
  
  print("Creating discriminator model...")
  discriminator = make_discriminator()

  #Optimize losses and performance:
  print("Creating optimizers...")
  generator_optimizer = tf.keras.optimizers.Adam(1e-4)
  discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

  print("Creating directory for model checkpoints...")
  checkpoint_dir = 'GAN_checkpoints'
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  
  checkpoint_prefix = os.path.join(checkpoint_dir, "checkpt")
  
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
                                  discriminator_optimizer=discriminator_optimizer, 
                                  generator=generator, discriminator=discriminator)
  
  print("Creating directories for output images...")
  make_output_directories(EXAMPLES_TO_GENERATE)

  # print("Restoring from checkpoint if available...")
  # checkpoint_available = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  # if checkpoint_available.assert_existing_objects_matched():
  #   print("Checkpoint restored successfully.\n")
  # else:
  #     print("Checkpoint restore not available.\n")
  # print("SETUP COMPLETE.\n\n")

  print("Beginning training loop...")
  for epoch in range(EPOCHS):
    start = time.time()
    print("Epoch:", epoch)
    for i, image_batch in enumerate(train_dataset):
      print("Training step:", i)
      train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

    print("Saving images at epoch:", epoch)
    if (epoch + 1) % 10 == 0: #save every 10 epochs
      display.clear_output(wait=True)
      generate_and_save_images(generator,
                              epoch + 1,
                              seed)

    # Save the model every 25 epochs
    print("Saving checkpoint of model...")
    if (epoch + 1) % 25 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      print("Checkpoint saved successfully to file:", checkpoint_dir)

    print ('Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           EPOCHS,
                           seed)
  
def main():  
  train()

if __name__ == "__main__":
  main()