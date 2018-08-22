import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from tqdm import tqdm
from __future__ import print_function, division
import matplotlib.pyplot as plt
import sys
import numpy as np

import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization
from keras.layers.core import Reshape, Dense, Dropout, Flatten, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
from keras.optimizers import Adam
from keras.constraints import Constraint

from keras import backend as K 

K.set_image_dim_ordering('th')
adam = Adam(0.0002, 0.5)
dLosses = []
gLosses = []

epochs = 800
batch_size = 128
save_interval = 50

LOAD_SAVED_MODEL = False

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}




# Load MNIST data
def load_data(path='mnist.npz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = "mnist.npz"
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)


# Generator
def build_generator(latent_dim):
    # TODO. Make sure to use a tanh activation at the last layer and compile the model before returning it
    # For upsampling the image, you could use UpSampling2D function from Keras library. 
    # Use adam optimizer with: adam = Adam(0.0002, 0.5)   
    generator = Sequential()
    generator.add(Dense(256, input_dim=latent_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02), W_constraint = WeightClip(2)))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    generator.summary()
    return generator


# Discriminator
def build_discriminator(img_shape):
    # TODO. Make sure to use a sigmoid at the last layer and compile the model before returning it
    # Use adam optimizer with adam = Adam(0.0002, 0.5)    
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02), W_constraint = WeightClip(2)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)
    discriminator.summary()
    return discriminator


##################### TODO #####################
# Now that we've built the disriminator and generator, lets combine the two for the full end to end system.
# This is where we are calculating D(G(z))!
# We will set up a Model object to train the generator to fool the discriminator. We need to turn of
# weight updates for the discriminator, create an Input object for the generator with the right
# dimension, run that through the generator, and run the output of the generator through the discriminator. 
def build_gan(discriminator, generator, latent_dim):   
    ganInput = Input(shape=(latent_dim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adam)
    gan.summary()
    return gan

# Parameters for our mnist dataset. 
img_rows = 28
img_cols = 28
channels = 1

img_shape = (channels, img_rows, img_cols)
latent_dim = 100

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('myimages/dcgan_generated_image_epoch_%d.png' % epoch)


def saveModels(epoch):
    generator.save('mymodels/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('mymodels/gan_discriminator_epoch_%d.h5' % epoch)

def loadModels(epoch):
    generator.save('mymodels/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('mymodels/gan_discriminator_epoch_%d.h5' % epoch)


# Load MNIST data and rescale -1 to 1
(X_train, y_train), (X_test, y_test) = load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)

# Build and compile the discriminator
discriminator = build_discriminator(X_train.shape[1:])

# Build and compile the generator
generator = build_generator(latent_dim)

# Build and compile the combined network
gan = build_gan(discriminator, generator, latent_dim)

batchCount = X_train.shape[0] / batch_size
if LOAD_SAVED_MODEL==False:
    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(int(batchCount))):

            # Get a random sample from real images, and from random noise
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

            # Generate fake MNIST images, from noise
            generatedImages = generator.predict(noise)

            # Lets train the discriminator first. 
            # We will concatenate the real images and fake images into a variable X
            X = np.concatenate((image_batch, generatedImages))

            # Create the labels for fake and real data, composed of 0s and 1s
            yDis = np.zeros(2*batch_size)
            yDis[:batch_size] = 1

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Now lets train the generator
            # Generate batch_size sized random noise 
            noise = np.random.uniform(0,1,size=[batch_size,latent_dim])

            # Generate the labels for the generator
            yGen = np.ones(batch_size)

            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 20 == 0:
            plotGeneratedImages(e)
            saveModels(e)
        plotLoss(e)

if LOAD_SAVED_MODEL==True:
    epoch=880
    loadModels(epoch)
    plotGeneratedImages(epoch=epoch)

plt.show()

'''
There are two major components within GANs which are Generators and Discriminators. A discriminator network assigns a probability that the image is real while the generator network takes some noise vector and outputs an image. When training the generative network, it learns which areas of the image to improve/change so that the discriminator would have a harder time differentiating its generated images from the real ones. The generative network keeps producing images that are closer in appearance to the real images while the discriminative network is trying to determine the differences between real and fake images. The ultimate goal is to have a generative network that can produce images which are indistinguishable from the real ones. After training for 500 epochs, the images produced after the first 300 epochs it shows a blurry image and does not have any real structure. Now after 400+ epochs, the image start to take shape and finally after 600+ epochs, its a bit clearer though there are several images that are still unrecognizable.
'''