'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 16 #256
epochs = 50
epsilon_std = 1.0
fix_var = False


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
if fix_var:
    kl_loss = - 0.5 * K.sum(1 + 0.0 - K.square(z_mean) - K.exp(0.0), axis=-1)
else:
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

# Take one image per digit and print its corresponding mapping coordinates in the latent space
row_format ="{:>20}" * (3)
print(row_format.format("", *["x", "y"]))
for i in xrange(10):
    print(row_format.format(i, *x_test_encoded[np.random.choice(np.where(y_test == i)[0])]))


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)


# Add a sequence of images which form an interpolation from one digit to another
n = 10
digit_size = 28

orig_encoded_1 = x_test_encoded[np.random.choice(np.where(y_test == 1)[0])]
orig_encoded_8 = x_test_encoded[np.random.choice(np.where(y_test == 8)[0])]


# TODO: change the sampling function (because this function samples evenly spaced numbers)
grid_x = np.linspace(min(orig_encoded_1[0], orig_encoded_8[0]),
                     max(orig_encoded_1[0], orig_encoded_8[0]), n)
grid_y = np.linspace(min(orig_encoded_1[1], orig_encoded_8[1]),
                     max(orig_encoded_1[1], orig_encoded_8[1]), n)

for i, (x, y) in enumerate(zip(grid_x, grid_y)):
    z_sample = np.array([[x, y]])
    x_decoded = generator.predict(z_sample)
    digit = x_decoded[0].reshape(digit_size, digit_size)
    plt.imsave("/Users/gal/Dropbox/gal/classes/ml/advanced_ml/ex4/digit_id_%d.png" % i, digit)

