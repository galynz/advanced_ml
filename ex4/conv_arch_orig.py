from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0
fix_var = False


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# Definition of Keras ConvNet architecture

input_shape = (28, 28, 1)
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
# shape info needed to build decoder model
shape = K.int_shape(x)
# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

dense1 = Dense(intermediate_dim, activation='relu')
dense2 = Dense(shape[1] * shape[2] * shape[3], activation='relu')
reshape1 = Reshape((shape[1], shape[2], shape[3]))
conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')
upsampling1 = UpSampling2D((2, 2))
conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')
upsampling2 = UpSampling2D((2, 2))
conv3 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
flatten1 = Flatten()
decoder_mean = Dense(original_dim, activation='sigmoid')

x = dense1(z)
x = dense2(x)
x = reshape1(x)
x = conv1(x)
x = upsampling1(x)
x = conv2(x)
x = upsampling2(x)
x = conv3(x)
x = flatten1(x)
x_decoded_mean = decoder_mean(x)

# x = Dense(intermediate_dim, activation='relu')(z)
# x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
# x = Reshape((shape[1], shape[2], shape[3]))(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
# x = Flatten()(outputs)
# decoder_mean = Dense(original_dim, activation='sigmoid')
# x_decoded_mean = decoder_mean(x)


# vae = Model(inputs, outputs, name='vae')
vae = Model(inputs, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))


# build a model to project inputs on the latent space
encoder = Model(inputs, z)
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


# Take one image per digit and print its corresponding mapping coordinates in the latent space
row_format ="{:>20}" * 3
print(row_format.format("", *["x", "y"]))
for i in xrange(10):
    print(row_format.format(i, *x_test_encoded[np.random.choice(np.where(y_test == i)[0])]))


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_x = dense1(decoder_input)
_x = dense2(_x)
_x = reshape1(_x)
_x = conv1(_x)
_x = upsampling1(_x)
_x = conv2(_x)
_x = upsampling2(_x)
_x = conv3(_x)
_x = flatten1(_x)
_x_decoded_mean = decoder_mean(_x)
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
    plt.imsave("/Users/gal/Dropbox/gal/classes/ml/advanced_ml/ex4/cnn_digit_id_%d.png" % i, digit)

