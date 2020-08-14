import numpy as np
import tensorflow as tf
import lightfield.file_io as fio
import matplotlib.pyplot as plt
import utils.utils as utils
from scipy.io import loadmat
import models.models as models

# Training loop for the optimization of the disparity map for the case of deblurring.
def train(epochs, optimizer, path, mp, dae, type, disp, img):

    # The input to the network must be adapted if it is the guidance network.
    if type is "GI":
        func = lambda x: tf.reshape(dae([tf.reshape(x[0], shape=(1,512,512,1)), x[1]]), shape=(512, 512))
    else:
        func = lambda x: tf.reshape(dae(tf.reshape(x, shape=(1,512,512,1))), shape=(512, 512))

    # Right here one must specify the kernel necessary to prepare the degraded version of the image.
    filt = loadmat("Kernels/kernel_GT_287.mat")['kernel']
    filt = filt.reshape((filt.shape[0], filt.shape[1], 1, 1))

    # Convolute and add noise to the disparity map.
    noise = np.random.normal(scale=utils.DS_RNG*0.01, size=(1,512,512,1))
    D = tf.Variable(tf.nn.conv2d(disp.reshape((1,512,512,1)), filt, strides=[1, 1, 1, 1], padding="SAME").numpy().reshape((512,512)) + noise.reshape((512,512)), dtype=tf.float32, trainable=True)

    ## TRAINING LOOP ##
    for i in range(epochs):
        with tf.GradientTape() as tape:

            # Computation of the loss and application of the gradient descent update for D.
            loss = compute_loss(disp, D, func, filt, noise, img, type)
            grads = tape.gradient(loss, [D])
            optimizer.apply_gradients(zip(grads, [D]))

            # Print the loss
            tf.print("Epoch: %d, loss: %1.6f" % (i+1, loss.numpy()))

        # Save a result image after 350 iterations.
        if (i+1) % 350 == 0 or loss.numpy() > 100:
            plt.figure(dpi=600)
            plt.subplot(1, 4, 1)
            plt.axis('off')
            blurred = tf.nn.conv2d(disp.reshape((1,512,512,1)), filt, strides=[1, 1, 1, 1], padding="SAME").numpy().reshape((512,512)) + noise.reshape((512, 512))
            plt.imshow(blurred)
            plt.subplot(1, 4, 2)
            plt.axis('off')
            plt.imshow(np.clip(D.numpy(), D.numpy().min(), D.numpy().max()))
            plt.subplot(1, 4, 3)
            plt.axis('off')
            plt.imshow(disp)
            plt.subplot(1, 4, 4)
            plt.axis('off')
            plt.imshow(filt.reshape((filt.shape[0], filt.shape[1]))/filt.max())
            degraded_image = (tf.nn.conv2d(disp.reshape((1, 512, 512, 1)), filt, strides=[1, 1, 1, 1], padding="SAME").numpy().reshape(
                (512, 512)) + noise.reshape((512, 512)))
            plt.savefig("../deblurring/" + path + "_350_" + mp + "_287.png")
            fio.write_pfm("../deblurring/" + path + "_350_" + mp  + "_287.pfm",
                          np.clip(D.numpy(), degraded_image.min(), degraded_image.max()))

# Computation of the loss.
def compute_loss(disp, D, func, filt, noise, img, type):

    # Again we have to adapt the input for the guidance network.
    if type is "GI":
        print('HERE')
        inp = [D , img]
    else:
        inp = D

    # Compute the loss
    rec_loss = tf.reduce_mean(tf.square((tf.nn.conv2d(tf.constant(disp.reshape((1,512,512,1)), dtype=tf.float32), filt, strides=[1, 1, 1, 1], padding="SAME") + tf.constant(noise, dtype=tf.float32)) - tf.nn.conv2d(tf.reshape(D, shape=(1,512,512,1)), filt, strides=[1, 1, 1, 1], padding="SAME")))

    # Execution of the network and computation of the prior.
    prior = tf.reduce_mean(tf.square(func(inp) - D))
    loss = rec_loss + prior

    return loss

# Construction of the models and loading of the weights.
def get_model(path, type, batch_norm):
    if type is 'GI':
        print('HERE')
        dae = models.GuidanceNet(batch_norm=batch_norm, sz=5).model
    elif type is 'P':
        dae = models.PlainNet(batch_norm=batch_norm, sz=10).model
    else:
        dae = models.ResNet(batch_norm=batch_norm, sz=5).model
    dae.load_weights(path)

    return dae

# Paths of the models one wants to evaluate.
model_paths = [("../Final/Networks/RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "RES", False)]

# Load the disparity maps and the corresponding images.
dino = np.flipud(fio.read_disparity("../data/dino")[0])
dino_img = fio.read_lightfield("../data/dino")[4,4].reshape(1,512,512,3)
ct = np.flipud(fio.read_disparity("../data/cotton")[0])
ct_img = fio.read_lightfield("../data/cotton")[4,4].reshape(1,512,512,3)

# Compute the optimized versions of the degraded disparity maps.
for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.5)
    train(350, opt, 'dino', model_paths[i][1], dae, model_paths[i][2], dino, dino_img)
for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.5)
    train(350, opt, 'cotton', model_paths[i][1], dae, model_paths[i][2], ct, ct_img)

