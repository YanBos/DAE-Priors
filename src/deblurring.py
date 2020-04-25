import numpy as np
import tensorflow as tf
import lightfield.file_io as fio
import matplotlib.pyplot as plt
import utils.utils as utils
from scipy.io import loadmat
import models.models as models

def train(epochs, optimizer, path, mp, dae, type, disp, img):

    # constant stuff
    if type is "GI":
        func = lambda x: tf.reshape(dae([tf.reshape(x[0], shape=(1,512,512,1)), x[1]]), shape=(512, 512))
    else:
        func = lambda x: tf.reshape(dae(tf.reshape(x, shape=(1,512,512,1))), shape=(512, 512))

    decay = tf.Variable(1, dtype=tf.float32)
    filt = loadmat("Kernels/kernel_GT_287.mat")['kernel']
    filt = filt.reshape((filt.shape[0], filt.shape[1], 1, 1))
    noise = np.random.normal(scale=utils.DS_RNG*0.01, size=(1,512,512,1))
    D = tf.Variable(tf.nn.conv2d(disp.reshape((1,512,512,1)), filt, strides=[1, 1, 1, 1], padding="SAME").numpy().reshape((512,512)) + noise.reshape((512,512)), dtype=tf.float32, trainable=True)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            # definition of the loss functions
            loss = compute_loss(disp, D, func, decay, filt, noise, img, type)
            grads = tape.gradient(loss, [D])
            optimizer.apply_gradients(zip(grads, [D]))
            #decay.assign(decay.numpy()/tf.sqrt(float(i + 2)))
            tf.print("Epoch: %d, loss: %1.6f" % (i+1, loss.numpy()))
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
            plt.savefig("/mnt/data/deblurring/" + path + "_350_" + mp + "_287.png")
            fio.write_pfm("/mnt/data/deblurring/" + path + "_350_" + mp  + "_287.pfm",
                          np.clip(D.numpy(), degraded_image.min(), degraded_image.max()))


def compute_loss(disp, D, func, decay, filt, noise, img, type):
    if type is "GI":
        print('HERE')
        inp = [D , img]
    else:
        inp = D
    rec_loss = tf.reduce_mean(tf.square((tf.nn.conv2d(tf.constant(disp.reshape((1,512,512,1)), dtype=tf.float32), filt, strides=[1, 1, 1, 1], padding="SAME") + tf.constant(noise, dtype=tf.float32)) - tf.nn.conv2d(tf.reshape(D, shape=(1,512,512,1)), filt, strides=[1, 1, 1, 1], padding="SAME")))
    prior = tf.reduce_mean(tf.square(func(inp) - D))
    loss = rec_loss + prior
    return loss

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

model_paths = [("/mnt/data/models/FINAL_V2/RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "RES", False),
                    ("/mnt/data/models/FINAL_V2/RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "RES", False),
                     ("/mnt/data/models/FINAL_V2/RES_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","RES_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "RES", True),
                      ("/mnt/data/models/FINAL_V2/RES_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","RES_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "RES", True),
                       ("/mnt/data/models/FINAL_V2/PLAIN_NOBN_NSZ-10_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","PLAIN_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "P", False),
                        ("/mnt/data/models/FINAL_V2/PLAIN_NOBN_NSZ-10_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","PLAIN_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "P", False),
                         ("/mnt/data/models/FINAL_V2/PLAIN_NSZ-10_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","PLAIN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "P", True),
                          ("/mnt/data/models/FINAL_V2/PLAIN_NSZ-10_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","PLAIN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "P", True),(
                   "/mnt/data/models/FINAL_V2/GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "GI", False),
                ("/mnt/data/models/FINAL_V2/GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "GI", False),
                 ("/mnt/data/models/FINAL_V2/GUIDE_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","GUIDE_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "GI", True),
                  ("/mnt/data/models/FINAL_V2/GUIDE_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","GUIDE_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "GI", True)]

dino = np.flipud(fio.read_disparity("/mnt/data/benchmark/training/dino")[0])
dino_img = fio.read_lightfield("/mnt/data/benchmark/training/dino")[4,4].reshape(1,512,512,3)
ct = np.flipud(fio.read_disparity("/mnt/data/benchmark/training/cotton")[0])
ct_img = fio.read_lightfield("/mnt/data/benchmark/training/cotton")[4,4].reshape(1,512,512,3)

for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.5)
    train(350, opt, 'dino', model_paths[i][1], dae, model_paths[i][2], dino, dino_img)
for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.5)
    train(350, opt, 'cotton', model_paths[i][1], dae, model_paths[i][2], ct, ct_img)

