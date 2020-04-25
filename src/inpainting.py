import numpy as np
import tensorflow as tf
import lightfield.file_io as fio
import matplotlib.pyplot as plt
import models.models as models
import time
import utils.utils as utils

def train(epochs, optimizer, path, mp, dae, masked, type, disp, img):

    if type is "GI":
        func = lambda x: tf.reshape(dae([tf.reshape(x[0], shape=(1,512,512,1)), x[1]]), shape=(512, 512))
    else:
        func = lambda x: tf.reshape(dae(tf.reshape(x, shape=(1,512,512,1))), shape=(512, 512))
    degraded_pixels = np.random.choice([False, True],
                                            size=[512, 512],
                                            p=[1 - masked, masked])
    degraded_image = np.copy(disp)
    degraded_image[degraded_pixels] = 0

    U = tf.Variable(degraded_image, trainable=True)
    start = time.perf_counter()
    for i in range(epochs):
        with tf.GradientTape() as tape:
            # definition of the loss functions
            loss = compute_loss(U, func, img, type)
            grads = tape.gradient(loss, [U])
            optimizer.apply_gradients(zip(grads, [U]))
            U_temp = np.copy(degraded_image)
            if type is "GI":
                inp = [U, img]
            else:
                inp = U
            U_temp[degraded_pixels] = func(inp).numpy()[degraded_pixels]
            U.assign(U_temp)
            del U_temp
            tf.print("Epoch: %d, loss: %1.6f" % (i+1, loss.numpy()))
        if (i+1) % 350 == 0 or loss.numpy() > 100:
            plt.figure(dpi=600)
            plt.subplot(1, 3, 1)
            plt.imshow(degraded_image)
            plt.subplot(1, 3, 2)
            plt.imshow(np.clip(U.numpy(), degraded_image.min(), degraded_image.max()))
            plt.subplot(1, 3, 3)
            plt.imshow(disp)
            plt.title(str(time.perf_counter() - start))
            plt.savefig("/mnt/data/inpainting/" + path + "_350_" + mp + "_"+ str(masked) +  ".png")
            fio.write_pfm("/mnt/data/inpainting/" + path + "_350_" + mp + "_"+ str(masked) + ".pfm", np.clip(U.numpy(), degraded_image.min(), degraded_image.max()))
            break

def compute_loss(U, func, img, type):
    if type is "GI":
        print('HERE')
        inp = [U + tf.random.normal(stddev=0.1*utils.DS_RNG, shape=(512,512)), img]
    else:
        inp = U + tf.random.normal(stddev=0.1*utils.DS_RNG, shape=(512,512))
    prior = tf.reduce_mean(tf.square(func(inp) - U))
    return prior

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

boxes = np.flipud(fio.read_disparity("/mnt/data/benchmark/training/boxes")[0])
boxes_img = fio.read_lightfield("/mnt/data/benchmark/training/boxes")[4,4].reshape(1,512,512,3)
sb = np.flipud(fio.read_disparity("/mnt/data/benchmark/training/sideboard")[0])
sb_img = fio.read_lightfield("/mnt/data/benchmark/training/sideboard")[4,4].reshape(1,512,512,3)

for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    train(350, opt, 'boxes', model_paths[i][1], dae, 0.5, model_paths[i][2], boxes, boxes_img)
for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    train(350, opt, 'sideboard', model_paths[i][1], dae, 0.5, model_paths[i][2], sb, sb_img)
for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    train(350, opt, 'boxes', model_paths[i][1], dae, 0.8, model_paths[i][2], boxes, boxes_img)
for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    train(350, opt, 'sideboard', model_paths[i][1], dae, 0.8, model_paths[i][2], sb, sb_img)

