import numpy as np
import tensorflow as tf
import lightfield.file_io as fio
import matplotlib.pyplot as plt
import models.models as models
import utils.utils as utils

def train(epochs, optimizer, path, mp, dae, type, disp, img, level):
    if type is "GI":
        func = lambda x: tf.reshape(dae([tf.reshape(x[0], shape=(1, 512, 512, 1)), x[1]]), shape=(512, 512))
    else:
        func = lambda x: tf.reshape(dae(tf.reshape(x, shape=(1, 512, 512, 1))), shape=(512, 512))

    decay = tf.Variable(1, dtype=tf.float32)
    D = tf.Variable(np.zeros_like(disp, dtype=np.float32), trainable=True)

    for i in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(disp, D, func, decay, level, img, type)
            grads = tape.gradient(loss, [D])
            optimizer.apply_gradients(zip(grads, [D]))
            decay.assign(1/tf.sqrt(float(i + 2)))
            tf.print("Epoch: %d, loss: %1.6f" % (i+1, loss.numpy()))
        if (i+1) % 350 == 0:
            degraded_image = tf.keras.layers.AveragePooling2D(pool_size=(level, level))\
                (tf.constant(disp, shape=(1, 512, 512, 1), dtype=tf.float32)).numpy().reshape((int(512/level),int(512/level)))
            plt.savefig("/mnt/data/upsampling/" + path + "_350_" + mp + "_" + str(level) + ".png")
            fio.write_pfm("/mnt/data/upsampling/" + path + "_350_" + mp + "_" + str(level) + ".pfm",
                          np.clip(D.numpy(), degraded_image.min(), degraded_image.max()))


# definition of the loss functions
def compute_loss(disp, D, func, decay, level, img, typea):
    if type is "GI":
        inp = [D + tf.random.normal(stddev=0.1 * utils.DS_RNG, shape=(512, 512)), img]
    else:
        inp = D + tf.random.normal(stddev=0.1 * utils.DS_RNG, shape=(512, 512))
    rec_loss = tf.reduce_mean(tf.square(tf.keras.layers.AveragePooling2D(pool_size=(level, level))
                                        (tf.constant(disp, shape=(1, 512, 512, 1), dtype=tf.float32))
                                        - tf.keras.layers.AveragePooling2D(pool_size=(level, level))(
        tf.reshape(D, shape=(1, 512, 512, 1)))))
    prior = tf.reduce_mean(tf.square(func(inp) - D))
    loss = rec_loss + decay * prior
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
                          ("/mnt/data/models/FINAL_V2/PLAIN_NSZ-10_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","PLAIN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "P", True),
               ("/mnt/data/models/FINAL_V2/GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "GI", False),
                ("/mnt/data/models/FINAL_V2/GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "GI", False),
                 ("/mnt/data/models/FINAL_V2/GUIDE_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","GUIDE_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "GI", True),
                  ("/mnt/data/models/FINAL_V2/GUIDE_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","GUIDE_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "GI", True)]

bg = np.flipud(fio.read_disparity("/mnt/data/lightfield/full_data/additional/boardgames")[0])
bg_img = fio.read_lightfield("/mnt/data/lightfield/full_data/additional/boardgames")[4,4].reshape(1,512,512,3)
tower = np.flipud(fio.read_disparity("/mnt/data/lightfield/full_data/additional/tower")[0])
tower_img = fio.read_lightfield("/mnt/data/lightfield/full_data/additional/tower")[4,4].reshape(1,512,512,3)

for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.3)
    train(350, opt, 'bg', model_paths[i][1], dae, model_paths[i][2], bg, bg_img, 4)
for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.3)
    train(350, opt, 'tower', model_paths[i][1], dae, model_paths[i][2], tower, tower_img, 4)
for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.3)
    train(350, opt, 'bg', model_paths[i][1], dae, model_paths[i][2],bg, bg_img, 8)
for i in range(len(model_paths)):
    dae = get_model(model_paths[i][0], model_paths[i][2], model_paths[i][3])
    opt = tf.keras.optimizers.Adam(learning_rate=0.3)
    train(350, opt, 'tower', model_paths[i][1], dae, model_paths[i][2], tower, tower_img, 8)