import numpy as np
import tensorflow as tf
import lightfield.file_io as fio
import matplotlib.pyplot as plt
import models.models as models
import utils.utils as utils

# Training loop for the optimization of the disparity map for the case of deblurring.
def train(epochs, optimizer, path, mp, dae, type, disp, img, level):

    # The input to the network must be adapted if it is the guidance network.
    if type is "GI":
        func = lambda x: tf.reshape(dae([tf.reshape(x[0], shape=(1, 512, 512, 1)), x[1]]), shape=(512, 512))
    else:
        func = lambda x: tf.reshape(dae(tf.reshape(x, shape=(1, 512, 512, 1))), shape=(512, 512))

    # Decay variable that is iteratively decreased (a weight for the prior) since for
    # the case of super-resolution there is usually no noise present.
    decay = tf.Variable(1, dtype=tf.float32)

    # The variable we are optimizing for.
    D = tf.Variable(np.zeros_like(disp, dtype=np.float32), trainable=True)

    ## TRAIN LOOP ##
    for i in range(epochs):

        with tf.GradientTape() as tape:

            # Computation of the loss
            loss = compute_loss(disp, D, func, decay, level, img, type)
            grads = tape.gradient(loss, [D])
            optimizer.apply_gradients(zip(grads, [D]))

            # Decreasing the decay variable.
            decay.assign(1/tf.sqrt(float(i + 2)))

            # Print the loss.
            tf.print("Epoch: %d, loss: %1.6f" % (i+1, loss.numpy()))

        # Save a result image and the disparity map after 350 iterations.
        if (i+1) % 350 == 0:
            degraded_image = tf.keras.layers.AveragePooling2D(pool_size=(level, level))\
                (tf.constant(disp, shape=(1, 512, 512, 1), dtype=tf.float32)).numpy().reshape((int(512/level),int(512/level)))
            plt.savefig("../upsampling/" + path + "_350_" + mp + "_" + str(level) + ".png")
            fio.write_pfm("../upsampling/" + path + "_350_" + mp + "_" + str(level) + ".pfm",
                          np.clip(D.numpy(), degraded_image.min(), degraded_image.max()))


# Computation of the loss.
def compute_loss(disp, D, func, decay, level, img, type):
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
bg = np.flipud(fio.read_disparity("../data/boardgames")[0])
bg_img = fio.read_lightfield("../data/boardgames")[4,4].reshape(1,512,512,3)
tower = np.flipud(fio.read_disparity("../data/tower")[0])
tower_img = fio.read_lightfield("../data/tower")[4,4].reshape(1,512,512,3)

# Compute the optimized versions of the degraded disparity maps (4x and 8x upsampling).
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