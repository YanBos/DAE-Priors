import numpy as np
import tensorflow as tf
import lightfield.file_io as fio
import matplotlib.pyplot as plt
import models.models as models
import time
import utils.utils as utils

# Training loop for the optimization of the disparity map for the case of inpainting.
def train(epochs, optimizer, path, mp, dae, masked, type, disp, img):

    # The input to the network must be adapted if it is the guidance network.
    if type is "GI":
        func = lambda x: tf.reshape(dae([tf.reshape(x[0], shape=(1,512,512,1)), x[1]]), shape=(512, 512))
    else:
        func = lambda x: tf.reshape(dae(tf.reshape(x, shape=(1,512,512,1))), shape=(512, 512))

    # Randomly choose a percentage of pixels that need to be inpainted later on.
    degraded_pixels = np.random.choice([False, True],
                                            size=[512, 512],
                                            p=[1 - masked, masked])

    # Setting the chosen pixels to zero.
    degraded_image = np.copy(disp)
    degraded_image[degraded_pixels] = 0

    # Initialize the variable (the degraded image we will now iteratively inpaint) to the initialization,
    # being the degraded image.
    U = tf.Variable(degraded_image, trainable=True)
    start = time.perf_counter()

    ## TRAIN LOOP ##
    for i in range(epochs):

        with tf.GradientTape() as tape:

            # Computation of the loss
            loss = compute_loss(U, func, img, type)

            # Taking the derivative of the loss with respect to our variable (the image we are optimizing for) U.
            grads = tape.gradient(loss, [U])

            # Doing one step of gradient descent (ADAM).
            optimizer.apply_gradients(zip(grads, [U]))

            # Creation of a temporary copy of the degraded image, since we need to reset these pixels
            # in the optimized image U (we want to keep original observations).
            U_temp = np.copy(degraded_image)

            if type is "GI":
                inp = [U, img]
            else:
                inp = U

            # Get the result from the execution of the network and use the values it returns for the degraded pixels.
            U_temp[degraded_pixels] = func(inp).numpy()[degraded_pixels]

            # Reassign U to an improved version having the original values at the pixel locations
            # where real data is present.
            U.assign(U_temp)
            del U_temp

            # Print the loss
            tf.print("Epoch: %d, loss: %1.6f" % (i+1, loss.numpy()))

        # Save a result image and the disparity map after 350 iterations.
        if (i+1) % 350 == 0 or loss.numpy() > 100:
            plt.figure(dpi=600)
            plt.subplot(1, 3, 1)
            plt.imshow(degraded_image)
            plt.subplot(1, 3, 2)
            plt.imshow(np.clip(U.numpy(), degraded_image.min(), degraded_image.max()))
            plt.subplot(1, 3, 3)
            plt.imshow(disp)
            plt.title(str(time.perf_counter() - start))
            plt.savefig("../inpainting/" + path + "_350_" + mp + "_"+ str(masked) +  ".png")
            fio.write_pfm("../inpainting/" + path + "_350_" + mp + "_"+ str(masked) + ".pfm", np.clip(U.numpy(), degraded_image.min(), degraded_image.max()))
            break

# Function computing the loss. This time it only computes the prior (the data term isn't necessary here, since we have locations where no data is present).
def compute_loss(U, func, img, type):
    if type is "GI":
        print('HERE')
        inp = [U + tf.random.normal(stddev=0.1*utils.DS_RNG, shape=(512,512)), img]
    else:
        inp = U + tf.random.normal(stddev=0.1*utils.DS_RNG, shape=(512,512))
    prior = tf.reduce_mean(tf.square(func(inp) - U))
    return prior

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
boxes = np.flipud(fio.read_disparity("../data//boxes")[0])
boxes_img = fio.read_lightfield("../data/boxes")[4,4].reshape(1,512,512,3)
sb = np.flipud(fio.read_disparity("../data/sideboard")[0])
sb_img = fio.read_lightfield("../data/sideboard")[4,4].reshape(1,512,512,3)

# Compute the optimized versions of the degraded disparity maps (For 50 and 80 percent of masked pixels).
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