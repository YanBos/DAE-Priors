import numpy as np
import skimage.transform
import cv2
import ftc.flow_to_color as color
import matplotlib.pyplot as plt
import scipy.signal, scipy.io, scipy.ndimage
import lightfield.file_io as fio
import tensorflow as tf
from skimage import color as skicolor
import models.models as models
import utils.utils as utils
TAR = 4.222982524343324
MAX_ITERS = 5
MAX_ITERS_INNER = 1

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

def tvl1_optical_flow(I0, I1, u1, u2, tau, lam, theta, warps, epsilon, dae, scale):

    # abbreviation for the thresholding step
    lt = lam*theta

    # check if the dimensions agree.
    assert(I0.shape == I1.shape == u1.shape == u2.shape), "The dimensions of the input arrays do not agree."

    # computation of the number of pixels
    width = I0.shape[1]
    height = I0.shape[0]
    size = width * height

    # computation of the gradient of I1
    I1y, I1x = np.gradient(I1)

    kernel_x = np.array([[1, -1, 0]])
    kernel_y = np.array([[1], [-1], [0]])

    x_and_y = np.array(np.dstack(np.meshgrid(np.arange(width), np.arange(height))), dtype=np.float32)

    # OUTER LOOP
    for i in range(warps):

        # get the warped target image and its warped derivatives
        flow = np.array(np.dstack((u1, u2)) + x_and_y, dtype=np.float32)

        I1w = cv2.remap(I1, flow, None, interpolation=cv2.INTER_CUBIC)
        I1wx = cv2.remap(I1x, flow, None, interpolation=cv2.INTER_CUBIC)
        I1wy = cv2.remap(I1y, flow, None, interpolation=cv2.INTER_CUBIC)

        grad = np.square(I1wx) + np.square(I1wy)

        # the constant part of the residual (Why constant? Because we optimize
        # the dual variables in the warping step, keeping one part of the
        # residual constant)
        rho_const = (I1w - I1wx * u1 - I1wy * u2 - I0)

        # iterator for the inner loop of optimization
        n = 0
        error = float("inf")

        # INNER LOOP - Optimization
        while error > epsilon**2 and n < MAX_ITERS:
            n += 1

            # FIRST STEP OF THE DUAL OPTIMIZATION
            rho = rho_const + (I1wx * u1 + I1wy * u2)

            T1 = np.less(rho, - lt * grad)
            T2 = np.greater(rho, lt * grad)
            T3 = np.less(grad, 1e-10)

            # going up the if staircase for better performance
            grad[T3] = 1e-11

            d1 = -rho*I1wx/grad
            d2 = -rho*I1wy/grad

            d1[T3] = 0
            d2[T3] = 0

            d1[T2] = -lt*I1wx[T2]
            d2[T2] = -lt*I1wy[T2]

            d1[T1] = lt*I1wx[T1]
            d2[T1] = lt*I1wy[T1]

            v1 = u1 + d1
            v2 = u2 + d2

            # SECOND STEP OF THE DUAL OPTIMIZATION

            n_inner = 0
            while error > epsilon ** 2 and n_inner < MAX_ITERS_INNER:
                n_inner += 1
                # computing the divergence of the flow fields for the fixed point iteration
                #div_p2 = divergence(p21, p22)

                # before the step
                u1_b = u1
                u2_b = u2

                # after the step
                inp_v1 = tf.constant(v1, shape=(1, height, width, 1), dtype=tf.float32)
                inp_v2 = tf.constant(v2, shape=(1, height, width, 1), dtype=tf.float32)
                I1_rgb = np.array(cv2.cvtColor(I1,cv2.COLOR_GRAY2RGB), dtype=np.float32)/255.
                u1 = dae([inp_v1 + tf.random.normal(stddev=0.05*utils.DS_RNG, shape=(1,height, width, 1)), tf.constant(I1_rgb, shape=(1,height, width, 3))], training=False).numpy().reshape((height, width))
                u2 = dae([inp_v2 + tf.random.normal(stddev=0.05*utils.DS_RNG, shape=(1,height, width, 1)),tf.constant(I1_rgb, shape=(1,height, width, 3))], training=False).numpy().reshape((height, width))

                # calculate the error
                error = np.sum(np.square(u1-u1_b) + np.square(u2-u2_b))

                error /= size

            print("     Inner loop cycles: {}".format(n_inner))
            u1 = scipy.signal.medfilt2d(u1, (5, 5))
            u2 = scipy.signal.medfilt2d(u2, (5, 5))

        print("Outer loop cycles: {}".format(n))
    return u1, u2


def divergence(p1, p2):

    p1x = scipy.signal.convolve2d(p1, [[0, 1, -1]], 'same')
    p2y = scipy.signal.convolve2d(p2, [[0],[1],[-1]], 'same')
    return p1x + p2y

def normalize_ch(I0, I1s):

    dim = I1s.shape[0]
    I1n = np.zeros_like(I1s)
    mx = I0.max() if I0.max() > I1s.max() else I1s.max()
    mn = I0.min() if I0.min() < I1s.min() else I1s.min()
    den = mx - mn
    if den > 0:
        I0n = np.array(255. * (I0 - mn) / den, dtype=np.uint8)
        for i in range(dim):
            I1n[i] = np.array(255. * (I1s[i]-mn) / den, dtype=np.uint8)

    return I0n, I1n

def normalize(I0, I1):

    mx = I0.max() if I0.max() > I1.max() else I1.max()
    mn = I0.min() if I0.min() < I1.min() else I1.min()
    den = mx - mn

    if den > 0:
        I0n = np.array(255. * (I0 - mn) / den, dtype=np.uint8)
        I1n = np.array(255. * (I1 - mn) / den, dtype=np.uint8)

    return I0n, I1n




def tvl1_optical_flow_multiscale(I0, I1, u1, u2, tau, lam, theta, warps, epsilon, nscales=-1, zfactor=2):

    I0, I1 = normalize(I0, I1)

    # not yet computed
    I0s = skimage.transform.pyramid_gaussian(I0, max_layer=nscales, order=3, sigma=0.8)
    I1s = skimage.transform.pyramid_gaussian(I1, max_layer=nscales, order=3, sigma=0.8)

    # not yet computed
    u1s = skimage.transform.pyramid_gaussian(u1, max_layer=nscales)
    u2s = skimage.transform.pyramid_gaussian(u2, max_layer=nscales)

    I0s = reversed([I for I in I0s])
    I0s = [np.array(I*255, np.uint8) for I in I0s]
    I1s = reversed([I for I in I1s])
    I1s = [np.array(I*255, np.uint8) for I in I1s]

    # not yet computed
    u1s = reversed([u1 for u1 in u1s])
    u1s = [u1 for u1 in u1s]
    u2s = reversed([u2 for u2 in u2s])
    u2s = [u2 for u2 in u2s]

    nscales = len(I0s)

    p11 = p12 = p21 = p22 = np.zeros_like(u1s[0])

    for i in range(nscales):
        #dae = tf.saved_model.load(model_paths[i])
        #print("Successfully read saved model %d." % (i + 1))
        u1s[i], u2s[i] = tvl1_optical_flow(I0s[i], I1s[i], u1s[i], u2s[i], tau, lam, theta, warps, epsilon, dae, nscales - 1 - i)

        if i != nscales - 1:
            u1s[i+1] = cv2.resize(u1s[i], (u1s[i+1].shape[1], u1s[i+1].shape[0]), interpolation=cv2.INTER_CUBIC)*zfactor
            u2s[i+1] = cv2.resize(u2s[i], (u2s[i+1].shape[1], u2s[i+1].shape[0]), interpolation=cv2.INTER_CUBIC)*zfactor

    flow = color.flow_to_color(u1s[nscales - 1], u2s[nscales - 1])

    plt.subplot(1,2,1)
    plt.imshow(flow)
    plt.subplot(1,2,2)
    plt.imshow(u1s[nscales - 1])
    #plt.show()
    plt.imsave('/mnt/data/OF/dae_res_0.1_tvl1_' + str(lam) + "_warps_" + str(warps) + ".png", flow)

    return u1s[0], u2s[0]


def import_lf(path):
    images = fio.read_lightfield(path)

    I_c = skicolor.rgb2gray(images[4, 4])
    I_l = skicolor.rgb2gray(images[4, 0])
    I_r = skicolor.rgb2gray(images[4, 8])
    I_t = skicolor.rgb2gray(images[0, 4])
    I_b = skicolor.rgb2gray(images[8, 4])

    return I_c, np.array([I_l, I_r, I_t, I_b])

model_paths = [("/mnt/data/models/FINAL_V2/RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu/weights.h5","RES_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.1-0.1_ACT-relu", "RES", False),
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

dae = get_model(model_paths[0][0], model_paths[0][2], model_paths[0][3])

I1, I0 = cv2.imread("/mnt/data/OF/OF/frame10.png", cv2.IMREAD_GRAYSCALE),cv2.imread("/mnt/data/OF/OF/frame11.png", cv2.IMREAD_GRAYSCALE)
u1, u2 = tvl1_optical_flow_multiscale(I0, I1, np.zeros_like(I0), np.zeros_like(I0), tau=0.25, lam=0.6, theta=0.25, warps=35, nscales=5, epsilon=0.01)

dino = np.flipud(fio.read_disparity("/mnt/data/benchmark/training/dino")[0])
