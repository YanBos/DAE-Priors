# Copyright (c) 2011, Javier Sánchez Pérez, Enric Meinhardt Llopis
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################
#
# The following implementation by Sánchez et al. [1] of the Dual TVL1 Optical Flow algorithm presented by
# Zach et al. [2] was translated into python by Yannick Bosch (yannick.bosch@uni-konstanz.de) and replaces the solution
# to the Total Variation based denoising problem within the scheme presented in [2] with a Denoising Autoencoder.
# Further instead of using an L1 loss this version uses an L2 loss.
#
# [1] Sánchez Pérez, Javier and Meinhardt-Llopis, Enric and Facciolo, Gabriele,
# TV-L1 Optical Flow Estimation,
# Image Processing On Line, Volume 3, pp. 137-150, 2013.
#
# [2] Zach, Christopher & Pock, Thomas & Bischof, Horst. (2007).
# A Duality Based Approach for Realtime TV-L1 Optical Flow.
# Pattern Recognition. 4713. 214-223. 10.1007/978-3-540-74936-3_22.

import numpy as np
import skimage.transform
import cv2
import ftc.flow_to_color as color
import matplotlib.pyplot as plt
import scipy.signal, scipy.io, scipy.ndimage
import tensorflow as tf
import models.models as models
import utils.utils as utils

TAR = 4.222982524343324
MAX_ITERS = 5
MAX_ITERS_INNER = 1

# Creation of the network and loading of its weights.
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

# Function for the computation of the optical flow (one scale).
def tvl2_optical_flow(I0, I1, u1, u2, tau, lam, theta, warps, epsilon, scale):

    # abbreviation for the thresholding step
    lt = lam*theta
    taut = tau/theta

    # check if the dimensions agree.
    assert(I0.shape == I1.shape == u1.shape == u2.shape), "The dimensions of the input arrays do not agree."

    # computation of the number of pixels
    width = I0.shape[1]
    height = I0.shape[0]
    size = width * height

    v1 = u1
    v2 = u2
    # define the value of the gradient of I1 (target image)
    grad = np.zeros_like(u1)

    # computation of the gradient of I1
    I1y, I1x = np.gradient(I1)

    kernel_x = np.array([[1, -1, 0]])
    kernel_y = np.array([[1], [-1], [0]])

    # OUTER LOOP
    for i in range(warps):

        # get the warped target image and its warped derivatives
        x_and_y = np.array(np.dstack(np.meshgrid(np.arange(width), np.arange(height))), dtype=np.float32)
        flow = np.array(np.dstack((u1, u2)) + x_and_y, dtype=np.float32)

        I1w = cv2.remap(I1, flow, None, interpolation=cv2.INTER_CUBIC)
        I1wx = cv2.remap(I1x, flow, None, interpolation=cv2.INTER_CUBIC)
        I1wy = cv2.remap(I1y, flow, None, interpolation=cv2.INTER_CUBIC)

        u_0 = u1
        u_02 = u2
        # iterator for the inner loop of optimization
        n = 0
        error = float("inf")

        # INNER LOOP - Optimization
        while error > epsilon**2 and n < MAX_ITERS:
            n += 1

            v1 = 1/(1+lt*I1wx**2) *(u1 - lt*I1wx*(I1w - u_0*I1wx - I0))
            v2 = 1/(1+lt*I1wy**2) * (u2 - lt*I1wy*(I1w - u_02 * I1wy - I0))

            n_inner = 0
            while error > epsilon**2 and n_inner < MAX_ITERS_INNER:
                n_inner += 1

                u1_b = u1
                u2_b = u2

                inp_v1 = tf.constant(v1, shape=(1, height, width, 1), dtype=tf.float32)
                inp_v2 = tf.constant(v2, shape=(1, height, width, 1), dtype=tf.float32)
                I1_rgb = np.array(cv2.cvtColor(I1, cv2.COLOR_GRAY2RGB), dtype=np.float32) / 255.
                u1 = dae([inp_v1 + tf.random.normal(stddev=0.05 * utils.DS_RNG, shape=(1, height, width, 1)),
                          tf.constant(I1_rgb, shape=(1, height, width, 3))], training=False).numpy().reshape(
                    (height, width))
                u2 = dae([inp_v2 + tf.random.normal(stddev=0.05 * utils.DS_RNG, shape=(1, height, width, 1)),
                          tf.constant(I1_rgb, shape=(1, height, width, 3))], training=False).numpy().reshape(
                    (height, width))

                # calculate the error
                error = np.sum(np.square(u1-u1_b) + np.square(u2-u2_b))

                error /= size

            print("     Inner loop cycles: {}".format(n_inner))
            u1 = scipy.signal.medfilt2d(u1, (5, 5))
            u2 = scipy.signal.medfilt2d(u2, (5, 5))

        print("Outer loop cycles: {}".format(n))

    return u1, u2

# Computation of the divergence.
def divergence(p1, p2):

    p1x = scipy.signal.convolve2d(p1, [[0, 1, -1]], 'same')
    p2y = scipy.signal.convolve2d(p2, [[0],[1],[-1]], 'same')
    return p1x + p2y

# Normalization of the images according to the minimum and the maximum of all images.
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

# Normalization of the images according to the minimum and the maximum of both images (for optical flow).
def normalize(I0, I1):

    mx = I0.max() if I0.max() > I1.max() else I1.max()
    mn = I0.min() if I0.min() < I1.min() else I1.min()
    den = mx - mn

    if den > 0:
        I0n = np.array(255. * (I0 - mn) / den, dtype=np.uint8)
        I1n = np.array(255. * (I1 - mn) / den, dtype=np.uint8)

    return I0n, I1n

# The wrapper for tvl2_dae (all scales).
def tvl2_optical_flow_multiscale(I0, I1, u1, u2, tau, lam, theta, warps, epsilon, nscales=-1, zfactor=2):

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
    for i in range(nscales):
        u1s[i], u2s[i] = tvl2_optical_flow(I0s[i], I1s[i], u1s[i], u2s[i], tau, lam, theta, warps, epsilon, nscales -1 -i)

        if i != nscales - 1:
            u1s[i+1] = cv2.resize(u1s[i], (u1s[i+1].shape[1], u1s[i+1].shape[0]), interpolation=cv2.INTER_CUBIC)*zfactor
            u2s[i+1] = cv2.resize(u2s[i], (u2s[i+1].shape[1], u2s[i+1].shape[0]), interpolation=cv2.INTER_CUBIC)*zfactor

    flow = color.flow_to_color(u1s[nscales - 1], u2s[nscales - 1])

    plt.subplot(1,2,1)
    plt.imshow(flow)
    plt.subplot(1,2,2)
    plt.imshow(u1s[nscales - 1])
    plt.show()
    plt.imsave('../data/OF/dae_guide_tvl2_' + str(lam) + "_warps_" + str(warps) + ".png", flow)

    return u1s[nscales - 1], u2s[nscales - 1]

# Paths of the models one wants to evaluate.
model_paths = [("../Final/Networks/RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "RES", False)]

dae = get_model(model_paths[0][0], model_paths[0][2], model_paths[0][3])

I1, I0 = cv2.imread("../data/OF/frame10.png", cv2.IMREAD_GRAYSCALE),cv2.imread("../data/OF/frame11.png", cv2.IMREAD_GRAYSCALE)

# Estimate the optical flow for the example image sequence in the thesis.
u1, u2 = tvl2_optical_flow_multiscale(I0, I1, np.zeros_like(I0), np.zeros_like(I0), tau=0.25, lam=0.6, theta=0.25, warps=2, nscales=5, epsilon=0.01)
