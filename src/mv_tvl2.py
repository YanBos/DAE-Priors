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
# Zach et al. [2] was translated into python by Yannick Bosch (yannick.bosch@uni-konstanz.de).
# The algorithm was modified to work with lightfields (thus using an L2 loss).
#
# [1] Sánchez Pérez, Javier and Meinhardt-Llopis, Enric and Facciolo, Gabriele,
# TV-L1 Optical Flow Estimation,
# Image Processing On Line, Volume 3, pp. 137-150, 2013.
#
# [2] Zach, Christopher & Pock, Thomas & Bischof, Horst. (2007).
# A Duality Based Approach for Realtime TV-L1 Optical Flow.
# Pattern Recognition. 4713. 214-223. 10.1007/978-3-540-74936-3_22.

import skimage.transform
import cv2
import ftc.flow_to_color as color
import matplotlib.pyplot as plt
import scipy.signal, scipy.io, scipy.ndimage
import lightfield.file_io as fio
import os
from skimage import color as skicolor
import time

MAX_ITERS = 5
MAX_ITERS_INNER = 1
VERBOSE = True
WARPED = False

# Function for importing the crosshair of a lightfield in grayscale.
def import_lf(path):
    images = fio.read_lightfield(path)

    I_c = skicolor.rgb2gray(images[4, 4])
    I_l = skicolor.rgb2gray(images[4, 0])
    I_r = skicolor.rgb2gray(images[4, 8])
    I_t = skicolor.rgb2gray(images[0, 4])
    I_b = skicolor.rgb2gray(images[8, 4])

    return I_c, np.array([I_l, I_r, I_t, I_b])

# Function for importing an entire lightfield in grayscale.
def import_lf_whole(path):
    images = fio.read_lightfield(path)
    images_gray = np.zeros(shape=[9, 9, 512, 512])
    for i in range(9):
        for j in range(9):
            images_gray[i, j] = skicolor.rgb2gray(images[i, j])

    return images_gray

# Computates the gradients for each image in the crosshair.
def compute_grads(I1s):

    I1s_grads = []

    for i in range(I1s.shape[0]):
        I1s_grads.append(np.gradient(I1s[i]))

    return np.array(I1s_grads)

# Computates the gradients for each image in the entire lightfield.
def compute_grads_whole(Is):

    Is_grads = np.zeros(shape=[9, 9, 2, Is[0, 0].shape[0], Is[0, 0].shape[1]])

    for i in range(9):
        for j in range(9):
            I_y, I_x = np.gradient(Is[i, j])
            Is_grads[i, j, 0] = I_x
            Is_grads[i, j, 1] = I_y

    return Is_grads

# Computation of the derivative for the update step of v (crosshair) as described in the appendix of my thesis.
def derivative(I1s, I1s_grads, I0, u1):

    width = I0.shape[1]
    height = I0.shape[0]

    # Empty sets
    I1s_grads_warped = []
    I1s_warped = []

    # coordinates for the fields
    x_and_y = np.array(np.dstack(np.meshgrid(np.arange(width), np.arange(height))), dtype=np.float32)

    ch = 1

    for i in range(I1s.shape[0]):
        if ch == 1:
            disp = np.dstack((u1/4., np.zeros_like(u1)))
        elif ch == 2:
            disp = -np.dstack((u1/4., np.zeros_like(u1)))
        elif ch == 3:
            disp = np.dstack((np.zeros_like(u1), u1/4.))
        else:
            disp = -np.dstack((np.zeros_like(u1), u1/4.))

        # map the images and gradients.
        flow = np.array(x_and_y + disp, dtype=np.float32)

        I1w, I1wx, I1wy = map_images(flow, I1s[i], I1s_grads[i][1], I1s_grads[i][0])

        ch += 1
        I1s_grads_warped.append((I1wy, I1wx))
        I1s_warped.append(I1w)

    I1s_grads_warped = np.array(I1s_grads_warped)
    I1s_warped = np.array(I1s_warped)

    if WARPED:
        disp_warped(I1s_warped)

    I_l = (I1s_warped[0] - u1/4. * I1s_grads_warped[0][1] - I0) * I1s_grads_warped[0][1]
    I_r = - (I1s_warped[1] + u1/4. * I1s_grads_warped[1][1] - I0) * I1s_grads_warped[1][1]
    I_t = + (I1s_warped[2] - u1/4. * I1s_grads_warped[2][0] - I0) * I1s_grads_warped[2][0]
    I_b = - (I1s_warped[3] + u1/4. * I1s_grads_warped[3][0] - I0) * I1s_grads_warped[3][0]

    A = I1s_grads_warped[0][1] ** 2 + I1s_grads_warped[1][1] ** 2 + I1s_grads_warped[2][0] ** 2 + I1s_grads_warped[3][0] ** 2
    B = np.array(I_l + I_r + I_t + I_b)

    return A, B

# Computation of the derivative for the update step of v (entire lightfield) as described in the appendix of my thesis.
def derivative_whole(Is, Is_grads, I0, u):
    
    Is_warped, Is_grads_warped = map_images_whole(Is, Is_grads, u)

    if WARPED:
        disp_warped(Is_warped)

    A = 0
    B = 0

    for i in range(9):
        for j in range(9):
            if i == 4 and j == 4:
                continue

            if i == 4 and j == 0:
                b = 0
            elif i == 4 and j == 8:
                b = 0
            elif i == 0 and j == 4:
                b = 0
            elif i == 8 and j == 0:
                b = 0
            in_d = (I_lf[i, j, 0]*Is_grads_warped[i, j, 0] + I_lf[i, j, 1]*Is_grads_warped[i, j, 1])
            inc = (Is_warped[i, j] - I0 - u*in_d)*in_d
            B += inc
            A += in_d**2

    return A, B

# Resampling of image I1 and its derivatives according to the estimated disparity map.
def map_images(flow, I1, I1x, I1y):

    I1w = cv2.remap(I1, flow, None, interpolation=cv2.INTER_CUBIC)
    I1wx = cv2.remap(I1x, flow, None, interpolation=cv2.INTER_CUBIC)
    I1wy = cv2.remap(I1y, flow, None, interpolation=cv2.INTER_CUBIC)

    return I1w, I1wx, I1wy

# Resampling of all images I1 (the entire ligthfield except for the central image) and
# its derivatives according to the estimated disparity map.
def map_images_whole(Is, Is_grads, u):
    
    # Empty sets
    Is_grads_warped = np.zeros_like(Is_grads)
    Is_warped = np.zeros_like(Is)

    # coordinates for the fields
    x_and_y = np.array(np.dstack(np.meshgrid(np.arange(Is[0, 0].shape[1]), np.arange(Is[0, 0].shape[0]))), dtype=np.float32)

    for i in range(9):
        for j in range(9):

            disp = np.dstack((I_lf[i, j, 0]*u, I_lf[i, j, 1]*u))
            flow = np.array(x_and_y + disp, dtype=np.float32)
            Is_warped[i, j] = cv2.remap(Is[i, j], flow, None, interpolation=cv2.INTER_CUBIC)
            Is_grads_warped[i, j, 0] = cv2.remap(Is_grads[i, j, 0], flow, None, interpolation=cv2.INTER_CUBIC)
            Is_grads_warped[i, j, 1] = cv2.remap(Is_grads[i, j, 1], flow, None, interpolation=cv2.INTER_CUBIC)

    return Is_warped/1., Is_grads_warped/1.

# Show the warped images.
def disp_warped(I1s_warped):

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(I1s_warped[i])
    plt.show()

# Computation of the forward gradient.
def forward_gradient(F):

    width = F.shape[1]
    height = F.shape[0]

    kernel_x = np.array([[1, -1, 0]])
    kernel_y = np.array([[1], [-1], [0]])

    Fx = scipy.signal.convolve2d(F, kernel_x, mode='same')
    Fx[:, width - 1] = 0
    Fy = scipy.signal.convolve2d(F, kernel_y, mode='same')
    Fy[height - 1, :] = 0

    return Fx, Fy

# Disparity estimation with the L2 norm using the crosshair of the lightfield (for one scale).
def mv_tvl2(I0, I1s, u1, tau, lam, theta, warps, epsilon):

    # abbreviation for the thresholding step
    lt = lam*theta
    taut = tau/theta

    # check if the dimensions agree.
    assert(I0.shape == I1s[0].shape == u1.shape), "The dimensions of the input arrays do not agree."

    # computation of the number of pixels
    width = I0.shape[1]
    height = I0.shape[0]
    size = width * height

    # computation of the gradient of I1
    I1s_grads = compute_grads(I1s)
    p11, p12, p21, p22 = np.zeros_like(u1), np.zeros_like(u1), np.zeros_like(u1), np.zeros_like(u1)

    # OUTER LOOP
    for i in range(warps):

        # compute the derivatives for the first step
        A, B = derivative(I1s, I1s_grads, I0, u1)

        # iterator for the inner loop of optimization
        n = 0
        error = float("inf")

        # INNER LOOP - Optimization
        while error > epsilon**2 and n < MAX_ITERS:
            n += 1

            v1 = (u1 - lt/4*B)/(1 + lt/4*A)

            n_inner = 0
            while error > epsilon**2 and n_inner < MAX_ITERS_INNER:
                n_inner += 1
                # computing the divergence of the flow fields for the fixed point iteration
                div_p1 = divergence(p11, p12)

                # before the step
                u1_b = u1

                # after the step
                u1 = v1 + theta * div_p1

                # calculate the error
                error = np.sum(np.square(u1-u1_b))

                error /= size

                # calculate the forward gradient
                u1x, u1y = forward_gradient(u1)

                # update the dual variables
                p11 = (p11 + taut * u1x) / (1.0 + taut * np.sqrt(np.square(u1x) + np.square(u1y)))
                p12 = (p12 + taut * u1y) / (1.0 + taut * np.sqrt(np.square(u1x) + np.square(u1y)))
            if VERBOSE:
                print("     Inner loop cycles: {}".format(n_inner))
            # Filter u1 and u2 to smooth out the field.
            u1 = scipy.signal.medfilt2d(u1, (3, 3))
        if VERBOSE:
            print("Outer loop cycles: {}".format(n))

    return u1, p11, p12

# Disparity estimation with the L2 norm using the entire lightfield (for one scale).
def mv_tvl2_whole(I0, Is, u1, tau, lam, theta, warps):

    # abbreviation for the thresholding step
    lt = lam*theta
    taut = tau/theta

    # check if the dimensions agree.
    assert(I0.shape == Is[0, 0].shape == u1.shape), "The dimensions of the input arrays do not agree."

    # computation of the number of pixels
    width = I0.shape[1]
    height = I0.shape[0]
    size = width * height

    # computation of the gradient of I1
    Is_grads = compute_grads_whole(Is)

    p11, p12, p21, p22 = np.zeros_like(u1), np.zeros_like(u1), np.zeros_like(u1), np.zeros_like(u1)

    # OUTER LOOP
    for i in range(warps):

        # compute the derivatives for the first step
        A, B = derivative_whole(Is, Is_grads, I0, u1)

        # iterator for the inner loop of optimization
        n = 0
        error = float("inf")

        # INNER LOOP - Optimization
        while error > (1e-2)**2 and n < MAX_ITERS:
            n += 1

            v1 = (u1 - lt/80*B)/(1 + lt/80*A)

            # SECOND STEP OF THE DUAL OPTIMIZATION
            n_inner = 0
            while error > (1e-2)**2 and n_inner < MAX_ITERS_INNER:
                n_inner += 1
                # computing the divergence of the flow fields for the fixed point iteration
                div_p1 = divergence(p11, p12)

                # before the step
                u1_b = u1

                # after the step
                u1 = v1 + theta * div_p1

                # calculate the error
                error = np.sum(np.square(u1-u1_b))

                error /= size

                # calculate the forward gradient
                u1x, u1y = forward_gradient(u1)

                # update the dual variables
                p11 = (p11 + taut * u1x) / (1.0 + taut * np.sqrt(np.square(u1x) + np.square(u1y)))
                p12 = (p12 + taut * u1y) / (1.0 + taut * np.sqrt(np.square(u1x) + np.square(u1y)))
            if VERBOSE:
                print("     Inner loop cycles: {}".format(n_inner))
            # Filter u1 and u2 to smooth out the field.
            u1 = scipy.signal.medfilt2d(u1, (5, 5))
        if VERBOSE:
            print("Outer loop cycles: {}".format(n))

    return u1

# Computation of the divergence
def divergence(p1, p2):

    p1x = scipy.signal.convolve2d(p1, [[0, 1, -1]], 'same')
    p2y = scipy.signal.convolve2d(p2, [[0],[1],[-1]], 'same')
    return p1x + p2y

# Normalization of the images according to the minimum and maximum of the crosshair images of the lightfield.
def normalize(I0, I1s):

    # Computation of minimum and maximum of all images together
    mx = [I0.max()]
    mn = [I0.min()]

    for i in range(I1s.shape[0]):
        mx.append(I1s[i].max())
        mn.append(I1s[i].min())

    mx = np.array(mx).max()
    mn = np.array(mn).min()
    den = mx - mn

    # fallback images
    I1s_n = I1s
    I0_n = I0

    if den > 0:
        I0n = np.array(255. * (I0 - mn) / den, dtype=np.uint8)
        for i in range(I1s.shape[0]):
            I1s_n[i] = np.array(255. * (I1s[i] - mn) / den, dtype=np.uint8)

    return I0n, I1s_n

# Normalization of the images according to the minimum and maximum of all images in the lightfield.
def normalize_whole(I0, Is):

    # Computation of minimum and maximum of all images together
    mx = [I0.max()]
    mn = [I0.min()]

    mx.append(Is.max())
    mn.append(Is.min())

    mx = np.array(mx).max()
    mn = np.array(mn).min()
    den = mx - mn

    # fallback images
    Is_n = Is
    I0_n = I0

    if den > 0:
        I0_n = np.array(255. * (I0 - mn) / den, dtype=np.uint8)
        Is_n = np.array(255. * (Is - mn) / den, dtype=np.uint8)

    return I0_n, Is_n

# The wrapper for mv_tvl2_whole (all scales).
def mv_tvl2_multiscale_whole(I0, Is, u1, tau, lam, theta, warps, nscales=-1, zfactor=2):

    Is_scaled = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for x in range(len(Is_scaled)):
        Is_scaled[x] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    I0, Is = normalize_whole(I0, Is)

    I0s = skimage.transform.pyramid_gaussian(I0/255, max_layer=nscales, order=3, sigma=0.8)

    # Construction of the image pyramid for every image in the lightfield
    for i in range(9):
        for j in range(9):
            # bring the images in the right order and range.
            I1_scaled = skimage.transform.pyramid_gaussian(Is[i, j]/255, max_layer=nscales, order=3, sigma=0.8)
            I1_scaled = reversed([I for I in I1_scaled])
            I1_scaled = [np.array(I*255, dtype=np.uint8) for I in I1_scaled]
            Is_scaled[i][j] = I1_scaled

    # bring the central images in the right order and range.
    I0s = reversed([I for I in I0s])
    I0s = [np.array(I*255, dtype=np.uint8) for I in I0s]

    # not yet computed
    u1s = skimage.transform.pyramid_gaussian(u1, max_layer=nscales)

    # not yet computed
    u1s = reversed([u1 for u1 in u1s])
    u1s = [u1 for u1 in u1s]

    # compute the number of scales
    nscales = len(I0s)

    Is_scaled = np.array(Is_scaled)

    # Execution of the multiview
    for i in range(nscales):
        u1s[i] = mv_tvl2_whole(I0s[i], Is_scaled[:, :, i], u1s[i], tau, lam, theta, warps)

        flow = color.flow_to_color(u1s[i]/den, np.zeros_like(u1s[i]))
        if VERBOSE:
            disp_flow_and_u(flow, u1s[i]/den)

        if i != nscales - 1:
            u1s[i+1] = cv2.resize(u1s[i], (u1s[i+1].shape[1], u1s[i+1].shape[0]), interpolation=cv2.INTER_CUBIC)*zfactor
    flow = color.flow_to_color(u1s[nscales - 1]/den, np.zeros_like(u1s[i]))
    disp_flow_and_u(flow, u1s[nscales - 1]/den)

    return u1s[nscales - 1]/den

# The wrapper for mv_tvl2 (all scales).
def mv_tvl2_multiscale(I0, I1s, u1, tau, lam, theta, warps, epsilon, nscales=-1, zfactor=2):

    I0, I1s = normalize(I0, I1s)

    I0s = skimage.transform.pyramid_gaussian(I0/255, max_layer=nscales, order=3, sigma=0.8)
    I1s_scaled = []

    # Compute image pyramids for every image in the crosshair.
    for i in range(I1s.shape[0]):
        # bring the crosshair images in the right order and range.
        I1_scaled = skimage.transform.pyramid_gaussian(I1s[i]/255, max_layer=nscales, order=3, sigma=0.8)
        I1_scaled = reversed([I for I in I1_scaled])
        I1_scaled = [np.array(I*255, dtype=np.uint8) for I in I1_scaled]
        I1s_scaled.append(I1_scaled)

    # bring the central images in the right order and range.
    I0s = reversed([I for I in I0s])
    I0s = [np.array(I*255, dtype=np.uint8) for I in I0s]

    # not yet computed
    u1s = skimage.transform.pyramid_gaussian(u1, max_layer=nscales)

    # not yet computed
    u1s = reversed([u1 for u1 in u1s])
    u1s = [u1 for u1 in u1s]

    # compute the number of scales
    nscales = len(I0s)

    I1s_scaled = np.array(I1s_scaled)

    for i in range(nscales):
        u1s[i] = mv_tvl2(I0s[i], I1s_scaled[:, i], u1s[i], tau, lam, theta, warps, epsilon)

        flow = color.flow_to_color(u1s[i], np.zeros_like(u1s[i]))
        if VERBOSE:
            disp_flow_and_u(flow, u1s[i])

        if i != nscales - 1:
            u1s[i+1] = cv2.resize(u1s[i], (u1s[i+1].shape[1], u1s[i+1].shape[0]), interpolation=cv2.INTER_CUBIC)*zfactor
    flow = color.flow_to_color(u1s[nscales - 1], np.zeros_like(u1s[i]))
    disp_flow_and_u(flow, u1s[nscales - 1])

    return u1s[nscales - 1]

# Displays the flow and the disparity map.
def disp_flow_and_u(flow, u):

    plt.subplot(1, 2, 1)
    plt.imshow(flow)
    plt.subplot(1, 2, 2)
    plt.imshow(u)
    plt.show()

# Preparation of the indices for the lightfield (needed for the derivative).
import numpy as np
I_lf = np.zeros(shape=(9,9,2))

lf_range = [x for x in reversed(range(-4, 5))]
for i in range(9):
    for j in range(9):
        I_lf[i, j, 0] = lf_range[j]

for i in range(9):
    for j in range(9):
        I_lf[j, i, 1] = lf_range[j]

I_lf_orig = I_lf

# Paths of the models one wants to evaluate.
model_paths = [("../Final/Networks/RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","RES_NOBN_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "RES", False)]

den = 1.

I_lf = I_lf_orig/den

paths = [("../data/backgammon", "backgammon"),
         ("../data/bedroom", "bedroom")]

# Estimation of the disparity for every scene in the Lightfield Analysis Benchmark.
for folder, scene in paths:
    fol = "MV_TVL2_lam_15_median_555555_FINAL"
    Is = import_lf_whole(folder)
    start = time.perf_counter()
    u = mv_tvl2_multiscale_whole(Is[4, 4], Is, np.zeros_like(Is[4, 4]), tau=0.25, lam=15,
                                 theta=0.25,
                                 warps=35, nscales=5)
    fn_dm = "../eval/FINAL/" + fol + "/disp_maps"
    fn_rt = "../eval/FINAL/" + fol + "/runtimes"
    if not os.path.exists(os.path.dirname(fn_dm)):
        os.makedirs("../eval/FINAL/" + fol)
        os.makedirs(fn_dm)
        os.makedirs(fn_rt)

    plt.imsave("../eval/FINAL/" + fol + "/disp_maps/u_" + scene + ".png", u)
    fio.write_pfm(filename="../eval/FINAL/" + fol + "/disp_maps/" + scene + ".pfm", image=u)
    file = open("../eval/FINAL/" + fol + "/runtimes/" + scene + ".txt", 'w')
    file.write(str(time.perf_counter() - start))
    file.flush()
    file.close()
