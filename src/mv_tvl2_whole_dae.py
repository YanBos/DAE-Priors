import skimage.transform
import cv2
import os
import ftc.flow_to_color as color
import matplotlib.pyplot as plt
import scipy.signal, scipy.io, scipy.ndimage
import lightfield.file_io as fio
from skimage import color as skicolor
import tensorflow as tf
import time
from utils import utils as utils
import models.models as models

MAX_ITERS = 5
MAX_ITERS_INNER = 1
VERBOSE = True
WARPED = False

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

def import_lf_whole(path):
    images = fio.read_lightfield(path)
    images_gray = np.zeros(shape=[9, 9, 512, 512])
    for i in range(9):
        for j in range(9):
            images_gray[i, j] = skicolor.rgb2gray(images[i, j])


    return images_gray, images[4,4]


def compute_grads_whole(Is):

    Is_grads = np.zeros(shape=[9, 9, 2, Is[0, 0].shape[0], Is[0, 0].shape[1]])

    for i in range(9):
        for j in range(9):
            I_y, I_x = np.gradient(Is[i, j])
            Is_grads[i, j, 0] = I_x
            Is_grads[i, j, 1] = I_y

    return Is_grads


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
            in_d = (I_lf[i, j, 0]*Is_grads_warped[i, j, 0] + I_lf[i, j, 1]*Is_grads_warped[i, j, 1])
            inc = (Is_warped[i, j] - I0 - u*in_d)*in_d
            A += inc
            B += in_d**2

    return B, A


def map_images_whole(Is, Is_grads, u):

    Is_grads_warped = np.zeros_like(Is_grads)
    Is_warped = np.zeros_like(Is)
    height, width = u.shape[0], u.shape[1]
    # coordinates for the fields
    x_and_y = np.array(np.dstack(np.meshgrid(np.arange(Is[0, 0].shape[1]), np.arange(Is[0, 0].shape[0]))), dtype=np.float32)


    for i in range(9):
        for j in range(9):
            disp = np.dstack((I_lf[i, j, 0]*u, I_lf[i, j, 1]*u))
            flow = np.array(x_and_y + disp, dtype=np.float32)
            Is_warped[i, j] = cv2.remap(Is[i, j], flow, None, interpolation=cv2.INTER_CUBIC)
            Is_grads[i, j, 1][0, :] = Is_grads[i, j, 1][height - 1, :] = 0
            Is_grads[i, j, 0][:, 0] = Is_grads[i, j, 0][:, width - 1] = 0
            Is_grads_warped[i, j, 0] = cv2.remap(Is_grads[i, j, 0], flow, None, interpolation=cv2.INTER_CUBIC)
            Is_grads_warped[i, j, 1] = cv2.remap(Is_grads[i, j, 1], flow, None, interpolation=cv2.INTER_CUBIC)
    return Is_warped/1., Is_grads_warped/1.


def disp_warped(I1s_warped):

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(I1s_warped[i])
    plt.show()


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


def mv_tvl2_optical_flow_whole(I0, I0_color, Is, u1, tau, lam, theta, warps, scale):

    # check if the dimensions agree.
    assert(I0.shape == Is[0, 0].shape == u1.shape), "The dimensions of the input arrays do not agree."
    p = scale
    if p % 2== 0:
        p = p + 3

    # computation of the number of pixels
    width = I0.shape[1]
    height = I0.shape[0]
    size = width * height

    # computation of the gradient of I1
    Is_grads = compute_grads_whole(Is)


    # OUTER LOOP
    for i in range(warps):

        # compute the derivatives for the first step
        B, A = derivative_whole(Is, Is_grads, I0, u1)

        # iterator for the inner loop of optimization
        n = 0
        error = float("inf")
        # INNER LOOP - Optimization
        while error > (1e-2)**2 and n < MAX_ITERS:
            n += 1

            v1 = (u1 - lam*theta[0]/80*A)/(1 + lam*theta[0]/80*B)

            n_inner = 0
            while error > (1e-2)**2 and n_inner < MAX_ITERS_INNER:
                n_inner += 1

                # before the step
                u1_b = u1

                # after the step
                inp = v1.reshape((1, height, width, 1))*2**scale
                u1 = dae([inp + tf.random.normal(stddev=theta[1]*utils.DS_RNG, shape=(1,height, width, 1)), tf.constant(I0_color, shape=(1, height, width,3), dtype=tf.float32)]).numpy().reshape((height, width))/(2**scale)

                # calculate the error
                error = np.sum(np.square(u1-u1_b))

                error /= size

            if VERBOSE:
                print("     Inner loop cycles: {}".format(n_inner))
            # Filter u1 and u2 to smooth out the field.
            u1 = scipy.signal.medfilt2d(u1, (median[scale],median[scale]))
        if VERBOSE:
            print("Outer loop cycles: {}".format(n))

    return u1


def divergence(p1, p2):

    p1x = scipy.signal.convolve2d(p1, [[0, 1, -1]], 'same')
    p2y = scipy.signal.convolve2d(p2, [[0],[1],[-1]], 'same')
    return p1x + p2y


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


def mv_tvl2_optical_flow_multiscale_whole(I0, I0_color, Is, u1, tau, lam, theta, warps, nscales=-1, zfactor=2):

    Is_scaled = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for x in range(len(Is_scaled)):
        Is_scaled[x] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    I0, Is = normalize_whole(I0, Is)

    I0s = skimage.transform.pyramid_gaussian(I0/255, max_layer=nscales, order=3, sigma=0.8)
    I0s_color = skimage.transform.pyramid_gaussian(I0_color / 255, max_layer=nscales, order=3, sigma=0.8, multichannel=True)

    for i in range(9):
        for j in range(9):
            # bring the crosshair images in the right order and range.
            I1_scaled = skimage.transform.pyramid_gaussian(Is[i, j]/255, max_layer=nscales, order=3, sigma=0.8)
            I1_scaled = reversed([I for I in I1_scaled])
            I1_scaled = [np.array(I*255, dtype=np.uint8) for I in I1_scaled]
            Is_scaled[i][j] = I1_scaled

    # bring the central images in the right order and range.
    I0s = reversed([I for I in I0s])
    I0s = [np.array(I*255, dtype=np.uint8) for I in I0s]
    I0s_color = reversed([I for I in I0s_color])
    I0s_color = [np.float32(I)[:,:,:3] for I in I0s_color]

    # not yet computed
    u1s = skimage.transform.pyramid_gaussian(u1, max_layer=nscales)

    # not yet computed
    u1s = reversed([u1 for u1 in u1s])
    u1s = [u1 for u1 in u1s]

    # compute the number of scales
    nscales = len(I0s)

    Is_scaled = np.array(Is_scaled)

    for i in range(nscales):
        u1s[i] = mv_tvl2_optical_flow_whole(I0s[i], I0s_color[i], Is_scaled[:, :, i], u1s[i], tau, lam, theta, warps, nscales - 1 - i)

        flow = color.flow_to_color(u1s[i]/den, np.zeros_like(u1s[i]))
        if VERBOSE:
            disp_flow_and_u(flow, u1s[i]/den)

        if i != nscales - 1:
            u1s[i+1] = cv2.resize(u1s[i], (u1s[i+1].shape[1], u1s[i+1].shape[0]), interpolation=cv2.INTER_CUBIC)*zfactor
    flow = color.flow_to_color(u1s[nscales - 1]/den, np.zeros_like(u1s[i]))
    #disp_flow_and_u(flow, u1s[nscales - 1]/den)

    return u1s[nscales - 1]/den


def disp_flow_and_u(flow, u):

    plt.subplot(1, 2, 1)
    plt.imshow(flow)
    plt.subplot(1, 2, 2)
    plt.imshow(u)
    plt.colorbar()
    plt.show()




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

paths = [("/mnt/data/benchmark/stratified/backgammon", "backgammon"),
         ("/mnt/data/benchmark/training/dino", "dino"),
         ("/mnt/data/benchmark/stratified/stripes", "stripes"),
         ("/mnt/data/benchmark/stratified/dots", "dots"),
         ("/mnt/data/benchmark/stratified/pyramids", "pyramids"),
         ("/mnt/data/benchmark/test/bedroom", "bedroom"),
         ("/mnt/data/benchmark/test/bicycle", "bicycle"),
         ("/mnt/data/benchmark/test/herbs", "herbs"),
         ("/mnt/data/benchmark/test/origami", "origami"),
         ("/mnt/data/benchmark/training/boxes", "boxes"),
         ("/mnt/data/benchmark/training/cotton", "cotton"),
         ("/mnt/data/benchmark/training/sideboard", "sideboard")]
median = [3,3,3,3,3,3]
den = 1.

model_paths = [("/mnt/data/models/FINAL_V2/GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu/weights.h5","GUIDE_NOBN_RSZ_NSZ-5_PSZ-64_BSZ-8_FILTERS-64_NL-0.0-0.2_ACT-relu", "GI", False),
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


for folder, scene in paths:

    fol = "GUIDE_NOBN_0.0-0.2_lam_15_median_333333_noise_0_05_FINAL"
    Is, c = import_lf_whole(folder)
    start = time.perf_counter()
    fn_dm = "/mnt/data/eval/FINAL/" + fol + "/disp_maps"
    fn_rt = "/mnt/data/eval/FINAL/" + fol + "/runtimes"
    if not os.path.exists(os.path.dirname(fn_dm)):
        print('Here!')
        os.makedirs("/mnt/data/eval/FINAL/" + fol)
        os.makedirs(fn_dm)
        os.makedirs(fn_rt)

    u = mv_tvl2_optical_flow_multiscale_whole(Is[4, 4], c, Is, np.zeros_like(Is[4, 4]), tau=0.25, lam=15,
                                              theta=(0.25, 0.5),
                                              warps=35, nscales=5)
    plt.imsave("/mnt/data/eval/FINAL/" + fol + "/disp_maps/u_" + scene + ".png", u)
    fio.write_pfm(filename="/mnt/data/eval/FINAL/" + fol + "/disp_maps/" + scene + ".pfm", image=u)
    file = open("/mnt/data/eval/FINAL/" + fol + "/runtimes/" + scene + ".txt", 'w')
    file.write(str(time.perf_counter() - start))
    file.flush()
    file.close()
