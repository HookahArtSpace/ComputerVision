import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo (or mean albedo for RGB input images) for a pixel is less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights. All N images have the same dimension.
                  The images can either be all RGB (height x width x 3), or all grayscale (height x width x 1).

    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    rows = images[0].shape[0]
    cols = images[0].shape[1]
    channels = images[0].shape[2]
    albedo = np.zeros((rows, cols, channels))
    normals = np.zeros((rows, cols, 3))
    lights_t = lights.T
    matrix = np.linalg.inv(lights_t@lights)@lights_t

    img_updated = np.array(images).reshape(len(images), rows*cols*channels)
    matrix_G = matrix @  img_updated
    matrix_G = matrix_G.T.reshape(rows, cols,channels, 3)
    albedo = np.linalg.norm(matrix_G, axis = -1)
    albedo = np.where(albedo < 1e-7, 0, albedo)
    mean_G = np.mean( matrix_G, axis = 2)
    kd = np.linalg.norm(mean_G, axis = -1)
    kd = np.expand_dims(kd, axis = 2)
    normals = np.where(kd<1e-7, 0, mean_G/kd)

    return albedo, normals



def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    height, width = points.shape[:2]
    new_p = np.zeros((height,width,2))
    p_matrix = K @ Rt
    pro = np.ones([4])

    for i in range(height):
        for j in range(width):
            pro[:3] = points[i][j][:]
            h = p_matrix @ pro
            h = h/h[2]
            new_p[i][j][:] = h[:2]

    return new_p


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc_impl' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc_impl' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    rows = image.shape[0]
    cols = image.shape[1]
    channels = image.shape[2]
    normalized = np.zeros((rows, cols, channels, ncc_size **2))
    for i in range(ncc_size):
        for j in range(ncc_size):
            mid = ncc_size //2
            normalized[mid : rows-mid, mid: cols-mid, :, i*ncc_size+j] = image[i:i+rows-ncc_size+1, j: j+cols-ncc_size+1, :]

    mean = np.mean(normalized, axis= -1)
    mean = mean.reshape((rows,cols,channels,1))
    normalized -= mean
    normalized = normalized.reshape((rows,cols,channels * ncc_size **2))
    denominator = np.linalg.norm(normalized, axis= -1).reshape((rows,cols,1))
    normalized = np.divide(normalized,denominator,where = denominator >= 1e-6)
    return normalized



def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc_impl.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.sum(image1 * image2, axis= -1)

    return ncc
