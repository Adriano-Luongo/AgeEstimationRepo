import random
import cv2
import numpy as np
import tensorflow as tf

############ FIT PLANE ##########
tmp_A = []
FIT_PLANE_SIZ = 16
for y in np.linspace(0, 1, FIT_PLANE_SIZ):
    for x in np.linspace(0, 1, FIT_PLANE_SIZ):
        tmp_A.append([y, x, 1])
Amatrix = np.matrix(tmp_A)


def _fit_plane(im):
    original_shape = im.shape
    if len(im.shape) > 2 and im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (FIT_PLANE_SIZ, FIT_PLANE_SIZ))
    if im.dtype == np.uint8:
        im = im.astype(float)
    # do fit
    A = Amatrix
    tmp_b = []
    for y in range(FIT_PLANE_SIZ):
        for x in range(FIT_PLANE_SIZ):
            tmp_b.append(im[y, x])
    b = np.matrix(tmp_b).T
    fit = (A.T * A).I * A.T * b

    fit[0] /= original_shape[0]
    fit[1] /= original_shape[1]


def linear_balance_illumination(im):
    if im.dtype == np.uint8:
        im = im.astype(float)
    if len(im.shape) == 2:
        im = np.expand_dims(im, 2)
    if im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    imout = im.copy()
    imest = _fit_plane(im[:, :, 0])
    if imest is not None:
        imout[:, :, 0] = im[:, :, 0] - imest + np.mean(imest)
    if im.shape[2] > 1:
        imout = cv2.cvtColor(imout, cv2.COLOR_YUV2BGR)
    return imout.reshape(im.shape)
############ END FIT PLANE ##########


def mean_std_normalize(inp, means=None, stds=None):
    assert (len(inp.shape) >= 3)
    d = inp.shape[2]
    if means is None and stds is None:
        means = []
        stds = []
        for i in range(d):
            stds.append(np.std(inp[:, :, i]))
            means.append(np.mean(inp[:, :, i]))
            if stds[i] < 0.001:
                stds[i] = 0.001
    outim = np.zeros(inp.shape)
    for i in range(d):
        if stds is not None:
            outim[:, :, i] = (inp[:, :, i] - means[i]) / stds[i]
        else:
            outim[:, :, i] = (inp[:, :, i] - means[i])
    return outim


def _random_normal_crop(n, maxval, positive=False, mean=0):
    gauss = np.random.normal(mean, maxval / 2, (n, 1)).reshape((n,))
    gauss = np.clip(gauss, mean - maxval, mean + maxval)
    if positive:
        return np.abs(gauss)
    else:
        return gauss


def random_brightness_contrast(img):
    # brightness and contrast
    a = _random_normal_crop(1, 0.5, mean=1)[0]
    b = _random_normal_crop(1, 48)[0]
    img = (img - 128.0) * a + 128.0 + b
    img = np.clip(img, 0, 255)
    return img


def random_saturation(image, label):
    image = tf.image.random_saturation(image, 3.1, 8.4, seed=random.randint(1, 60))
    return (image, label)


def random_flip(img):
    # flip
    if random.randint(0, 1):
        img = np.fliplr(img)
    return img


def random_monochrome(x, random_fraction_yes=0.2):
    if random.random() < random_fraction_yes:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        if len(x.shape) == 2:
            x = x[:, :, np.newaxis]
        x = np.repeat(x, 3, axis=2)
    return x


def random_image_rotate(img, rotation_center):
    angle_deg = _random_normal_crop(1, 10)[0]
    M = cv2.getRotationMatrix2D(rotation_center, angle_deg, 1.0)
    nimg = cv2.warpAffine(img, M, dsize=img.shape[0:2], borderMode=cv2.BORDER_REPLICATE)
    if len(nimg.shape) < 3:
        nimg = nimg[:, :, np.newaxis]
    return nimg


def random_image_skew(img):
    s = _random_normal_crop(2, 0.1, positive=True)
    M = np.array([[1, s[0], 1], [s[1], 1, 1]])
    nimg = cv2.warpAffine(img, M, dsize=img.shape[0:2], borderMode=cv2.BORDER_REPLICATE)
    if len(nimg.shape) < 3:
        nimg = nimg[:, :, np.newaxis]
    return nimg


class VGGFace2Augmentation:
    def augment(self, img):
        img = random_monochrome(img, random_fraction_yes=0.2)
        img = random_flip(img)
        return img


class DefaultAugmentation:
    def augment(self, img):
        img = random_image_rotate(img, (124, 124))
        img = random_image_skew(img)
        img = random_brightness_contrast(img)
        img = random_flip(img)
        return img


def equalize_hist(img):
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = np.array(img)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(img)


def tf_equalize_histogram(image):
    values_range = tf.constant([0., 255.], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(tf.to_float(image), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.to_float(cdf - cdf_min) * 255. / tf.to_float(pix_cnt - 1))
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return eq_hist
