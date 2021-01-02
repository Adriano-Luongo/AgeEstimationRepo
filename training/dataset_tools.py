import random
import cv2
import numpy as np
import tensorflow as tf
from threading import Lock
from PIL import Image, ImageEnhance,ImageFilter
FIT_PLANE_SIZ=16
############ FIT PLANE ##########
tmp_A = []
FIT_PLANE_SIZ=16
for y in np.linspace(0,1,FIT_PLANE_SIZ):
    for x in np.linspace(0,1,FIT_PLANE_SIZ):
        tmp_A.append([y, x, 1])
Amatrix = np.matrix(tmp_A)

def _fit_plane(im):
    original_shape=im.shape
    if len(im.shape)>2 and im.shape[2]>1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (FIT_PLANE_SIZ,FIT_PLANE_SIZ))
    if im.dtype==np.uint8:
        im = im.astype(float)
    # do fit
    A = Amatrix
    tmp_b = []
    for y in range(FIT_PLANE_SIZ):
        for x in range(FIT_PLANE_SIZ):
            tmp_b.append(im[y,x])
    b = np.matrix(tmp_b).T
    fit = (A.T * A).I * A.T * b

    fit[0]/=original_shape[0]
    fit[1]/=original_shape[1]

def linear_balance_illumination(im):
    if im.dtype==np.uint8:
        im = im.astype(float)
    if len(im.shape)==2:
        im = np.expand_dims(im,2)
    if im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    imout = im.copy()
    imest = _fit_plane(im[:,:,0])
    if imest is not None:
        imout[:,:,0] = im[:,:,0] - imest + np.mean(imest)
    if im.shape[2] > 1:
        imout = cv2.cvtColor(imout, cv2.COLOR_YUV2BGR)
    return imout.reshape(im.shape)

############ END FIT PLANE ##########

def mean_std_normalize(inp, means=None, stds=None):
    assert (len ( inp.shape ) >= 3)
    d = inp.shape[2]
    if means is None and stds is None:
        means = []
        stds = []
        for i in range ( d ):
            stds.append ( np.std ( inp[:, :, i] ) )
            means.append ( np.mean ( inp[:, :, i] ) )
            if stds[i] < 0.001:
                stds[i] = 0.001
    outim = np.zeros ( inp.shape )
    for i in range ( d ):
        if stds is not None:
            outim[:, :, i] = (inp[:, :, i] - means[i]) / stds[i]
        else:
            outim[:, :, i] = (inp[:, :, i] - means[i])
    return outim


def _random_normal_crop(n, maxval, positive=False, mean=0):
    gauss = np.random.normal ( mean, maxval / 2, (n, 1) ).reshape ( (n,) )
    gauss = np.clip ( gauss, mean - maxval, mean + maxval )
    if positive:
        return np.abs ( gauss )
    else:
        return gauss


def random_brightness_contrast(img):
    # brightness and contrast
    a = _random_normal_crop ( 1, 0.5, mean=1 )[0]
    b = _random_normal_crop ( 1, 48 )[0]
    img = (img - 128.0) * a + 128.0 + b
    img = np.clip ( img, 0, 255 )
    img = img.astype ( np.uint8 )
    return img


def random_flip(img):
    # flip
    if random.randint ( 0, 1 ):
        img = np.fliplr ( img )
    return img


def random_monochrome(x, random_fraction_yes=0.2):
    if random.random () < random_fraction_yes:
        x = cv2.cvtColor ( x, cv2.COLOR_BGR2GRAY )
        if len ( x.shape ) == 2:
            x = x[:, :, np.newaxis]
        x = np.repeat ( x, 3, axis=2 )
    return x


def random_saturation(image, label):
    image = tf.image.random_saturation ( image, 3.1, 8.4, seed=random.randint ( 1, 60 ) )
    return (image, label)


def flip_up_down_random(image, label):
    image = tf.image.flip_up_down ( image )
    return (image, label)


def random_image_rotate(img, rotation_center):
    angle_deg = _random_normal_crop ( 1, 10 )[0]
    M = cv2.getRotationMatrix2D ( rotation_center, angle_deg, 1.0 )
    nimg = cv2.warpAffine ( img, M, dsize=img.shape[0:2], borderMode= cv2.BORDER_REFLECT )
    if len ( nimg.shape ) < 3:
        nimg = nimg[:, :, np.newaxis]
    return nimg  # .reshape(img.shape)


def random_image_skew(img):
    s = _random_normal_crop ( 2, 0.1, positive=True )
    M = np.array ( [[1, s[0], 1], [s[1], 1, 1]] )
    nimg = cv2.warpAffine ( img, M, dsize=img.shape[0:2], borderMode= cv2.BORDER_REFLECT )
    if len ( nimg.shape ) < 3:
        nimg = nimg[:, :, np.newaxis]
    return nimg  # .reshape(img.shape)


class VGGFace2Augmentation ():
    def before_cut(self, frame):
        frame = random_monochrome ( frame, random_fraction_yes=0.2 )
        return frame

    def after_cut(self, img):
        img = random_flip ( img )
        return img


class DefaultAugmentation ():
    def before_cut(self, frame):
        frame = random_image_rotate(frame, (124,124))
        frame = random_image_skew(frame)
        return frame

    def after_cut(self, img):
        img = random_brightness_contrast ( img )
        img = random_flip ( img )
        return img


def equalize_hist(img):
    if len ( img.shape ) > 2 and img.shape[2] > 1:
        img = np.array(img)
        img_yuv = cv2.cvtColor ( img, cv2.COLOR_BGR2YUV )
        img_yuv[:, :, 0] = cv2.equalizeHist ( img_yuv[:, :, 0] )
        return cv2.cvtColor ( img_yuv, cv2.COLOR_YUV2BGR )
    else:
        return cv2.equalizeHist ( img )


def tf_equalize_histogram(image):

    values_range = tf.constant([0., 255.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(tf.to_float(image), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.to_float(cdf - cdf_min) * 255. / tf.to_float(pix_cnt - 1))
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return eq_hist

'''
def _apply_custom_preprocessing ( tensor_image , preprocessing ):

    image = np.asarray(tensor_image)

    if preprocessing == 'vggface2':
        ds_means = np.array([91.4953, 103.8827, 131.0912])
        ds_stds = None
    elif preprocessing == "no_normalization":
        ds_means = None
        ds_stds = None
    else:
        ds_means = np.array ( [0.485, 0.456, 0.406] ) * 255
        ds_stds = np.array ( [0.229, 0.224, 0.225] ) * 255

    print(f"Il tipo di preprocessing è {preprocessing}")

    def preprocess( image ):
        image = np.asarray(image)
        if preprocessing == 'full_normalization':
            image= equalize_hist ( image )
            image = image.astype ( np.float32 )
            image = linear_balance_illumination ( image )
            if np.abs ( np.min ( image ) - np.max ( image ) ) < 1:
                print ( "WARNING: Image is =%d" % np.min ( image ) )
            else:
                image = mean_std_normalize ( image )
        elif preprocessing == 'z_normalization':
            image = mean_std_normalize ( image, ds_means, ds_stds )
        elif preprocessing == 'vggface2':
            image = mean_std_normalize ( image, ds_means, ds_stds )

        return image

    return tf.cast(preprocess(image),tf.uint8)

def _apply_custom_augmentation(tensor_image, custom_augmentation):

    image = np.asarray(tensor_image)

    if custom_augmentation is None:
        augmentation = DefaultAugmentation ()
    else:
        augmentation = custom_augmentation

    def augment(image):
        image = np.asarray ( image )
        image = augmentation.before_cut ( image )
        image = augmentation.after_cut ( image )

        return image

    return tf.cast(augment(image),tf.uint8)'''


VGGFACE2_MEANS = np.array([131.0912, 103.8827, 91.4953]) # RGB
#VGGFACE2_l = np.array([91.4953, 103.8827, 131.0912]) # BGR

class DataGenerator ( tf.keras.utils.Sequence ):

    'Generates data for Keras'
    def __init__(self, data, target_shape, data_size_in_batches , preprocessing='full_normalization', batch_size=64,  with_augmentation=True, custom_augmentation=None,
                 drop_reaminder=True):

        if preprocessing not in ['full_normalization', 'z_normalization', 'vggface2', 'no_normalization']:
            raise Exception ( 'unknown preprocessing: %s' % preprocessing )

        self.mutex = Lock ()
        self.data = data
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.on_epoch_end ()
        self.preprocessing = preprocessing
        self.drop_remainder = drop_reaminder
        self.data_size_in_batches = data_size_in_batches

        if preprocessing == 'vggface2':
            self.ds_means = VGGFACE2_MEANS
            self.ds_stds = None
        elif preprocessing == "no_normalization":
            self.ds_means = None
            self.ds_stds = None
        else:
            self.ds_means = np.array ( [0.485, 0.456, 0.406] ) * 255
            self.ds_stds = np.array ( [0.229, 0.224, 0.225] ) * 255

        if with_augmentation and custom_augmentation is None:
            self.augmentation = DefaultAugmentation ()
        else:
            self.augmentation = custom_augmentation



    def __len__(self):
        return self.data_size_in_batches


    def __getitem__(self, index):
        self.mutex.acquire ()
        print(index)
        if self.cur_index >= len ( self ):

            print ( "Reset->unexpected!" )
            # raise StopIteration
            self.on_epoch_end()
        i = self.cur_index
        self.cur_index += 1
        self.mutex.release ()
        batch = self._load_batch()
        return batch

    def on_epoch_end(self):
        self.mutex.acquire ()
        self.cur_index = 0
        print ( 'Shuffle set' )
        self.data = self.data.shuffle(2048)
        self.mutex.release ()

    def _load_batch(self):

        # Funzione necessaria al resize delle immagini
        def resize_images(image, label):
            image = tf.cast ( tf.image.resize ( image, self.target_shape ), tf.uint8 )
            return image, label

        # Effettuiamo il resize delle immagini al target shape dato dall'utente
        data = self.data.map ( resize_images )

        while True:

            # Mischiamo il dataset
            data = data.shuffle ( 2048 )

            # Dividiamo il dataset in batches, aggiungiamo il prefetching e ritorniamo data
            # drop_remainder serve a eliminare l'ultimo batch che è costituito da un numero di elementi minore di batch_size
            data = data.batch ( batch_size=self.batch_size, drop_remainder=self.drop_remainder )

            # Come da documentazione la pipeline di trasformazione dovrebbe terminare con prefetch
            # questo permette di caricare in maniera preventiva i dati necessari alla rete nel prossimo step
            # si consiglia un buffersize pari almeno a un batch_size
            data = data.prefetch ( buffer_size = 2 )

            for batch in data:
                images , labels = batch
                print("STo per ritornare un batch, però ancora devo fare processing")

                #Applichiamo l'augmentation selezionata
                images,labels = _apply_custom_augmentation ( (images,labels) , self.augmentation )
                print("ho applicato l'augmentation")

                # Apply preprocessing
                images,labels = _apply_custom_preprocessing ( (images,labels) , self.preprocessing )
                print ( "ho applicato il preprocessing" )

                yield images,labels
