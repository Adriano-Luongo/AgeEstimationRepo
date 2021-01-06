#!/usr/bin/python3

from tqdm import tqdm
import os
import tensorflow as tf
import sys
import numpy as np
import cv2
from functools import partial
import glob
from math import floor
import matplotlib.pyplot as plt
import tensorflow_io as tfio

sys.path.append("../training")
from dataset_tools import equalize_hist, linear_balance_illumination, mean_std_normalize, DefaultAugmentation

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATA_DIR = "data"

VGGFACE2_MEANS = np.array([131.0912, 103.8827, 91.4953])  # RGB

#This function is called after reading the Tfrecord file. Parses the binary record into a predifined structure
def read_tfrecord(record, labeled):

    tfrecord_format = (
        {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.float32),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
        if labeled
        #In case we are testing, the tfrecord file does not contain any label,
        # it actually contains the path to the iamge
        else {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'path': tf.io.FixedLenFeature([], tf.string),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
    )

    # First we parse the record into the structure above
    parsed_record = tf.io.parse_single_example(record, tfrecord_format)

    # We extract the raw image from the parsed record into a uint8 format
    image = tf.io.decode_raw(parsed_record['image_raw'], tf.uint8)

    # Pick the height and width parameters to reshape the image as its previous shape
    width = parsed_record['width']
    height = parsed_record['height']

    #Reshape of the image
    image = tf.reshape(image, (height, width, 3))

    # Return (image,label) or (image, path) depending on the fact that is trianing,validation or test
    if labeled:
        label = parsed_record['label']
        return image, label
    else:
        path = parsed_record['path']
        return image, path


def load_dataset(filenames, labeled=True):

    # Load the dataset from filenames
    dataset = tf.data.TFRecordDataset(filenames)

    if labeled:
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        # automatically interleaves reads from multiple files
        dataset = dataset.with_options(ignore_order)
        # uses data as soon as it streams in, rather than in its original order

    # Calls the read_tfrecord function on every element in dataset in an efficient way
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or (images,path) if labeled=False
    return dataset


def _load_vgg2(tfr_dir, partition, labeled):

    # Depending on the partition we are selecting we glob every file inside the partition folder
    tfr_dir = tfr_dir.replace("\\", "/")
    tfrecords_files = glob.glob(os.path.join(tfr_dir, f"{partition}.reduced(random)_*.tfrecord"))

    #We check if we have to handle validation/training data or test data to parse it properly
    if not labeled:
        return load_dataset(tfrecords_files, False)
    else:
        return load_dataset(tfrecords_files)


class Vgg2DatasetAge:

    def get_batch_size(self):
        return self.batch_size

    def get_number_of_batches(self):
        return floor(self.size / self.batch_size)

    def get_size(self):
        return self.size

    def get_data(self):


        def _apply_custom_preprocessing(tensor_image):

            image = np.asarray(tensor_image)
            # VGGFACE2 dataset mean and deviation in BGR format
            ds_means = np.array([91.4953, 103.8827, 131.0912])
            ds_stds = None

            image = np.asarray(image)
            image = mean_std_normalize(image, ds_means, ds_stds)

            return image

        def _apply_custom_augmentation(tensor_image):

            custom_augmentation = self.custom_augmentation
            if custom_augmentation is None:
                augmentation = DefaultAugmentation()
            else:
                augmentation = custom_augmentation

            def augment(image):
                image = np.asarray(image)
                image = augmentation.augment(image)
                return image

            return augment(tensor_image)

        #Wrapper functions to handle tensors as eagertensors
        def random_augmentation(image):
            im_shape = image.shape
            [image, ] = tf.py_function(_apply_custom_augmentation, [image], [tf.uint8])
            image.set_shape(im_shape)
            return image

        def data_preprocessing(image):
            im_shape = image.shape
            [image, ] = tf.py_function(_apply_custom_preprocessing, [image], [tf.int16])
            image.set_shape(im_shape)
            return image

        def resize_images(image):
            image = tf.cast(tf.image.resize(image, self.target_shape), tf.int16)
            return image

        # Labeled and augment only for the training set
        if self.labeled and self.augment:
            # Training set
            print("Training set preprocessing layer")
            process_images = tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: tfio.experimental.color.rgb_to_bgr(x)),
                tf.keras.layers.Lambda(lambda x: random_augmentation(x)),
                tf.keras.layers.Lambda(lambda x: data_preprocessing(x)),
                tf.keras.layers.Lambda(lambda x: resize_images(x))
            ])
            self.data = self.data.map(lambda x, y: (process_images(x), y),
                                      num_parallel_calls=AUTOTUNE)
        elif self.labeled:
            # Validation set
            print("Validation set preprocessing layer")
            process_images = tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: tfio.experimental.color.rgb_to_bgr(x)),
                tf.keras.layers.Lambda(lambda x: data_preprocessing(x)),
                tf.keras.layers.Lambda(lambda x: resize_images(x))
            ])
            self.data = self.data.map(lambda x, y: (process_images(x), y),
                                      num_parallel_calls=AUTOTUNE)
        else:
            # Test set
            print("Test set preprocessing layer")
            process_images = tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: tfio.experimental.color.rgb_to_bgr(x)),
                tf.keras.layers.Lambda(lambda x: data_preprocessing(x)),
                tf.keras.layers.Lambda(lambda x: resize_images(x))
            ])
            self.data = self.data.map(lambda x, y: (process_images(x), y),
                                      num_parallel_calls=AUTOTUNE)

        if self.labeled:
            # Rendiamo il dataset ripetuto
            self.data = self.data.repeat()
            # Shuffle the data
            self.data = self.data.shuffle(2048, reshuffle_each_iteration=True)

        # Divide data into batches of the desired size
        self.data = self.data.batch(batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        #As specified in documentation the input pipeline should end with a prefetch operation, this serves to
        # prefetch batches before actually required increasing the speed of all the operations that require getting
        # batches of data.
        self.data = self.data.prefetch(buffer_size=AUTOTUNE)

        return self.data

    def __init__(self,
                 partition='train',
                 data_dir='vggface2_data/<part>',
                 target_shape=(224, 224, 3),
                 batch_size=4,
                 custom_augmentation=None,
                 preprocessing='vgg_face'):

        if partition.startswith('train'):
            load_partition = 'train'
            self.augment = True
            self.labeled = True
            self.drop_remainder = True
        elif partition.startswith('val'):
            load_partition = 'val'
            self.augment = False
            self.labeled = True
            self.drop_remainder = True
        elif partition.startswith('test'):
            load_partition = 'test'
            self.augment = False
            self.labeled = False
            self.drop_remainder = False
        else:
            raise Exception("unknown partition")

        self.batch_size = batch_size
        self.target_shape = [target_shape[0], target_shape[1]]
        self.custom_augmentation = custom_augmentation

        self.size = 0
        self.preprocessing = preprocessing


        print('Loading %s data...' % partition)

        # data_root directs to dataset/data
        data_root = os.path.join(EXT_ROOT, DATA_DIR)

        # Convert from vggface2_data/<part> to the specific partition
        data_dir = data_dir.replace('<part>', load_partition)
        tfr_partition_dir = os.path.join(data_root, data_dir)

        # here we load the dataset required, every element is a tuple (image, age) (image,path) depending
        # on the required partition
        self.data = _load_vgg2(tfr_partition_dir, load_partition, self.labeled)

        print(f"Caricamento del dataset da {tfr_partition_dir}")

        # Contiamo il numero di elementi nella partizione
        if load_partition.startswith('train'):
            self.size = 300000
        elif load_partition.startswith('val'):
            self.size = 20000
        elif load_partition.startswith('test'):
           for _ in self.data:
                self.size = self.size + 1

        print(f"Numero totale di immagini: {self.size}")


# Testing
def main():
    dataset_utility = Vgg2DatasetAge('test', target_shape=(224, 224, 3), batch_size=4,
                                     preprocessing='no')
    j = 0
    data = dataset_utility.get_data()

    for image, label in data:
        image = np.array(image)
        plt.xlabel(label.numpy().decode('ascii'))
        plt.imshow(image)
        plt.show()
        if j == 2:
            break
        j += 1


if __name__ == '__main__':
    main()
