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


def read_tfrecord(record, labeled):
    # Per ogni record all'interno del dataset eseguiamo questa funzione
    tfrecord_format = (
        {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.float32),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
        if labeled
        else {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
    )
    # Prima parsiamo il record secondo la struttura usata per convertirlo inizialemente
    parsed_record = tf.io.parse_single_example(record, tfrecord_format)

    # Una volta parsato procediamo con prelevare i singoli campi
    # Iniziamo con il prelevare l'immagine in byte e convertirla in un immagine formato uint8
    image = tf.io.decode_raw(
        parsed_record['image_raw'], tf.uint8
    )
    # print ( f"Il numero totale degli elementi dell'immagine decodificata è {len ( image )}" )

    # Preleviamo altezza e larghezza per effettuare il resize dell'immagine
    width = parsed_record['width']
    height = parsed_record['height']

    # Prese altezza e larghezza rimodelliamo l'immagine
    image = tf.reshape(image, (height, width, 3))

    # Ritorniamo le informazioni necessarie
    if labeled:
        label = parsed_record['label']
        return image, label
    else:
        return image


def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed

    # Load the dataset from filenames
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files

    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order

    # Converte il dataset da un formato grezzo a coppie (immagine, età) per poi essere dato al model.fit
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )

    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def _load_vgg2(tfr_dir, partition, labeled):
    # Prendiamo tutti i file tfrecords e li posizioniamo in una lista
    tfr_dir = tfr_dir.replace("\\", "/")
    tfrecords_files = glob.glob(os.path.join(tfr_dir, f"{partition}.reduced(random)_*.tfrecord"))

    # Creiamo l'oggetto dataset a partire dai file(s) tfrecord
    # Prendiamo la parte di dataset che è stata richiesta
    if not labeled:
        return load_dataset(tfrecords_files, False)
    else:
        return load_dataset(tfrecords_files)


class Vgg2DatasetAge:

    def get_batch_size(self):
        return self.batch_size

    def get_number_of_batches(self):
        # print(f"Il numero di elementi è {str(self.size)} mentre la batch size è {str(self.batch_size)} quindi ho {str(floor(self.size/self.batch_size))}")
        return floor(self.size / self.batch_size)

    def get_size(self):
        return self.size

    def get_data(self):
        # Shuffle del dataset

        def _apply_custom_preprocessing(tensor_image):
            image = np.asarray(tensor_image)
            ds_means = np.array([91.4953, 103.8827, 131.0912])
            ds_stds = None
            image = np.asarray(image)
            image = mean_std_normalize(image, ds_means, ds_stds)

            return tf.cast(image, tf.uint8)

        def _apply_custom_augmentation(tensor_image):

            custom_augmentation = self.custom_augmentation
            if custom_augmentation is None:
                augmentation = DefaultAugmentation()
            else:
                augmentation = custom_augmentation

            def augment(image):
                image = np.asarray ( image )
                image = augmentation.before_cut(image)
                image = augmentation.after_cut(image)

                return image

            return tf.cast(augment(tensor_image), tf.uint8)

        def random_augmentation(image):
            im_shape = image.shape
            [image, ] = tf.py_function(_apply_custom_augmentation, [image], [tf.uint8])
            image.set_shape(im_shape)
            return image

        def data_preprocessing(image):
            im_shape = image.shape
            [image, ] = tf.py_function(_apply_custom_preprocessing, [image], [tf.uint8])
            image.set_shape(im_shape)
            return image

        def resize_images(image):
            image = tf.cast(tf.image.resize(image, self.target_shape), tf.uint8)
            return image

        # Labeled and augment only for the training set
        if self.labeled and self.augment:
            # Training set
            process_images = tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: tfio.experimental.color.rgb_to_bgr(x)),
                tf.keras.layers.Lambda(lambda x: random_augmentation(x)),
                #tf.keras.layers.Lambda(lambda x: data_preprocessing(x)),
                tf.keras.layers.Lambda(lambda x: resize_images(x))
            ])
            self.data = self.data.map(lambda x, y: (process_images(x), y),
                                      num_parallel_calls=AUTOTUNE)
        elif self.labeled:
            # Validation set
            process_images = tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: tfio.experimental.color.rgb_to_bgr(x)),
                tf.keras.layers.Lambda(lambda x: data_preprocessing(x)),
                tf.keras.layers.Lambda(lambda x: resize_images(x))
            ])
            self.data = self.data.map(lambda x, y: (process_images(x), y),
                                      num_parallel_calls=AUTOTUNE)
        else:
            # Test set
            process_images = tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: tfio.experimental.color.rgb_to_bgr(x)),
                tf.keras.layers.Lambda(lambda x: data_preprocessing(x)),
                tf.keras.layers.Lambda(lambda x: resize_images(x))
            ])
            self.data = self.data.map(lambda x: process_images(x),
                                      num_parallel_calls=AUTOTUNE)

        self.data = self.data.shuffle(2048)

        # Rendiamo il dataset ripetuto
        self.data = self.data.repeat()

        # Dividiamo il dataset in batches
        self.data = self.data.batch(batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        # Come da documentazione la pipeline di trasformazione dovrebbe terminare con prefetch
        # questo permette di caricare in maniera preventiva i dati necessari alla rete nel prossimo step
        # si consiglia un buffersize pari almeno a un batch_size
        # Use buffered prefecting on all datasets
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
        elif partition.startswith('val'):
            load_partition = 'val'
            self.augment = False
            self.labeled = True
        elif partition.startswith('test'):
            load_partition = 'test'
            self.augment = False
            self.labeled = False
        else:
            raise Exception("unknown partition")

        self.batch_size = batch_size
        self.target_shape = [target_shape[0], target_shape[1]]
        self.custom_augmentation = custom_augmentation

        self.size = 0
        self.preprocessing = preprocessing
        # Impostarlo a True per eliminare l'ultimo batch che non ha un numero di samples pari a batch_size
        self.drop_remainder = True
        print('Loading %s data...' % partition)

        # data_root punta a dataset/data
        data_root = os.path.join(EXT_ROOT, DATA_DIR)

        # convertiamo dalla directory generica alla partizione particolare
        # vggface2_data/<part>
        data_dir = data_dir.replace('<part>', load_partition)

        # specifichiamo il path dove si trova il tfrecord relativo alla  partizione
        tfr_partition_dir = os.path.join(data_root, data_dir)

        # Apriamo ogni singolo file presente all'interno della cartella e ne preleviamo il contenuto
        self.data = _load_vgg2(tfr_partition_dir, load_partition, self.labeled)
        # Data è un insieme di coppie (image, label)

        print(f"Caricamento del dataset da {tfr_partition_dir}")

        # Contiamo il numero di elementi nella partizione
        if load_partition.startswith('train'):
            self.size = 600000
        elif load_partition.startswith('val'):
            self.size = 20000
        elif load_partition.startswith('test'):
            for _ in self.data:
                self.size = self.size + 1

        print(f"Numero totale di immagini: {self.size}")


# Testing
def main():
    dataset_utility = Vgg2DatasetAge('train', target_shape=(224, 224, 3), batch_size=4,
                                     preprocessing='full_normalization')
    print(f"Il numero di elementi all'interno del dataset è {dataset_utility.get_size()}")

    # Nel caso di get_data
    j = 0
    data = dataset_utility.get_data()

    for record in data:
        images, ages = record
        for image in images:
            j += 1
            print(j)
            if j > 0 and j < 20:
                print(j)
                image = np.array(image)
                plt.xlabel(cv2.mean(image))
                plt.imshow(image)
                print("minimo: " + str(image.min()))
                plt.show()

    print(j)


if __name__ == '__main__':
    main()
