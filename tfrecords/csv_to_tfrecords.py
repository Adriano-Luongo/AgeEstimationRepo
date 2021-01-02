#!/usr/bin/python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import cv2
import os

from dataset.face_detector import FaceDetector, findRelevantFace
from image_cut_tool import enclosing_square, add_margin, cut


DATASET_DIR = '/home/simone/Desktop/dataset/train'

CSV_TRAIN_FILE_4 = '/home/simone/Desktop/Training tfrecords/4/train.reduced(random)_4.csv'
CSV_TRAIN_FILE_5 = '/home/simone/Desktop/Training tfrecords/5/train.reduced(random)_5.csv'
CSV_TRAIN_FILE_6 = '/home/simone/Desktop/Training tfrecords/6/train.reduced(random)_6.csv'
CSV_TRAIN_FILES = [CSV_TRAIN_FILE_4, CSV_TRAIN_FILE_5, CSV_TRAIN_FILE_6]

TFRECORD_FILE_4 = '/home/simone/Desktop/Training tfrecords/4/train.reduced(random)_4.tfrecord'
TFRECORD_FILE_5 = '/home/simone/Desktop/Training tfrecords/5/train.reduced(random)_5.tfrecord'
TFRECORD_FILE_6 = '/home/simone/Desktop/Training tfrecords/6/train.reduced(random)_6.tfrecord'
TFRECORD_FILES = [TFRECORD_FILE_4, TFRECORD_FILE_5, TFRECORD_FILE_6]

UNCUT_FILE_4_PATH = '/home/simone/Desktop/Training tfrecords/4/train.reduced(random)_4.txt'
UNCUT_FILE_5_PATH = '/home/simone/Desktop/Training tfrecords/5/train.reduced(random)_5.txt'
UNCUT_FILE_6_PATH = '/home/simone/Desktop/Training tfrecords/6/train.reduced(random)_6.txt'
UNCUT_FILE_4 = open(UNCUT_FILE_4_PATH, 'a')
UNCUT_FILE_5 = open(UNCUT_FILE_5_PATH, 'a')
UNCUT_FILE_6 = open(UNCUT_FILE_6_PATH, 'a')
UNCUT_FILES = [UNCUT_FILE_4, UNCUT_FILE_5, UNCUT_FILE_6]

FACE_DETECTOR = None

discarded = 0   # Images that could not be read
non_cut = 0     # Images where no face could be detected


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant.
def image_example(image, label, height, width):
    feature = {
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'label': _float_feature(label),
        'image_raw': _bytes_feature(image.tobytes())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def detect_face_caffe(frame):
    global FACE_DETECTOR
    if FACE_DETECTOR is None:
        FACE_DETECTOR = FaceDetector(min_confidence=0.6)

    # Preleva il volto che si trova più al centro dell'immagine
    face = findRelevantFace(FACE_DETECTOR.detect(frame), frame.shape[1], frame.shape[0])
    if face is None:
        return None
    roi = enclosing_square(face['roi'])
    roi = add_margin(roi, 0.2)
    return roi


def entire_image_roi(img):
    return [0, 0, img.shape[1], img.shape[0]]


def extract_face(image_path, index):
    global discarded, non_cut

    # Read the image from file
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection and cut
    if image is not None:
        face_roi = detect_face_caffe(image)
        if face_roi is None:
            print("WARNING! Using entire image, no face detected {}".format(image_path))
            UNCUT_FILES[index].write(image_path+'\n')
            face_roi = entire_image_roi(image)
            non_cut += 1

        # Cut the image according to roi
        image = cut(image, face_roi)
    else:
        print("WARNING! Unable to read %s" % image_path)
        discarded += 1
        return None
    return image


# Decoding function
def parse_record(record):
    name_to_features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.float32),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    return tf.io.parse_single_example(record, name_to_features)


def decode_record(record):
    image = tf.io.decode_raw(
        record['image_raw'], tf.uint8
    )
    # print(f"Il numero totale degli elementi dell'immagine decodificata è {len(image)}")
    width = record['width']
    height = record['height']
    label = record['label']
    # print(f"La label di questa immagine è {label}")
    # print ( f"L'altezza dell'immagine è {height} mentre la larghezza è {width}" )
    image = tf.reshape(image, (height, width, 3))
    return image, label


def create_tfrecord(index):
    # Get CSV file to read the ages
    print("Inizio lettura csv " + CSV_TRAIN_FILES[index])
    dataset_csv = pd.read_csv(CSV_TRAIN_FILES[index])

    # Write the dataset in the file
    with tf.io.TFRecordWriter(TFRECORD_FILES[index]) as writer:

        for _, row in tqdm(dataset_csv.iterrows()):
            # Get the image path from the csv file and make it an absolute path to the dataset.
            image_path = os.path.join(DATASET_DIR, row['percorso'])
            age = float(row['età'])

            # Extract the face within a rectangle
            image = extract_face(image_path, index)

            # Read image and save it in the tfrecord file
            tf_example = image_example(image, age, image.shape[0], image.shape[1])
            writer.write(tf_example.SerializeToString())


    print("Non cropped images: "+ str(non_cut))
    print("Discarded_images: " + str(discarded))

    UNCUT_FILES[index].write("Non cropped images: "+ str(non_cut) + '\n')
    UNCUT_FILES[index].write("Discarded_images: " + str(discarded) + '\n')
    UNCUT_FILES[index].close()


def main():
    create_tfrecord(0)
    create_tfrecord(1)
    create_tfrecord(2)



'''    # Read the file
    dataset = tf.data.TFRecordDataset(TFRECORD_FILE)

    for record in dataset:
        parsed_record = parse_record(record)
        decoded_record = decode_record(parsed_record)
        image, label = decoded_record

        plt.imshow(np.array(image))
        plt.title(np.array(label))

        plt.show()'''


if __name__ == '__main__':
    main()



