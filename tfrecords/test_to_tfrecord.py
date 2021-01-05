#!/usr/bin/python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import tensorflow as tf
import glob
import cv2

from dataset.face_detector import FaceDetector, findRelevantFace
from image_cut_tool import enclosing_square, add_margin, cut

import matplotlib.pyplot as plt


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
def image_example(image, path, height, width):
    feature = {
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'path': _bytes_feature(path.encode('utf-8')),
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


def extract_face(image_path):
    global discarded, non_cut
    # Read the image from file
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Perform face detection and cut
    if image is not None:
        face_roi = detect_face_caffe(image)
        if face_roi is None:
            print("WARNING! Using entire image, no face detected {}".format(image_path))
            face_roi = entire_image_roi(image)
            non_cut += 1
        # Cut the image according to roi
        image = cut(image, face_roi)
    else:
        print("WARNING! Unable to read %s" % image_path)
        discarded += 1
        return None
    return image

def create_tfrecord():
    # Get CSV file to read the ages
    test_tfrecord_file = "D:/tfrecord_test/test.tfrecord"
    images_folder = "C:/Users/Adria/OneDrive/Desktop/test_images/test"
    images_path = glob.glob(images_folder+"/*/*.jpg")
    # Write the dataset in the file

    with tf.io.TFRecordWriter(test_tfrecord_file) as writer:

        for image_path in tqdm(images_path):
            # Extract the face within a rectangle

            image = extract_face(image_path)
            image_path = image_path.replace ( "\\", "/" )
            image_path = image_path.replace("C:/Users/Adria/OneDrive/Desktop/test_images/test/","")
            # Read image and save it in the tfrecord file
            tf_example = image_example(image, image_path , image.shape[0], image.shape[1])
            writer.write(tf_example.SerializeToString())

    print("Non cropped images: "+ str(non_cut))
    print("Discarded_images: " + str(discarded))


def main():
    create_tfrecord()



if __name__ == '__main__':
    main()



