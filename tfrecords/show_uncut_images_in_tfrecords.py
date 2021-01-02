import cv2
from glob import glob
import numpy as np

def main(folder):
    path = '/home/simone/Desktop/Training tfrecords/' + str(folder) + '/train.*.txt'
    filename = glob(path)
    print(filename)
    with open(filename[0]) as f:
        image_paths = f.readlines()
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        for image_path in image_paths:
            image = cv2.imread(image_path.replace('\n', ''))
            cv2.imshow('image', image)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main(3)

