from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import os

DATASET_PATH = '/home/simone/Desktop/dataset/train/'
CSV_LABELS = 'train.age_detected.csv'


def main():
    plot_random_images()


def get_age_from_image(dataset, image_path):
    row = dataset.loc[dataset['percorso'] == image_path]
    print(row['età'].to_numpy()[0])
    return row['età'].to_numpy()[0]


def plot_random_images():
    plt.figure(figsize=(6, 6))
    subplot_index = 1

    # Get CSV file to read the ages
    print("Inizio lettura csv")
    dataset = pd.read_csv(CSV_LABELS)

    for i in range(0, 4):
        # Create a figure with 3x3 subplots
        plt.subplot(2, 2, subplot_index)
        # Get a random image form the DATASET_PATH
        image, name = get_random_image(DATASET_PATH)
        plt.imshow(image)

        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)

        # Retrieve the age
        age = get_age_from_image(dataset, name)
        plt.xlabel(name.replace('.jpg', '') + " - " + "{:.2f}".format(age), size = 10, labelpad=-10)

        # Use the next subplot
        subplot_index += 1

    plt.show()


def get_random_image(path):
    random_folder = random.choice([
        x for x in os.listdir(path)
        if os.path.isdir(os.path.join(path, x))
    ])
    sub_folder = path + random_folder
    random_image = random.choice([
        x for x in os.listdir(sub_folder)
        if os.path.isfile(os.path.join(sub_folder, x))
    ])
    image_path = sub_folder + '/' + random_image

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    name = random_folder+'/'+random_image

    return image, name


if __name__ == "__main__":
    main()
