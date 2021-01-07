import random
import pandas as pd

# Desired dataset size
DESIRED_SIZE = 100000

CSV_SOURCE_FILE = "train.age_detected.csv"

CSV_TRAIN_FILE_1 = '/media/simone/Dati/ARTIFICIAL VISION/Training tfrecords/1/train.reduced(random)_1.csv'
CSV_TRAIN_FILE_2 = '/media/simone/Dati/ARTIFICIAL VISION/Training tfrecords/2/train.reduced(random)_2.csv'
CSV_TRAIN_FILE_3 = '/media/simone/Dati/ARTIFICIAL VISION/Training tfrecords/3/train.reduced(random)_3.csv'
CSV_TRAIN_FILE_4 = '/media/simone/Dati/ARTIFICIAL VISION/Training tfrecords/4/train.reduced(random)_4.csv'
CSV_TRAIN_FILE_5 = '/media/simone/Dati/ARTIFICIAL VISION/Training tfrecords/5/train.reduced(random)_5.csv'
CSV_TRAIN_FILE_6 = '/media/simone/Dati/ARTIFICIAL VISION/Training tfrecords/6/train.reduced(random)_6.csv'
CSV_TRAIN_FILE_7 = '/media/simone/Dati/ARTIFICIAL VISION/Training tfrecords/7/train.reduced(random)_7.csv'
CSV_TRAIN_FILE_8 = '/media/simone/Dati/ARTIFICIAL VISION/Training tfrecords/8/train.reduced(random)_8.csv'
CSV_TRAIN_FILE_9 = '/media/simone/Dati/ARTIFICIAL VISION/Training tfrecords/9/train.reduced(random)_9.csv'

CSV_VALID_FILE = '/media/simone/Dati/ARTIFICIAL VISION/Validation tfrecords/1/val.reduced(random)_1.csv'
CSV_VALID_FILE_2 = '/media/simone/Dati/ARTIFICIAL VISION/Validation tfrecords/2/val.reduced(random)_2.csv'


def main():
    random_subset_7()
    random_subset_8()
    random_subset_9()

def random_subset_7():
    print("Inizio lettura csv...")
    data_source = pd.read_csv(CSV_SOURCE_FILE)
    n_source = sum(1 for line in open(CSV_SOURCE_FILE)) - 1
    data_train_1 = pd.read_csv(CSV_TRAIN_FILE_1)
    data_train_2 = pd.read_csv(CSV_TRAIN_FILE_2)
    data_train_3 = pd.read_csv(CSV_TRAIN_FILE_3)
    data_train_4 = pd.read_csv(CSV_TRAIN_FILE_4)
    data_train_5 = pd.read_csv(CSV_TRAIN_FILE_5)
    data_train_6 = pd.read_csv(CSV_TRAIN_FILE_6)
    data_valid = pd.read_csv(CSV_VALID_FILE)

    data_train_1 = data_train_1.percorso.values
    data_train_2 = data_train_2.percorso.values
    data_train_3 = data_train_3.percorso.values
    data_train_4 = data_train_4.percorso.values
    data_train_5 = data_train_5.percorso.values
    data_train_6 = data_train_6.percorso.values
    data_valid = data_valid.percorso.values

    items = 0
    dups = 0

    data_train_7 = pd.DataFrame(columns=['percorso', 'età'])

    while items < DESIRED_SIZE:
        index = random.randint(1, n_source)
        row = data_source.iloc[index]

        if row['percorso'] in data_train_1 or row['percorso'] in data_valid or \
                row['percorso'] in data_train_2 or row['percorso'] in data_train_3 or row['percorso'] in data_train_4 or \
                row['percorso'] in data_train_5 or row['percorso'] in data_train_6 or row[
            'percorso'] in data_train_7.percorso.values:
            dups += 1
        else:
            items += 1
            data_train_7 = data_train_7.append(row, ignore_index=True)
            if items % 5000 == 0:
                print("Stiamo a " + str(items))

    print("Fine creazione csv di validation, salvataggio...")
    print("Duplicati: " + str(dups))

    print("Salvataggio nuovo csv 7 ...")
    data_train_7.to_csv(CSV_TRAIN_FILE_7)


def random_subset_8():
    print("Inizio lettura csv...")
    data_source = pd.read_csv(CSV_SOURCE_FILE)
    n_source = sum(1 for line in open(CSV_SOURCE_FILE)) - 1
    data_train_1 = pd.read_csv(CSV_TRAIN_FILE_1)
    data_train_2 = pd.read_csv(CSV_TRAIN_FILE_2)
    data_train_3 = pd.read_csv(CSV_TRAIN_FILE_3)
    data_train_4 = pd.read_csv(CSV_TRAIN_FILE_4)
    data_train_5 = pd.read_csv(CSV_TRAIN_FILE_5)
    data_train_6 = pd.read_csv(CSV_TRAIN_FILE_6)
    data_train_7 = pd.read_csv(CSV_TRAIN_FILE_7)
    data_valid = pd.read_csv(CSV_VALID_FILE)

    data_train_1 = data_train_1.percorso.values
    data_train_2 = data_train_2.percorso.values
    data_train_3 = data_train_3.percorso.values
    data_train_4 = data_train_4.percorso.values
    data_train_5 = data_train_5.percorso.values
    data_train_6 = data_train_6.percorso.values
    data_train_7 = data_train_7.percorso.values
    data_valid = data_valid.percorso.values

    items = 0
    dups = 0

    data_train_8 = pd.DataFrame(columns=['percorso', 'età'])

    while items < DESIRED_SIZE:
        index = random.randint(1, n_source)
        row = data_source.iloc[index]

        if row['percorso'] in data_train_1 or row['percorso'] in data_valid or \
                row['percorso'] in data_train_2 or row['percorso'] in data_train_3 or row['percorso'] in data_train_4 or \
                row['percorso'] in data_train_5 or row['percorso'] in data_train_6 or row['percorso'] in data_train_7 or row[
            'percorso'] in data_train_8.percorso.values:
            dups += 1
        else:
            items += 1
            data_train_8 = data_train_8.append(row, ignore_index=True)
            if items % 5000 == 0:
                print("Stiamo a " + str(items))

    print("Fine creazione csv di validation, salvataggio...")
    print("Duplicati: " + str(dups))

    print("Salvataggio nuovo csv 8 ...")
    data_train_8.to_csv(CSV_TRAIN_FILE_8)


def random_subset_9():
    print("Inizio lettura csv...")
    data_source = pd.read_csv(CSV_SOURCE_FILE)
    n_source = sum(1 for line in open(CSV_SOURCE_FILE)) - 1
    data_train_1 = pd.read_csv(CSV_TRAIN_FILE_1)
    data_train_2 = pd.read_csv(CSV_TRAIN_FILE_2)
    data_train_3 = pd.read_csv(CSV_TRAIN_FILE_3)
    data_train_4 = pd.read_csv(CSV_TRAIN_FILE_4)
    data_train_5 = pd.read_csv(CSV_TRAIN_FILE_5)
    data_train_6 = pd.read_csv(CSV_TRAIN_FILE_6)
    data_train_7 = pd.read_csv(CSV_TRAIN_FILE_7)
    data_train_8 = pd.read_csv(CSV_TRAIN_FILE_8)
    data_valid = pd.read_csv(CSV_VALID_FILE)

    data_train_1 = data_train_1.percorso.values
    data_train_2 = data_train_2.percorso.values
    data_train_3 = data_train_3.percorso.values
    data_train_4 = data_train_4.percorso.values
    data_train_5 = data_train_5.percorso.values
    data_train_6 = data_train_6.percorso.values
    data_train_7 = data_train_7.percorso.values
    data_train_8 = data_train_8.percorso.values
    data_valid = data_valid.percorso.values

    items = 0
    dups = 0

    data_train_9 = pd.DataFrame(columns=['percorso', 'età'])

    while items < DESIRED_SIZE:
        index = random.randint(1, n_source)
        row = data_source.iloc[index]

        if row['percorso'] in data_train_1 or row['percorso'] in data_valid or \
                row['percorso'] in data_train_2 or row['percorso'] in data_train_3 or row['percorso'] in data_train_4 or \
                row['percorso'] in data_train_5 or row['percorso'] in data_train_6 or row['percorso'] in data_train_7 \
                or row['percorso'] in data_train_8 or row['percorso'] in data_train_9.percorso.values:
            dups += 1
        else:
            items += 1
            data_train_9 = data_train_9.append(row, ignore_index=True)
            if items % 5000 == 0:
                print("Stiamo a " + str(items))

    print("Fine creazione csv di validation, salvataggio...")
    print("Duplicati: " + str(dups))

    print("Salvataggio nuovo csv 9 ...")
    data_train_9.to_csv(CSV_TRAIN_FILE_9)


if __name__ == '__main__':
    main()
