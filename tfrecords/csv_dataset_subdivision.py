import random
import pandas as pd

# Desired dataset size
DESIRED_SIZE = 20000

CSV_SOURCE_FILE = "train.age_detected.csv"

CSV_TRAIN_FILE_1 = '/home/simone/Desktop/Training tfrecords/1/train.reduced(random)_1.csv'
CSV_TRAIN_FILE_2 = '/home/simone/Desktop/Training tfrecords/2/train.reduced(random)_2.csv'
CSV_TRAIN_FILE_3 = '/home/simone/Desktop/Training tfrecords/3/train.reduced(random)_3.csv'
CSV_TRAIN_FILE_4 = '/home/simone/Desktop/Training tfrecords/4/train.reduced(random)_4.csv'
CSV_TRAIN_FILE_5 = '/home/simone/Desktop/Training tfrecords/5/train.reduced(random)_5.csv'
CSV_TRAIN_FILE_6 = '/home/simone/Desktop/Training tfrecords/6/train.reduced(random)_6.csv'


CSV_VALID_FILE = '/home/simone/Desktop/Validation tfrecords/val.reduced(random)_1.csv'

CSV_VALID_FILE_2 = '/home/simone/Desktop/Validation tfrecords/val.reduced(random)_2.csv'


def main():
    random_subset()


def random_subset():
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

    data_valid_2 = pd.DataFrame(columns=['row', 'percorso', 'età'])

    while items < DESIRED_SIZE:
        index = random.randint(1, n_source)
        row = data_source.iloc[index]

        if row['percorso'] in data_train_1 or row['percorso'] in data_valid or \
                row['percorso'] in data_train_2 or row['percorso'] in data_train_3 or row['percorso'] in data_train_4 or \
                row['percorso'] in data_train_5 or row['percorso'] in data_train_6 or row['percorso'] in data_valid_2.percorso.values:
            dups += 1
        else:
            items += 1
            data_valid_2 = data_valid_2.append(row, ignore_index=True)
            if items % 5000 == 0:
                print("Stiamo a " + str(items))

    print("Fine creazione csv di validation, salvataggio...")
    print("Duplicati: " + str(dups))

    print("Salvataggio nuovo csv...")
    data_valid_2.to_csv(CSV_VALID_FILE_2)




if __name__ == '__main__':
    main()
