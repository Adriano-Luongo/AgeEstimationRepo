import numpy as np
import time
import random
import cv2
import sys
import os
import keras
from tqdm import tqdm





sys.path.append("../training")
from training.dataset_tools import _readcsv

NUM_CLASSES = 8631 + 500

PARTITION_TRAIN = 0
PARTITION_VAL = 1
PARTITION_TEST = 2

vgg2ids = None
ids2vgg = None


def _load_identities(idmetacsv):
    global vgg2ids
    global ids2vgg
    if ids2vgg is None:
        # Vgg 2 ids mappa l'ID dell'identità / nome cartella in una tupla (nome identità, indice di riga di identiti_meta.csv)
        vgg2ids = {}
        # Lista contenente le tuple (nome identità, ID identità)
        ids2vgg = []
        arr = _readcsv(idmetacsv)
        i = 0
        for line in arr:
            try:
                #Numero della cartella del tipo n0123231 senza n. (Ovvero l'ID della persona)
                vggnum = int(line[0][1:])
                #Usa come chiave il nome della cartella (L'ID) e come valore la tupla (nome, indice di riga)
                vgg2ids[vggnum] = (line[1], i)
                #Aggiunge alla lista la tupla (nome, ID della persona)
                ids2vgg.append((line[1], vggnum))
                i += 1
            except Exception:
                pass
        print(len(ids2vgg), len(vgg2ids), NUM_CLASSES)
        assert (len(ids2vgg) == NUM_CLASSES)
        assert (len(vgg2ids) == NUM_CLASSES)


#Queste due funzioni successive chiamano entrambe load_identities passando il percorso al file identity_meta.csv
# La funzione load_identities scritta sopra. Serve a riempire vgg2ids e ids2vgg per essere usate dopo da queste due


def get_id_from_vgg2(vggidn, idmetacsv='vggface2/identity_meta.csv'):
    # Questa funzione
    _load_identities(idmetacsv)
    try:
        # Dato l'ID della persona ti ritorna la tupla (nome, indice di riga di identity_meta.csv)
        return vgg2ids[vggidn]
    except KeyError:
        print('ERROR: n%d unknown' % vggidn)
        return 'unknown', -1


def get_vgg2_identity(idn, idmetacsv='vggface2/identity_meta.csv'):
    _load_identities(idmetacsv)
    try:
        return ids2vgg[idn]
    except IndexError:
        print('ERROR: %d unknown', idn)
        return 'unknown', -1









