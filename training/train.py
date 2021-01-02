import argparse

available_nets = ['resnet50', 'senet50', 'vgg16']
available_normalizations = ['full_normalization', 'vggface2']
available_augmentations = ['default', 'vggface2', 'no']
available_optimizers = ['sgd', 'adam']
available_modes = ['train', 'training', 'test', 'train_inference', 'test_inference']
available_lpf = [0, 1, 2, 3, 5, 7]

parser = argparse.ArgumentParser(description='Common training and evaluation.')
parser.add_argument('--lpf', dest='lpf_size', type=int, choices=available_lpf, default=1,
                    help='size of the lpf filter (1 means no filtering)')
parser.add_argument('--cutout', action='store_true', help='use cutout augmentation')
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--lr', default='0.001', help='Initial learning rate or init:factor:epochs', type=float, dest='lr')
parser.add_argument('--optimizer', default='adam', type=str, choices=available_optimizers)
parser.add_argument('--momentum', action='store_true')
parser.add_argument('--dataset', dest='dataset', type=str, default="vggface2_age",
                    help='dataset to use for the training')
parser.add_argument('--mode', dest='mode', type=str, choices=available_modes, default='train', help='train or test')
parser.add_argument('--epoch', dest='test_epoch', type=int, default=None,
                    help='epoch to be used for testing, mandatory if mode=test')
parser.add_argument('--training-epochs', dest='n_training_epochs', type=int, default=220,
                    help='epoch to be used for training, default 220')
parser.add_argument('--dir', dest='base_path', type=str, default=None,
                    help='directory for reading/writing training data and logs')
parser.add_argument('--batch', dest='batch_size', type=int, default=64, help='batch size.')
parser.add_argument('--ngpus', dest='ngpus', type=int, default=1, help='Number of gpus to use.')
parser.add_argument('--sel_gpu', dest='selected_gpu', type=str, default="0",
                    help="one number or two numbers separated by a hyphen")
parser.add_argument('--net', type=str, default='vgg16', choices=available_nets, help='Network architecture')
parser.add_argument('--resume', type=bool, default=False, help='resume training')
parser.add_argument('--resumepath', type=str, default=None, help='path to the checkpoint file')
parser.add_argument('--pretraining', type=str, default='vggface',
                    help='Pretraining weights, do not set for None, can be vggface or imagenet or a file')
parser.add_argument('--preprocessing', type=str, default='vggface2', choices=available_normalizations)
parser.add_argument('--augmentation', type=str, default='default', choices=available_augmentations)
parser.add_argument('--testweights', type=str, default=None, help='Path to the weights for testing')

args = parser.parse_args()

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import os
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from model_build import senet_model_build, vggface_custom_build, resnet_model_build

## Specify the dataset we are going to use
if args.dataset == 'vggface2_age':
    sys.path.append("../dataset")
    from vgg2_dataset_age import Vgg2DatasetAge as Dataset
else:
    print('unknown dataset %s' % args.dataset)
    exit(1)

# Learning Rate
lr = args.lr.split(':')
initial_learning_rate = float(lr[0])  # 0.002
learning_rate_decay_factor = float(lr[1]) if len(lr) > 1 else 0.5
learning_rate_decay_epochs = int(lr[2]) if len(lr) > 2 else 40

# Epochs to train
n_training_epochs = args.n_training_epochs

# Batch size
batch_size = args.batch_size


def step_decay_schedule(initial_lr, decay_factor, step_size):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return keras.callbacks.LearningRateScheduler(schedule, verbose=1)


# Model building
INPUT_SHAPE = None


def get_model():
    global INPUT_SHAPE
    if args.net.startswith('resnet'):
        print("RESNET Network")
        INPUT_SHAPE = (224, 224, 3)
        return resnet_model_build(INPUT_SHAPE, args.pretraining)
    elif args.net.startswith('senet'):
        print("SENET Network")
        INPUT_SHAPE = (224, 224, 3)
        return senet_model_build(INPUT_SHAPE, args.pretraining)
    else:
        print("VGGFACE Network")
        INPUT_SHAPE = (224, 224, 3)
        return vggface_custom_build(INPUT_SHAPE, args.pretraining, args.net)


# GPU allocation
gpu_to_use = [str(s) for s in args.selected_gpu.split(',') if s.isdigit()]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])

# Get the requested model
model, feature_layer = get_model()

# Print the model to look if everything was made correctly
model.summary()

# Se abbiamo scelto di usare il weight_decay allora viene impostato sui layer corretti
if args.weight_decay:
    weight_decay = args.weight_decay  # 0.0005
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, keras.layers.DepthwiseConv2D) or isinstance(
                layer, keras.layers.Dense):
            layer.add_loss(keras.regularizers.l2(weight_decay)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(keras.regularizers.l2(weight_decay)(layer.bias))

# Scegliamo la funzione di loss e la metrica per valutare l'accuratezza del modello
loss = 'mean_squared_error'
accuracy_metrics = keras.metrics.mean_absolute_error

# Scegliamo l'optimizer
if args.optimizer == 'adam':
    optimizer = keras.optimizers.Adam(lr=args.lr)
else:
    optimizer = keras.optimizers.SGD(momentum=0.9) if args.momentum else 'sgd'

# Compiliamo il modello
model.compile(loss=loss, optimizer=optimizer, metrics=accuracy_metrics)

# Creazione della directory in cui salvare i pesi dell'allenamento corrente e i grafi di TensorBoard
datetime = datetime.today().strftime('%Y%m%d_%H%M%S')  # prendiamo la data e l'ora attuale
dirname = "train_results_of_" + datetime  # Nome della directory fino al cambio del TODO
dirname = os.path.join(args.base_path, dirname)
if not os.path.isdir(dirname):
    os.makedirs(dirname)

# Creazione sotto-directory per i pesi
filepath = os.path.join(dirname, "weights")
if not os.path.isdir(filepath):
    os.makedirs(filepath)  # Questa è la directory in cui viene salvato il file dei pesi
# Nome del file dei pesi
weight_file_name = os.path.join(filepath, "checkpoint.{epoch:02d}.hdf5")

# Creazione sotto-directory per tensorBoard
tensor_board_directory = os.path.join(dirname, "tensorBoard")
if not os.path.isdir(filepath):
    os.makedirs(tensor_board_directory)  # Questa è la directory in cui vengono salvati i file per Tensor Board per noi.

# AUGMENTATION
if args.augmentation == 'vggface2':
    from dataset_tools import VGGFace2Augmentation
    custom_augmentation = VGGFace2Augmentation()
else:
    from dataset_tools import DefaultAugmentation
    custom_augmentation = DefaultAugmentation()

if args.mode.startswith('train'):
    print("TRAINING %s" % dirname)
    dataset_training = Dataset('train', target_shape=INPUT_SHAPE, batch_size=batch_size,
                               preprocessing=args.preprocessing, custom_augmentation=custom_augmentation)
    dataset_validation = Dataset('val', target_shape=INPUT_SHAPE, batch_size=batch_size,
                                 preprocessing=args.preprocessing)

    # Creiamo lo scheduler da dare alla funzione di training
    #lr_sched = step_decay_schedule(initial_lr=initial_learning_rate,decay_factor=learning_rate_decay_factor, step_size=learning_rate_decay_epochs)

    # Specifichiamo la variabile da tenere sotto controllo in fase di allenamento per salvare un CheckPoint
    monitor = 'val_loss'
    checkpoint = keras.callbacks.ModelCheckpoint(weight_file_name, verbose=1, save_best_only=True, monitor=monitor)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=tensor_board_directory, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, tbCallBack]

    if args.resume:
        # Inserire l'epoca e i pesi da cui ripartire
        resume_path = args.resumepath
        initial_epoch = int(resume_path[-7:-5])
        print(f'Resuming from epoch {initial_epoch}')
        model.load_weights(resume_path)
    else:
        initial_epoch = 0

    print(dataset_training.get_size())

    model.fit(dataset_training.get_data(), validation_data=dataset_validation.get_data(),
              validation_steps=dataset_validation.get_number_of_batches(),
              verbose=1, callbacks=callbacks_list, steps_per_epoch=dataset_training.get_number_of_batches(),
              epochs=n_training_epochs, workers=8,
              initial_epoch=initial_epoch)


elif args.mode == 'test':
    # Load the weights
    model.load_weights(args.testweights)

    def evalds(part):
        dataset_test = Dataset(part, target_shape=INPUT_SHAPE, augment=False, preprocessing=args.preprocessing)
        print('Evaluating %s results...' % part)
        result = model.evaluate(dataset_test.get_data(), verbose=1, workers=4)
        print('%s results: loss %.3f - accuracy %.3f' % (part, result[0], result[1]))

        #TODO: To be implemented: Saving .csv


    evalds('test')
    evalds('val')
    evalds('train')
