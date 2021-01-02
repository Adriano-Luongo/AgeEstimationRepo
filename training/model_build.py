import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from keras.regularizers import l2
import sys


def senet_model_build(input_shape=(224, 224, 3), num_classes=1, weights="imagenet"):
    print("Building senet", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras-squeeze-excite-network')

    from keras_squeeze_excite_network.se_resnet import SEResNet
    m1 = SEResNet(weights=weights, input_shape=input_shape, include_top=True, pooling='avg', weight_decay=0)
    features = m1.layers[-2].output
    x = Flatten(name='flatten')(features)
    x = Dense(512, activation='relu', kernel_regularizer=l2(l=0.01), bias_regularizer=l2(l=0.01), name='fc6')(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(l=0.01), bias_regularizer=l2(l=0.01), name='fc7')(x)
    out = Dense(1, activation='linear', name='fc8')(x)
    model = keras.models.Model(m1.input, out)
    for l in model.layers:
        l.trainable = True
    for l in model.layers[:-6]:
        l.trainable = False
    return model, features


def resnet_model_build(input_shape=(224, 224, 3), weights="vggface"):
    print("Building resnet-50 ", input_shape, "- num_classes 1 - weights", weights)

    from keras_vggface.vggface import VGGFace
    # create a vggface2 model

    vgg_model = VGGFace(model='resnet50', weights=weights, input_shape=input_shape, include_top=False)
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    middle = Dense(512, kernel_initializer='normal', activation='relu', name='features')(x)
    out = Dense(1, kernel_initializer='normal', activation='linear', name='age_estiamte')(middle)
    custom_vgg_model = keras.models.Model(vgg_model.input, out)

    # Fine tuning all the net
    for layer in custom_vgg_model.layers:
        layer.trainable = True
    '''
    # Train last 2 layers
    custom_vgg_model.layers[-1].trainable = True
    custom_vgg_model.layers[-2].trainable = True
    '''
    features = custom_vgg_model.layers[-4].output

    return custom_vgg_model, features


def vggface_custom_build(input_shape, weights="vggface2", net="vgg16"):
    sys.path.append('keras_vggface')
    hidden_dim = 4096
    from keras_vggface.vggface import VGGFace
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    features = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(features)
    x = Dense(hidden_dim, activation='relu', kernel_regularizer=l2(l=0.01), bias_regularizer=l2(l=0.01), name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', kernel_regularizer=l2(l=0.01), bias_regularizer=l2(l=0.01), name='fc7')(x)
    out = Dense(1, activation='linear', name='fc8')(x)
    custom_vgg_model = keras.Model(vgg_model.input, out)

    return custom_vgg_model, features
