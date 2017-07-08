from convnetskeras.convnets import *
from convnetskeras.customlayers import *
from convnetskeras.imagenet_tool import *

from keras.layers import Input, Embedding, LSTM, Dense, Flatten
from keras.models import Model
import keras


def goturn(learning_rate=0.001):

    img1 = Input(shape=(3,227,227), name='main_input1')
    img2 = Input(shape=(3,227,227), name='main_input2')

    # AlexNet Current Frame
    alex1 = convnet('alexnet',weights_path="../alexnet_weights.h5", heatmap=False)
    alex1_inter = Model(
                        inputs=alex1.input,
                        outputs=alex1.get_layer('dense_2').output,
                        ) # last hiden layer
    alex1_out = alex1_inter(img1)

    # AlexNet Previous Frame
    alex2 = convnet('alexnet',weights_path="../alexnet_weights.h5", heatmap=False)
    alex2_inter = Model(inputs=alex2.input, outputs=alex2.get_layer('dense_2').output)
    alex2_out = alex2_inter(img2)

    # Concatinating them
    embed = keras.layers.concatenate([alex1_out, alex2_out]) # (None, 4096x2=8192)

    # Fully Connected layer
    x = Dense(2048, activation='relu', kernel_initializer='TruncatedNormal')(embed)
    x = Dense(2048, activation='relu', kernel_initializer='TruncatedNormal')(x)
    x = Dense(2048, activation='relu', kernel_initializer='TruncatedNormal')(x)
    out = Dense(4, name='main_output')(x)

    model = Model([img1, img2], out)

    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    # adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=sgd, loss='mean_absolute_error')

    return model


































#
