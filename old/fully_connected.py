

from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from imagenet_utils import _obtain_input_shape, decode_predictions


'''NOT USING ANYMORE'''

def fully_connected(embed, y):

    x = Input(shape=(len(embed),))
    x = Dense(1024, activation='relu', kernel_initializer='TruncatedNormal')(x)
    x = Dense(1024, activation='relu', kernel_initializer='TruncatedNormal')(x)
    x = Dense(1024, activation='relu', kernel_initializer='TruncatedNormal')(x)
    output = Dense(4)(x)
    model = Model(inputs=x, outputs=output)

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.fit(embed,y)

    return None
