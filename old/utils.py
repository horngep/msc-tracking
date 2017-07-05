from xception import *
from keras.models import Model
from keras import layers
from keras.layers import merge
from keras.layers import Flatten
from keras.layers import Concatenate


def extract_features(image):

    model = Xception_re3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

    inter_1 = Model(inputs=model.input, outputs=model.get_layer('block4_pool').output) # (batchsize, 19, 19, 728)
    feature_1 = inter_1.predict(image).flatten()

    inter_2 = Model(inputs=model.input, outputs=model.get_layer('block13_sepconv1_act').output)  # (batchsize, 19, 19, 728)
    feature_2 = inter_2.predict(image).flatten()

    inter_3 = Model(inputs=model.input, outputs=model.get_layer('block14_sepconv2_act').output)  # (batchsize, 10, 10, 2048)
    feature_3 = inter_3.predict(image).flatten()

    features = np.append([[feature_1], [feature_2]], [feature_3])

    # TODO: currently, this is at 700k units per image (x2 x batchsize = total unit)
    # this is too large to fit into memory
    # TODO: Apply a convolution layer 1x1xC to each feature to reduce the dimensions
    # Can probably do it here (not xception.py)

    return features



def get_labels(dataset='ALOV300'):



    return
























#
