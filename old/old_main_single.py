#from cnn import *
from goturn import *
import keras.preprocessing.image as image
from imagenet_utils import *
from utils import *
import time
import numpy as np
import cv2
from keras import backend as K
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf
import pdb
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # get gt files
    gt_file_path = '/home/ren/Desktop/data/alov300++/alov300++_rectangleAnnotation_full/01-Light/01-Light_video00002.ann'
    list_of_gtstring = ann_to_list(gt_file_path) # ['line1', 'line2',...]


    prepend_path = '/home/ren/Desktop/data/alov300++/imagedata++/01-Light/01-Light_video00002/'

    # Get Current frame
    gtstring_1 = list_of_gtstring[0]
    frameno_1, topleft_cf, bottomright_cf = parse_gtstring(gtstring_1)

    img_path1 = prepend_path + '0000000' + str(frameno_1) + '.jpg'
    img1 = cv2.imread(img_path1)
    original_dim = img1.shape
    img1 = crop_image(img1, topleft_cf, bottomright_cf)
    img1 = cv2.resize(img1, (227, 227)) # (227,227,3)
    img1 = img1.astype(np.float)
    img1 = np.expand_dims(img1, axis=0)
    img1 = preprocess_input(img1)    # this process change RBG->BGR - but image already BGR?
    img1 = np.transpose(img1, (0,3,1,2)) # (1,3,227,227)


    # Get Next frame
    gtstring_2 = list_of_gtstring[1]
    frameno_2, topleft_gt, bottomright_gt = parse_gtstring(gtstring_2)

    img_path2 = prepend_path + '0000000' + str(frameno_2) + '.jpg'
    img2 = cv2.imread(img_path2)
    img2 = crop_image(img2, topleft_cf, bottomright_cf) # NOTE: crop region is the same as current frame NOT next frame
    img2_tmp = img2
    img2 = cv2.resize(img2, (227, 227)) # (227,227,3)
    img2 = img2.astype(np.float)
    img2 = np.expand_dims(img2, axis=0)
    img2 = preprocess_input(img2)    # this process change RBG->BGR - but image already BGR?
    img2 = np.transpose(img2, (0,3,1,2)) # (1,3,227,227)

    # testing alexnet
    # alex1 = convnet('alexnet',weights_path="alexnet_weights.h5", heatmap=False)
    # model = Model(inputs=alex1.input, outputs=alex1.output)
    # model.compile(optimizer='sgd', loss='mse')
    # preds = model.predict(img1)
    # print('Predicted:', decode_predictions(preds, top=5)[0])

    gt = convert_gt(original_dim, topleft_cf, bottomright_cf, topleft_gt, bottomright_gt)

    model = goturn()
    model.fit([img1, img2], gt, epochs=1, batch_size=1)

    preds = model.predict([img1, img2])
    print(preds)

    pdb.set_trace()

    # ops = []
    # for layer in model.layers:
    #    if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
    #       original_w = K.get_value(layer.W)
    #       converted_w = convert_kernel(original_w)
    #       ops.append(tf.assign(layer.W, converted_w).op)
    # K.get_session().run(ops)
    # model.save_weights('my_weights_tensorflow.h5')


    # OLD WAY - BEFORE CROPPING
    # img_path2 = '../cat.jpg'
    # img2 = image.load_img(img_path2, target_size=(227, 227))
    # img2 = image.img_to_array(img2)
    # img2 = np.expand_dims(img2, axis=0)
    # img2 = preprocess_input(img2)
    # img2 = np.transpose(img2, (0,3,1,2)) # (1,3,227,227)





    # embed1 = extract_features(x1)
    # embed2 = extract_features(x2)
    # embed = np.append([embed1], [embed2])
    #
    # print(len(embed))

    #fully_connected(embed, y)






# input shape (None, 299, 299, 3)
