
from utils import ann_to_list, parse_gtstring, crop_image, preprocess_crop
from utils import convert_gt, dict_to_bbox
import cv2
import numpy as np
from PIL import Image
import pdb
from goturn import goturn
import os, random
import xmltodict
import random
# from imagenet_utils import *

def batch_generator(batch_size, foldername='train+val'):


    '''
    generate batch for ALOV300++
    input: batch_size
    foldername = {'train', 'val', 'test'}
    output: X, Y
    X (batch_size, 2), each with preprocessed and cropped image pairs [img1, img2]
    Y (batch_size, 4), each with the ground truth values relative to search region between 0 and 10
    '''


    #X = np.zeros([batch_size, 2])
    X1 = list()
    X2 = list()
    Y = np.zeros([batch_size, 4])
    it = 0 # use for iterating

    #for i in range(0, batch_size):
    while True:

        # combining val and trainning data
        if foldername == 'train+val':
            if random.uniform(0, 1) <= 0.25:
                foldername = 'val'
            else:
                foldername = 'train'

        # Randomly select an ann file
        ann_prepend = '/datadrive/ren/data/ann' + foldername
        img_prepend = '/datadrive/ren/data/imagedata++'
        # ann_prepend = '/home/ren/Desktop/data/alov300++/ann/' + foldername
        # img_prepend = '/home/ren/Desktop/data/alov300++/imagedata++'

        ann_path1 = random.choice(os.listdir(ann_prepend)) # to folder level
        ann_prepend1 = os.path.join(ann_prepend, ann_path1)
        ann_path2 = random.choice(os.listdir(ann_prepend1)) # to file level (select .ann)
        ANN_PATH = os.path.join(ann_prepend1, ann_path2)

        #print(ANN_PATH)

        # Read the ann file and randomly select current and next frame
        list_of_gtstring = ann_to_list(ANN_PATH) # ['line1', 'line2',...]
        rand = random.randint(0, len(list_of_gtstring)-2 )
        gtstring_1 = list_of_gtstring[rand]
        gtstring_2 = list_of_gtstring[rand+1]

        frameno_1, topleft_cf, bottomright_cf = parse_gtstring(gtstring_1)
        frameno_2, topleft_gt, bottomright_gt = parse_gtstring(gtstring_2)

        #print(frameno_1, frameno_2)
        #print(topleft_cf, bottomright_cf)
        #print(topleft_gt, bottomright_gt)

        # Get image path for img1 and img2
        img_path = os.path.join(img_prepend, ann_path1)
        img_path = os.path.join(img_path, ann_path2)
        img_path = img_path.replace(".ann", "/")

        IMG_PATH1 = img_path + str(frameno_1).zfill(8) + '.jpg'
        IMG_PATH2 = img_path + str(frameno_2).zfill(8) + '.jpg'

        #print('img1: ', IMG_PATH1)
        #print('img2: ', IMG_PATH2)

        # Current frame
        img1 = cv2.imread(IMG_PATH1)
        img1 = crop_image(img1, topleft_cf, bottomright_cf)
        img1 = preprocess_crop(img1)

        # Next frame
        img2 = cv2.imread(IMG_PATH2)
        img2 = crop_image(img2, topleft_cf, bottomright_cf) # NOTE: crop region is the same as current frame NOT next frame
        # img2_tmp = img2
        img2 = preprocess_crop(img2)

        # Convert ground truth of next frame to be between 0 and 10
        gt = convert_gt(img1.shape, topleft_cf, bottomright_cf, topleft_gt, bottomright_gt)

        # Put into X and Y
        X1.append(img1)
        X2.append(img2)
        Y[it] = gt
        it += 1

        if it >= batch_size:
            it = 0
            X1 = np.asarray(X1) #(batch_size, 1, 3, 227, 227)
            X1 = np.squeeze(X1, axis = 1) #(batch_size, 3, 227, 227)

            X2 = np.asarray(X2) #(batch_size, 1, 3, 227, 227)
            X2 = np.squeeze(X2, axis = 1) #(batch_size, 3, 227, 227)

            yield [X1, X2], Y
            X1 = list()
            X2 = list()

#

def batch_generator_imagenet(batch_size, foldername='train'):
    '''
    generate batch for ImageNet Video
    input: batch_size
    foldername = {'train', 'val', 'test'}
    output: X, Y
    X (batch_size, 2), each with preprocessed and cropped image pairs [img1, img2]
    Y (batch_size, 4), each with the ground truth values relative to search region between 0 and 10
    '''


    X1 = list()
    X2 = list()
    Y = np.zeros([batch_size, 4])
    it = 0 # use for iterating

    while True:
        # Local and Azure
        # prepend = '/home/ren/Desktop/data/imageNet/ILSVRC/'
        prepend = '/datadrive/imagenet/ILSVRC/'

        if foldername == 'train':
            #  TODO: NOT TESTED
            xml_prepend_o = os.path.join(prepend, 'Annotations/VID/train')
            img_prepend_o = os.path.join(prepend, 'Data/VID/train')
            fol = random.choice(os.listdir(xml_prepend_o)) # randomly choose between the 4 folders
            xml_prepend = os.path.join(xml_prepend_o, fol)
            img_prepend = os.path.join(img_prepend_o, fol)
        elif foldername == 'val':
            xml_prepend = os.path.join(prepend, 'Annotations/VID/val')
            img_prepend = os.path.join(prepend, 'Data/VID/val')




        xml_fold = random.choice(os.listdir(xml_prepend)) # to folder level
        xml_path = os.path.join(xml_prepend, xml_fold)
        num_xml = sum(os.path.isfile(os.path.join(xml_path, f)) for f in os.listdir(xml_path))

        fileno_1 = random.randint(0, num_xml-2) # randomly select a frame
        fileno_2 = fileno_1 + 1

        XML_PATH1 = os.path.join(xml_path, str(int(fileno_1)).zfill(6) + '.xml')
        XML_PATH2 = os.path.join(xml_path, str(int(fileno_2)).zfill(6) + '.xml')
        # print(XML_PATH1)

        with open(XML_PATH1) as fd:
            dict1 = xmltodict.parse(fd.read())
            folder1 = dict1['annotation']['folder']
            filename1 = dict1['annotation']['filename']

            if 'object' in  dict1['annotation']:
                num_object = len(dict1['annotation']['object'])
                OBJECT_ID = random.randint(0, num_object - 1)

                topleft_1, bottomright_1 = dict_to_bbox(dict1, OBJECT_ID)
            else:
                continue

        with open(XML_PATH2) as fd:
            dict2 = xmltodict.parse(fd.read())
            folder2 = dict2['annotation']['folder']
            filename2 = dict2['annotation']['filename']



            if 'object' in  dict2['annotation']:
                if len(dict2['annotation']['object']) == len(dict1['annotation']['object']):
                    topleft_2, bottomright_2 = dict_to_bbox(dict2, OBJECT_ID)
                else:
                    continue
            else:
                continue

        if foldername == 'train':
            IMG_PATH1 = img_prepend_o + '/' + folder1 + '/' + filename1 + '.JPEG'
            IMG_PATH2 = img_prepend_o + '/' + folder2 + '/' + filename2 + '.JPEG'
        elif foldername == 'val':
            IMG_PATH1 = img_prepend + '/' + folder1 + '/' + filename1 + '.JPEG'
            IMG_PATH2 = img_prepend + '/' + folder2 + '/' + filename2 + '.JPEG'

        # Current frame
        img1 = cv2.imread(IMG_PATH1)
        img1 = crop_image(img1, topleft_1, bottomright_1)
        img1 = preprocess_crop(img1)

        # Next frame
        img2 = cv2.imread(IMG_PATH2)
        img2 = crop_image(img2, topleft_2, bottomright_2) # NOTE: crop region is the same as current frame NOT next frame
        img2 = preprocess_crop(img2)

        # Convert ground truth of next frame to be between 0 and 10
        gt = convert_gt(img1.shape, topleft_1, bottomright_1, topleft_2, bottomright_2)


        # Put into X and Y
        X1.append(img1)
        X2.append(img2)
        Y[it] = gt
        it += 1

        if it >= batch_size:
            it = 0
            X1 = np.asarray(X1) #(batch_size, 1, 3, 227, 227)
            X1 = np.squeeze(X1, axis = 1) #(batch_size, 3, 227, 227)

            X2 = np.asarray(X2) #(batch_size, 1, 3, 227, 227)
            X2 = np.squeeze(X2, axis = 1) #(batch_size, 3, 227, 227)

            yield [X1, X2], Y
            X1 = list()
            X2 = list()


        # return

if __name__ == "__main__":

    batch_generator_imagenet(8, 'train')
    # batch_generator(8,'train')






















#
