
import numpy as np
from PIL import Image
import pdb
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
from imagenet_utils import preprocess_input
import os
#%matplotlib inline



def plot_boundingbox(img_path, topleft, bottomright, vidname):

    '''
    show the boudning box in the image
    '''

    width = bottomright[0]-topleft[0]
    height = bottomright[1]-topleft[1]
    #print('>>> Bounding box location - topleft:', topleft, ' width:', width,' height:', height)

    im = np.array(Image.open(img_path), dtype=np.uint8)
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    rect = patches.Rectangle(topleft,width,height,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    #plt.show()
    plt.savefig('../visualise_results/' + vidname + '.png')

    return None


def ann_to_list(_gt_file_path):
    ''' extract .ann file, turn into list'''

    _f = open(_gt_file_path, "r")
    _f = _f.read()
    _f = _f.split('\n')
    _list_of_gtstring = _f[:-1]

    return _list_of_gtstring

def parse_gtstring(_gtstring):
    ''' parse framno, topleft coordinates and bottomeright coordinates from each line of string '''

    _gtstring = _gtstring.split(' ')
    #print(_gtstring)
    _gtstring = [round(float(i)) for i in _gtstring] # convert to int
    _frameno = _gtstring[0]
    #print(_gtstring)

    # OLD
    #_topleft = _gtstring[3:5]
    #_bottomright = _gtstring[7:9]

    #print('original: ' ,_gtstring[3:5], _gtstring[7:9])


    # sometimes the coordinates are in wrong order, instead do:
    x_axis = list( _gtstring[i] for i in [1,3,5,7] )
    y_axis = list( _gtstring[i] for i in [2,4,6,8] )
    _topleft = [min(x_axis), min(y_axis)]
    _bottomright = [max(x_axis), max(y_axis)]
    #print('new: ' ,_topleft, _bottomright)


    return _frameno, _topleft, _bottomright


def crop_image(img, topleft, bottomright):
    '''
    pad original whole image relative to the size of the bounding box. Then
    crop the image with 2x the original bounding box size centered at the gt of
    current frame
    '''

    original_dim = img.shape # (ori_height, ori_width, 3)
    #print('original image dimension: ', original_dim)

    # bbox width and height
    width = (bottomright[0]-topleft[0])
    height = (bottomright[1]-topleft[1])

    # plt.imshow(img)
    # plt.show()

    # Padding original images based on search region (near gt of CURRENT frame)
    #print(int(height/2),int(height/2),int(width/2),int(width/2))
    #print(original_dim)
    padded_img = cv2.copyMakeBorder(img,int(height/2),int(height/2),int(width/2),int(width/2),cv2.BORDER_CONSTANT,value=[0,0,0])
    padded_dim = padded_img.shape
    #print('padded_img dimension: ', padded_dim)

    # padded_img_show = Image.fromarray(padded_img, 'RGB')
    # padded_img_show.show()

    crop_x1 = int(topleft[0])
    crop_y1 = int(topleft[1])
    crop_x2 = int(bottomright[0] + width)
    crop_y2 = int(bottomright[1] + height)

    # there is a tiny bug here, but shouldnt impact performance, idex accessing
    # will raise error anyway
    # if crop_x1 < 0 or crop_y1 < 0:
    #     e = 'error: cropping coordinates exceed padded image dimensions'
    #     print(e)
    #     pdb.set_trace()
    # if crop_x2 > padded_dim[1] or crop_y2 > padded_dim[0]:
    #     e = 'error: cropping coordinates exceed padded image dimensions'
    #     print(e)
    #     pdb.set_trace()

    cropped_img = padded_img[crop_y1:crop_y2, crop_x1:crop_x2]

    # plt.imshow(cropped_img)
    # plt.show()
    #print('cropped_img shape: ', cropped_img.shape)

    # show original crop (bbox)
    # a = img[topleft[1]:bottomright[1], topleft[0]:bottomright[0]]
    # plt.imshow(a)
    # plt.show()

    return cropped_img

def preprocess_crop(img_crop):
    '''
    Shortening all the preprocessing steps
    '''
    img_crop = cv2.resize(img_crop, (227, 227)) # (227,227,3)
    img_crop = img_crop.astype(np.float)
    img_crop = np.expand_dims(img_crop, axis=0)
    img_crop = preprocess_input(img_crop)    # this process change RBG->BGR - but image already BGR?
    img_crop = np.transpose(img_crop, (0,3,1,2)) # (1,3,227,227)

    return img_crop



def convert_gt(original_dim, topleft_cf, bottomright_cf, topleft_gt, bottomright_gt):
    '''
    Convert the ground truth topleft_gt and bottomright_gt from the output of
    parse_gtstring() into number between 0 and 10, that is relative to the search
    region
    Return: shape = (1, 4)
      [topleft_x, topleft_y, bottomright_x, bottomright_y] (between 0 and 10)
    '''

    # bbox width and height
    width = (bottomright_cf[0]-topleft_cf[0])
    height = (bottomright_cf[1]-topleft_cf[1])
    # search region's topleft and bottom right coordinates
    crop_x1 = int(topleft_cf[0])
    crop_y1 = int(topleft_cf[1])
    crop_x2 = int(bottomright_cf[0] + width)
    crop_y2 = int(bottomright_cf[1] + height)

    # next frame new ground truth relative to padded image
    gt_x1 = int(topleft_gt[0] + width/2)
    gt_y1 = int(topleft_gt[1] + height/2)
    gt_x2 = int(bottomright_gt[0] + width/2)
    gt_y2 = int(bottomright_gt[1] + height/2)

    # make it between 0 and 10
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    gt_x1 = 10*(gt_x1 - crop_x1)/crop_width
    gt_y1 = 10*(gt_y1 - crop_y1)/crop_height
    gt_x2 = 10*(gt_x2 - crop_x1)/crop_width
    gt_y2 = 10*(gt_y2 - crop_y1)/crop_height


    # NOTE: if gt is not between 0 and 10. it should still be fine for training, since
    # we are training a regressors and we are backproping the 'differences' anyway
    # NOTE: if bottom right is less than topleft (both x and y) sth is wrong
    # (not yet implement), should be though

    gt = np.array([[gt_x1, gt_y1, gt_x2, gt_y2]])
    #print('ground truth values: ', gt)
    #print('ground truth dimensions: ', gt.shape)
    return gt




def convert_y_to_original_coordinates(y, original_dim, topleft_cf, bottomright_cf):
    '''
    convert regressed y between 0 and 10 (though can exceed 0 and 10) back into original
    image coordinates (non-padded)
    input:
        y = [topleft_x topleft_y bottomright_x bottomright_y]
        original_dim: original image dimensions
        topleft_cf, bottomright_cf: ground truth bbox location with regards to current frame
        (ie. half the search region)
    output:
        y_ori = [topleft, bottomright] = [[topleft_x topleft_y], bottomright_x bottomright_y]
        BUT in original coordinates
    '''

    width = (bottomright_cf[0]-topleft_cf[0])
    height = (bottomright_cf[1]-topleft_cf[1])
    # search region
    crop_x1 = int(topleft_cf[0] - width/2)
    crop_y1 = int(topleft_cf[1] - height/2)
    crop_x2 = int(bottomright_cf[0] + width/2)
    crop_y2 = int(bottomright_cf[1] + height/2)

    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    y_ori = np.zeros([1,4])
    y_ori[0,0] = int(crop_x1 + y[0,0]*crop_width/10)
    y_ori[0,1] = int(crop_y1 + y[0,1]*crop_height/10)
    y_ori[0,2] = int(crop_x1 + y[0,2]*crop_width/10)
    y_ori[0,3] = int(crop_y1 + y[0,3]*crop_height/10)

    # DEAL WITH OUT OF REGION
    if y_ori[0,0] < 0:
        # print(' Regressed value out of region, coordinates modified to be at the edge')
        y_ori[0,0] = 0

    if y_ori[0,1] < 0:
        # print(' Regressed value out of region, coordinates modified to be at the edge')
        y_ori[0,1] = 0

    if y_ori[0,2] > original_dim[1]:
        # print(' Regressed value out of region, coordinates modified to be at the edge')
        y_ori[0,2] = original_dim[1]

    if y_ori[0,3] > original_dim[0]:
        # print(' Regressed value out of region, coordinates modified to be at the edge')
        y_ori[0,3] = original_dim[0]


    return y_ori


def inference(model, i, list_of_gtstring, img_path, topleft_cf, bottomright_cf):

    '''
    shorthening the inference line
    output: y_gt, y_ori  (numpy array (1,4))
    '''


    gtstring_1 = list_of_gtstring[i]
    gtstring_2 = list_of_gtstring[i+1]

    if i == 0:
        frameno_1, topleft_cf, bottomright_cf = parse_gtstring(gtstring_1)
    else:
        frameno_1, _, _ = parse_gtstring(gtstring_1)

    frameno_2, topleft_gt, bottomright_gt = parse_gtstring(gtstring_2)


    y_gt = np.zeros([1,4])
    y_gt[0] = topleft_gt[0], topleft_gt[1], bottomright_gt[0], bottomright_gt[1]


    IMG_PATH1 = os.path.join(img_path, str(frameno_1).zfill(8) + '.jpg')
    IMG_PATH2 = os.path.join(img_path, str(frameno_2).zfill(8) + '.jpg')

    img1 = cv2.imread(IMG_PATH1)
    img1 = crop_image(img1, topleft_cf, bottomright_cf)
    img1 = preprocess_crop(img1)
    img2_ori = cv2.imread(IMG_PATH2)
    img2 = crop_image(img2_ori, topleft_cf, bottomright_cf)
    img2 = preprocess_crop(img2)

    y = model.predict([img1, img2])
    y_ori = convert_y_to_original_coordinates(y,
                                            img2_ori.shape,
                                            topleft_cf,
                                            bottomright_cf)
    topleft_cf = y_ori[0,0:2]
    bottomright_cf = y_ori[0,2:4]

    return y_gt, y_ori, topleft_cf, bottomright_cf







def plot_gt_on_searchregion(gt,img):
    '''
    **temporary**
    plot grond truth (between 0 and 10) on the search region
    Note: Does not work for gt outside of 0 and 10

    '''

    # TODO: need to modify to match the new gt data type (numpy array) - pbb not?

    dim = img.shape # (heigh, width, 3)
    img_width = dim[1]
    img_height = dim[0]
    bbox_width = int((gt[2]-gt[0])*img_width/10)
    bbox_height = int((gt[3]-gt[1])*img_height/10)
    bbox_topleft = [int(gt[0]*img_width/10), int(gt[1]*img_height/10)]

    # plotting
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle(bbox_topleft,bbox_width,bbox_height,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()

    return None























if __name__ == "__main__":

    gt_file_path = '/home/ren/Desktop/data/alov300++/alov300++_rectangleAnnotation_full/01-Light/01-Light_video00009.ann'
    list_of_gtstring = ann_to_list(gt_file_path) # ['line1', 'line2',...]

    gtstring = list_of_gtstring[1] # first line
    frameno, topleft, bottomright = parse_gtstring(gtstring)

    # get image according to frameno, and plot bounding boxes
    img_path = '/home/ren/Desktop/data/alov300++/imagedata++/01-Light/01-Light_video00009/0000000' + str(frameno) + '.jpg'
    plot_boundingbox(img_path, topleft, bottomright)




















#
