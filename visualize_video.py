
from utils import plot_boundingbox, ann_to_list, parse_gtstring, crop_image
from utils import preprocess_crop, convert_gt, convert_y_to_original_coordinates
from utils import inference
from metrics import compute_IOU
import pdb
import numpy as np
import cv2
import os
import time

#import config as cfg

def evaluate(model):

    '''
    Evaluate the model
    input: model
    output: evaluation metrics
    '''

    # Evaluation setup
    overlap_threshold = 0.2
    threshold_list =  np.arange(0.05,1.05,0.05)
    success_list = np.zeros(len(threshold_list))

    success_count = 0
    total_frames = 0
    total_failures = 0
    total_overlap = 0
    infer_time = 0
    infer_count = 0

    ann_prepend = '/home/ren/Desktop/data/alov300++/ann/test/'
    img_prepend = '/home/ren/Desktop/data/alov300++/imagedata++'

    for category in os.listdir(ann_prepend): # eg. 01-Light
        folders = os.path.join(ann_prepend, category)

        cat_failures = 0

        for ann_files in os.listdir(folders): # .ann files
            ANN_PATH = os.path.join(folders, ann_files)
            list_of_gtstring = ann_to_list(ANN_PATH)

            img_path = os.path.join(img_prepend, category)
            img_path = os.path.join(img_path, ann_files)
            img_path = img_path.replace(".ann", "")

            # for each image in /01-Light/01-Light_video00001
            for i in range(0,len(list_of_gtstring)-1):
                if i == 0:
                    topleft_cf = 0
                    bottomright_cf = 0

                t1 = time.time()
                y_gt, y_ori, topleft_cf, bottomright_cf = inference(model,i,list_of_gtstring,img_path,topleft_cf,bottomright_cf)
                t2 = time.time()


                # we lost the box (got a straight line not a box)
                # everything else beyond this is not tracked
                if y_ori[0,0] == y_ori[0,2] or y_ori[0,1] == y_ori[0,3]:
                    frame_left = len(list_of_gtstring) - i

                    total_frames += frame_left
                    cat_failures += frame_left
                    # all overlap = 0
                    break


                # Evaluation
                overlap = compute_IOU(y_ori, y_gt)
                total_overlap += overlap
                infer_time += (t2-t1)
                infer_count += 1

                if overlap >= overlap_threshold:
                    success_count += 1
                    total_frames += 1
                else:
                    total_frames += 1
                    cat_failures += 1

                for j in range(0,len(threshold_list)):
                    if overlap >= threshold_list[j]:
                        success_list[j] += 1

        total_failures += cat_failures
        print('Category: ', category,' - #failures = ', cat_failures)

    # summary
    success_rate = success_count/total_frames
    avg_overlap = total_overlap/total_frames
    avg_fps = infer_count/infer_time

    print('===============')

    print('Evaluation Summary')
    print('Overlap threshold = ', overlap_threshold)
    print('Success rate: ', success_rate)
    print('avg Overlap: ', avg_overlap)
    print('Totoal failures: ', total_failures)

    print('===============')

    print('avg FPS: ', avg_fps)

    print('===============')
    success_rate_list = success_list/total_frames
    print('overlap threshold lists', threshold_list)
    print('success_rate_list (acc)', success_rate_list)

    print('===============')

    # return
    # overlap,
    # failures_each_class - depends on overlapping threshold
    # failures (sum up the failures in each class),
    # accuracy - depends on overlapping threshold

    # fps,

    return




def visualize_video(model):

    vid_name = '01-Light_video00008'
    vid_path = '01-Light/' + vid_name
    # chosen video path for ann
    ANN_PATH = '/home/ren/Desktop/data/alov300++/ann/test/' + vid_path + '.ann'
    list_of_gtstring = ann_to_list(ANN_PATH)

    # chosen video path for video frames
    img_path = '/home/ren/Desktop/data/alov300++/imagedata++/' + vid_path + '/'


    for i in range(0,len(list_of_gtstring)-1):

        # Get Image
        gtstring_1 = list_of_gtstring[i]
        gtstring_2 = list_of_gtstring[i+1]

        if i == 0:
            frameno_1, topleft_cf, bottomright_cf = parse_gtstring(gtstring_1)
        else:
            frameno_1, _, _ = parse_gtstring(gtstring_1)

        frameno_2, _, _ = parse_gtstring(gtstring_2)


        IMG_PATH1 = img_path + str(frameno_1).zfill(8) + '.jpg'
        IMG_PATH2 = img_path + str(frameno_2).zfill(8) + '.jpg'

        img1 = cv2.imread(IMG_PATH1)
        img1 = crop_image(img1, topleft_cf, bottomright_cf)
        img1 = preprocess_crop(img1)

        img2_ori = cv2.imread(IMG_PATH2)
        img2 = crop_image(img2_ori, topleft_cf, bottomright_cf)
        img2 = preprocess_crop(img2)

        # Regressed
        y = model.predict([img1, img2])
        # convert regressed results into [topleft, bottomright]
        y_ori = convert_y_to_original_coordinates(y, img2_ori.shape, topleft_cf, bottomright_cf)


        # plot boudning boxes
        if i == 0:
            # add first image to the list (gt)
            vidname = vid_name + str(frameno_1)
            plot_boundingbox(IMG_PATH1, topleft_cf, bottomright_cf, vidname)

        vidname = vid_name + '_f'+ str(frameno_2)
        plot_boundingbox(IMG_PATH2, y_ori[0,0:2], y_ori[0,2:4], vidname)


        # We are using the previous tracking results track next frame
        topleft_cf = y_ori[0,0:2]
        bottomright_cf = y_ori[0,2:4]


    return len(list_of_gtstring)



def visualise_calipsa(model):

    vid_name = 'car1'
    vid_path = '/home/ren/Desktop/data/calipsa/clips/' + vid_name
    import os

    number_of_files = len([item for item in os.listdir(vid_path) if os.path.isfile(os.path.join(vid_path, item))])

    # Need to specify what to track
    topleft_cf = [504,37]
    bottomright_cf = [540,72]

    start_frame = 21
    for i in range(start_frame ,number_of_files-1):
        #name = 'frame {}.png'.format(i)
        IMG_PATH1 = vid_path + '/frame-' + str(i) + '.png'
        IMG_PATH2 = vid_path + '/frame-' + str(i+1) + '.png'

        img1 = cv2.imread(IMG_PATH1)
        img1 = crop_image(img1, topleft_cf, bottomright_cf)
        img1 = preprocess_crop(img1)

        img2_ori = cv2.imread(IMG_PATH2)
        img2 = crop_image(img2_ori, topleft_cf, bottomright_cf)
        img2 = preprocess_crop(img2)

        # Regressed
        y = model.predict([img1, img2])
        # convert regressed results into [topleft, bottomright]
        y_ori = convert_y_to_original_coordinates(y, img2_ori.shape, topleft_cf, bottomright_cf)


        # plot boudning boxes
        if i == start_frame :
            # add first image to the list (gt)
            vidname = vid_name +  '_f' + str(i)
            plot_boundingbox(IMG_PATH1, topleft_cf, bottomright_cf, vidname)

        vidname = vid_name + '_f'+ str(i+1)
        plot_boundingbox(IMG_PATH2, y_ori[0,0:2], y_ori[0,2:4], vidname)


        # We are using the previous tracking results track next frame
        topleft_cf = y_ori[0,0:2]
        bottomright_cf = y_ori[0,2:4]



    return number_of_files
























#
