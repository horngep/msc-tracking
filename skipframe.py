from utils import plot_boundingbox, ann_to_list, parse_gtstring, crop_image
from utils import preprocess_crop, convert_gt, convert_y_to_original_coordinates
from utils import inference
from metrics import compute_IOU
import pdb
import numpy as np
import cv2
import os
import time


def skipframe(model, frameskip):



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
            for i in range(0,len(list_of_gtstring)-2, frameskip+1):
                if i == 0:
                    topleft_cf = 0
                    bottomright_cf = 0

                t1 = time.time()
                y_gt, y_ori, topleft_cf, bottomright_cf = inference_skipframe(model,i,
                                                                    list_of_gtstring,
                                                                    img_path,
                                                                    topleft_cf,
                                                                    bottomright_cf,
                                                                    frameskip)
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

def inference_skipframe(model, i, list_of_gtstring, img_path, topleft_cf, bottomright_cf, frameskip):

    '''
    shorthening the inference line
    output: y_gt, y_ori  (numpy array (1,4))
    '''


    gtstring_1 = list_of_gtstring[i]
    gtstring_2 = list_of_gtstring[i+frameskip+1]

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
