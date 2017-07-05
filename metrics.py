import numpy as np






def compute_IOU(y_ori, y_gt):
    '''
    Compute Intersection over Union (Overlap)
    Input:
        y_ori - output of model, converted to original dimension
        y_gt - ground truth coordinates of images
        both numpy array size (1,4)
    Output:
        overlap (IoU) value between 0 and 1
    '''

    boxA = y_ori[0]
    boxB = y_gt[0]

    # they does intersect
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value

    if iou <= 1 and iou > 0:
        return iou
    else:
        return 0







































#
