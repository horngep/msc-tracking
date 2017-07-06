import os
import pdb
from utils import *


'''
    Count number of total data by looping to count the ann files
'''


ann_dir = '/home/ren/Desktop/data/alov300++/ann/train'




# 307 ann files
# 15877 labeled images
# (9431/3288/3158) (train/val/test) (60/20/20)
count = 0

for folder in os.listdir(ann_dir):
    subfolder = os.path.join(ann_dir, folder)
    for annfile in os.listdir(subfolder):
        path = os.path.join(subfolder, annfile)
        list_of_gtstring = ann_to_list(path)
        count += len(list_of_gtstring)

print('train data: ',count)





















#
