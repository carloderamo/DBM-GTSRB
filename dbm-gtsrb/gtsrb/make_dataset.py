import os

from pylearn2.datasets import preprocessing, gtsrb
from pylearn2.utils import serial
from copy import deepcopy

'''
this script loads gtsrb dataset according to the desired size and then
preprocess the whole dataset. During all this process both the normal
resized and cropped images together with the preprocessed ones, are
saved.
BE AWARE THAT ONLY EXAMPLE FROM THE START TO STOP ARE PREPROCESSED
AND SAVED!!!!!!
'''

gtsrb_list = {0 : 'std_gtsrb', 1 : 'imadjust_gtsrb', 2 : 'histeq_gtsrb', 3 : 'adapthisteq_gtsrb', 4 : 'hsv_gtsrb', 5 : 'chr_gtsrb', 6 : 'imadjust_rgb_gtsrb', 7 : 'adapthisteq_lab_gtsrb', 8 : 'imadjust_rgb_adapthisteq_gtsrb'}
IMADJUST = 0
HISTEQ = 0
ADAPTHISTEQ = 1
HSV = 0
CHR = 0
IMADJUST_RGB = 0
ADAPTHISTEQ_LAB = 0
IMADJUST_RGB_ADAPTHISTEQ = 0
WHITENING = 0

IMG_SIZE = [32, 32]
PIXELS = IMG_SIZE[0] * IMG_SIZE[1]
BOUND = 1000
RGB = 1
CROP = 0

if IMADJUST:
    which_gtsrb = gtsrb_list[1]
    BOUNDING_BOX = 0
elif HISTEQ:
    which_gtsrb = gtsrb_list[2]
    BOUNDING_BOX = 0
elif ADAPTHISTEQ:
    which_gtsrb = gtsrb_list[3]
    BOUNDING_BOX = 0
elif HSV:
    which_gtsrb = gtsrb_list[4]
    BOUNDING_BOX = 0
elif CHR:
    which_gtsrb = gtsrb_list[5]
    BOUNDING_BOX = 0
elif IMADJUST_RGB:
    which_gtsrb = gtsrb_list[6]
    BOUNDING_BOX = 0
elif ADAPTHISTEQ_LAB:
    which_gtsrb = gtsrb_list[7]
    BOUNDING_BOX = 0
elif IMADJUST_RGB_ADAPTHISTEQ:
    which_gtsrb = gtsrb_list[8]
    BOUNDING_BOX = 0
else:
    which_gtsrb = gtsrb_list[0]
    BOUNDING_BOX = 1

TRAIN_START = 0
TRAIN_STOP = 39209
VALID_START = TRAIN_STOP
VALID_STOP = 39209
TEST_START = 0
TEST_STOP = 12630

train = gtsrb.GTSRB(which_set='train', which_gtsrb=which_gtsrb, img_size=IMG_SIZE, bound=BOUND, rgb=RGB, bounding_box=BOUNDING_BOX, crop=CROP)
valid = deepcopy(train)
test = gtsrb.GTSRB(which_set='test', which_gtsrb=which_gtsrb, img_size=IMG_SIZE, bound=BOUND, rgb=RGB, bounding_box=BOUNDING_BOX, crop=CROP)

train.X, train.y = train.X[TRAIN_START:TRAIN_STOP], train.y[TRAIN_START:TRAIN_STOP]
valid.X, valid.y = valid.X[VALID_START:VALID_STOP], valid.y[VALID_START:VALID_STOP]
test.X, test.y = test.X[TEST_START:TEST_STOP], test.y[TEST_START:TEST_STOP]

print 'Preprocessing...'
pipeline = preprocessing.Pipeline()
# image preprocessing
pipeline.items.append(preprocessing.GlobalContrastNormalization(use_std=True))
if WHITENING:
    pipeline.items.append(preprocessing.ZCA(filter_bias=0.5))
train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
valid.apply_preprocessor(preprocessor=pipeline, can_fit=False)
test.apply_preprocessor(preprocessor=pipeline, can_fit=False)
            
data_path = os.path.join('${PYLEARN2_DATA_PATH}', 'gtsrb')
data_path = serial.preprocess(data_path)
train_path = os.path.join(data_path,'preprocessed', 'preprocessed_train.pkl')
valid_path = os.path.join(data_path, 'preprocessed', 'preprocessed_valid.pkl')
test_path = os.path.join(data_path, 'preprocessed', 'preprocessed_test.pkl')
serial.save(train_path, train)
serial.save(valid_path, valid)
serial.save(test_path, test)
