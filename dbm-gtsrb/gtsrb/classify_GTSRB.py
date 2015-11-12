from pylearn2.utils import serial
from theano import tensor as T
from theano import function
import csv

img_indexes =  [0, 2, 3, 6, 8, 12, 13, 14, 19, 23, 33, 37, 39, 42,
                45, 50, 69, 78, 96, 216, 314, 328, 331, 358, 368,
                475, 600, 806, 883, 7418, 7435, 7541, 7553, 10761,
                10773, 10998, 11358, 11647, 11825, 11956, 12111,
                12351, 12404] # index of the test set image to be tested in signs
model_name = 'mlp'
folder = 'std_gtsrb'
errors = []

data_path = '${PYLEARN2_DATA_PATH}/gtsrb'
data_path = serial.preprocess(data_path)
if model_name == 'mlp':
    model_path = "/home/deramo/workspace/dbnsignsdetection_pl2/dbnsignsdetection_pl2/gtsrb/dbm/dbm_gtsrb_mlp_dropout.pkl"
    test_path = data_path + '/preprocessed/augmented_test.pkl'
else:
    model_path = "/home/deramo/workspace/dbnsignsdetection_pl2/dbnsignsdetection_pl2/gtsrb/dbm/dbm_gtsrb_softmax.pkl"
    test_path = data_path + folder + '/Final_Test/Images/test_rgb_bb_nocrop32x32:1000.pkl.gz'

try:
    model = serial.load(model_path)
except Exception, e:
    print "error loading {}:".format(model_path)
    print e
    quit(-1)

def predict(model_name, model_path, x):

    X = model.get_input_space().make_theano_batch()
    if model_name == 'mlp':
        Y = model.fprop(X)
    
    Y = T.argmax(Y, axis = 1)
    f = function([X], Y)
    
    x = x.reshape([1, model.visible_layer.nvis + model.layers[1].dim])
    y = f(x)
    
    return y[0]

print 'Checking network predictions...'
with open(data_path + '/std_gtsrb/Final_Test/Images/GT-final_test.csv') as f:
    reader = csv.reader(f, delimiter = ';')  # csv parser for annotations file
    reader.next() # skip header
    csv_table = map(tuple, reader)
    for img_index in img_indexes:
        dataset = serial.load(test_path)
        test_img = dataset.X[img_index]
        
        prediction = predict(model_name, model_path, test_img)
        
        if str(prediction) != csv_table[img_index][7]:
            errors = errors + [[img_index, prediction]]

print 'Printing errors...'
for e in errors:
    print 'Image ' + str(e[0]) + ' assigned to label ' + str(e[1])
error_percentage = len(errors) / float(len(img_indexes))
print 'Errors percentage: ' + str(error_percentage * 100) + '%'