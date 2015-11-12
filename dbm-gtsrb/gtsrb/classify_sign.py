from pylearn2.utils import serial
from pylearn2.scripts.dbm.augment_input import augment_input
from theano import tensor as T
from theano import function
from PIL import Image
import numpy
import os

cwd = os.getcwd()

filenames = [f for f in os.listdir(cwd + "/signs/gtsdb_conv") if os.path.isfile(os.path.join(cwd + "/signs/gtsdb_conv", f))]  # select the name of the image
#filenames = ["../"]
PROB_THRESHOLD = 0.8  # select probability threshold
MF_STEPS = 1
model = 'dbm'

def predict(dbm, mlp, x):

    X = mlp.get_input_space().make_theano_batch()
    Y = mlp.fprop(X)
    Y = T.max_and_argmax(Y, axis = 1)
    f = function([X], Y)
    x = x.reshape([1, dbm.visible_layer.nvis + dbm.hidden_layers[1].detector_layer_dim])
    y = f(x)
    if y[0] < PROB_THRESHOLD:
        return y[0][0], -1
    return y[0][0], y[1][0]
        
def split_rgb(X):
    ''' 
    modify the matrix in such a way that each image 
    is stored with a rgb configuration (all reds, 
    all greens and all blues)
    '''
    first = True
    for img in X:
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        if first == True:
            rgb = numpy.asarray([numpy.concatenate([r, g, b])])
            first = False
        else:
            rgb = numpy.append(rgb, [numpy.concatenate([r, g, b])], axis=0)

    return rgb

for filename in filenames:
    data_path = cwd + "/signs/gtsdb_conv/" + filename
    mlp_path = cwd + "/" + model + "/" + model + "_gtsrb_mlp_dropout.pkl"
    dbm_path = cwd + "/" + model + "/" + model + "_gtsrb.pkl"
    if model == 'dbm':
        grbm_path = cwd + "/grbm/grbm_gtsrb.pkl"
        grbm = serial.load(grbm_path)
    
    mlp, dbm = serial.load(mlp_path), serial.load(dbm_path)
    
    img = Image.open(data_path)
    X = numpy.asarray([numpy.uint8(img.getdata())])
    X = numpy.cast['float32'](X)
    X = split_rgb(X)
    X = (X - numpy.mean(X)) / numpy.std(X)
    
    # Data augmentation
    if model == 'dbm':
        X = augment_input(X, dbm, MF_STEPS, "img", grbm)
    else:
        X = augment_input(X, dbm, MF_STEPS, "img")
    
    print 'Checking network predictions...'
    prob, prediction = predict(dbm, mlp, X)
    if prediction != -1:
        print "The image has been classified as " + str(prediction) + " with probability " + str(prob) + "."