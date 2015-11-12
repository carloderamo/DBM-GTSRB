import os
import numpy

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode
from theano import function

##################
### PARAMETERS ###
##################

N_HIDDEN = 500

# PRETRAINING
TRAIN_GRBM = 1
TRAIN_DBM = 0
SOFTMAX = 0

MAX_EPOCHS_GRBM = 100
MAX_EPOCHS_DBM = 0

@no_debug_mode
def test_train_example():
    
    # path definition
    train_path = cwd = os.getcwd()
    
    try:
        os.chdir(train_path)
        
        # START PRETRAINING
        # load and train grbm    
        train_yaml_path = os.path.join(train_path, 'grbm_mnist.yaml')
        layer_yaml = open(train_yaml_path, 'r').read()
        hyper_params = {'train_stop' : 60000,
                        'batch_size' : 100,
                        'monitoring_batches' : 5,
                        'nhid' : N_HIDDEN,
                        'max_epochs' : MAX_EPOCHS_GRBM,
                        'save_path' : train_path
                        }

        layer_yaml = layer_yaml % (hyper_params)
        train = yaml_parse.load(layer_yaml)

        if TRAIN_GRBM:
            print '\n-----------------------------------'
            print '     Unsupervised pre-training'
            print '-----------------------------------\n'
            
            print '\nPre-Training grbm...\n'
            train.main_loop()
        
        # load and train second layer
        if SOFTMAX:
            train_yaml_path = os.path.join(train_path, 'grbm_mnist_dbm_softmax.yaml')
        else:
            train_yaml_path = os.path.join(train_path, 'grbm_mnist_dbm.yaml')
        layer_yaml = open(train_yaml_path, 'r').read()
        hyper_params = {'train_stop' : 60000,
                        'batch_size' : 100,
                        'nhid' : N_HIDDEN,
                        'max_epochs' : MAX_EPOCHS_DBM,
                        'save_path' : train_path
                        }
        
        layer_yaml = layer_yaml % (hyper_params)
        train = yaml_parse.load(layer_yaml)
        
        grbm = serial.load(os.path.join(train_path, 'grbm_mnist.pkl'))
        train.model.hidden_layers[0].set_weights(grbm.get_weights())
        bias_param = grbm.get_params()[1]
        fun = function([], bias_param)
        cuda_bias = fun()
        bias = numpy.asarray(cuda_bias)
        train.model.visible_layer.set_biases(bias)
        bias_param = grbm.get_params()[2]
        fun = function([], bias_param)
        cuda_bias = fun()
        bias = numpy.asarray(cuda_bias)
        train.model.hidden_layers[0].set_biases(bias)

        if TRAIN_DBM:
            print '\nTraining dbm...\n'
            train.main_loop()
        else:
            serial.save('grbm_mnist_dbm.pkl', train.model)

    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()