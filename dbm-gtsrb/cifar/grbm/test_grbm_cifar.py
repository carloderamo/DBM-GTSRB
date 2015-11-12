import os
import numpy

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode
from theano import function

##################
### PARAMETERS ###
##################

N_VIS = 1024 * 3
N_HIDDEN = 4000

# PRETRAINING
TRAIN_GRBM = 1
TRAIN_DBM = 0
SOFTMAX = 0

MAX_EPOCHS_GRBM = 3000
MAX_EPOCHS_DBM = 0

@no_debug_mode
def test_train_example():
    
    # path definition
    train_path = cwd = os.getcwd()
    data_path = os.path.join(train_path, '..', 'datasets')
    
    try:
        os.chdir(train_path)
        
        # START PRETRAINING
        # load and train grbm
        train_yaml_path = os.path.join(train_path, 'grbm_cifar.yaml')
        layer_yaml = open(train_yaml_path, 'r').read()
        hyper_params = {'batch_size' : 100,
                        'monitoring_batches' : 5,
                        'nvis' : N_VIS,
                        'nhid' : N_HIDDEN,
                        'max_epochs' : MAX_EPOCHS_GRBM,
                        'data_path' : data_path,
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
            train_yaml_path = os.path.join(train_path, 'grbm_cifar_dbm_softmax.yaml')
        else:
            train_yaml_path = os.path.join(train_path, 'grbm_cifar_dbm.yaml')
        layer_yaml = open(train_yaml_path, 'r').read()
        hyper_params = {'batch_size' : 100,
                        'nvis' : N_VIS,
                        'nhid' : N_HIDDEN,
                        'max_epochs' : MAX_EPOCHS_DBM,
                        'data_path' : data_path,
                        'save_path' : train_path
                        }
        
        layer_yaml = layer_yaml % (hyper_params)
        train = yaml_parse.load(layer_yaml)
        
        grbm = serial.load(os.path.join(train_path, 'grbm_cifar.pkl'))
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
            serial.save('grbm_cifar_dbm.pkl', train.model)

    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()