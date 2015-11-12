import os
import numpy

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode
from theano import function

##################
### PARAMETERS ###
##################

N_VIS = 32 * 32 * 3
N_HIDDEN = 4000

# PRETRAINING
TRAIN = 1

MAX_EPOCHS = 2000

@no_debug_mode
def test_train_example():
    
    # path definition
    train_path = cwd = os.getcwd()
    data_path = os.path.join('${PYLEARN2_DATA_PATH}', 'gtsrb', 'preprocessed')
    data_path = serial.preprocess(data_path)

    try:
        os.chdir(train_path)
        
        # START PRETRAINING
        # load and train grbm
        train_yaml_path = os.path.join(train_path, 'grbm_gtsrb.yaml')
        layer_yaml = open(train_yaml_path, 'r').read()
        hyper_params = {'batch_size' : 100,
                        'monitoring_batches' : 5,
                        'nvis' : N_VIS,
                        'nhid' : N_HIDDEN,
                        'max_epochs' : MAX_EPOCHS,
                        'data_path' : data_path,
                        'save_path' : train_path
                        }

        layer_yaml = layer_yaml % (hyper_params)
        train = yaml_parse.load(layer_yaml)
        
        if TRAIN:
            print '\n-----------------------------------'
            print '     Unsupervised pre-training'
            print '-----------------------------------\n'
            
            print '\nPre-Training grbm...\n'
            train.main_loop()
        
        # load dbm
        train_yaml_path = os.path.join(train_path, 'grbm_gtsrb_dbm.yaml')
        layer_yaml = open(train_yaml_path, 'r').read()
        hyper_params = {'batch_size' : 100,
                        'nvis' : N_VIS,
                        'nhid' : N_HIDDEN,
                        'data_path' : data_path
                        }
        
        layer_yaml = layer_yaml % (hyper_params)
        train = yaml_parse.load(layer_yaml)
        
        grbm = serial.load(os.path.join(train_path, 'grbm_gtsrb.pkl'))
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
        
        # save dbm
        serial.save('grbm_gtsrb_dbm.pkl', train.model)

    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()