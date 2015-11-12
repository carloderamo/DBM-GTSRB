from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
import os
import numpy
from pylearn2.testing import no_debug_mode

##################
### PARAMETERS ###
##################

PRETRAINING = 1
TRAINING = 1
FINETUNING = 1

N_HIDDEN_0 = 400
N_HIDDEN_1 = 600

# PRETRAINING
MAX_EPOCHS_L1 = 100
MAX_EPOCHS_L2 = 200

# TRAINING
MAX_EPOCHS_DBM = 500
SOFTMAX = 1
    
@no_debug_mode
def test_train_example():
    
    # path definition
    train_path = cwd = os.getcwd()
    data_path = os.path.join(train_path, '..', 'datasets')
    grbm_path = os.path.join(train_path, '..', 'grbm')
    grbm = serial.load(os.path.join(grbm_path, 'grbm_cifar.pkl'))
    NVIS = grbm.nhid
    
    try:
        os.chdir(train_path)
        
        # START PRETRAINING
        # load and train first layer    
        train_yaml_path = os.path.join(train_path, 'dbm_cifar_l1.yaml')
        layer1_yaml = open(train_yaml_path, 'r').read()
        hyper_params_l1 = {'batch_size' : 100,
                           'monitoring_batches' : 5,
                           'nvis' : NVIS,
                           'nhid' : N_HIDDEN_0,
                           'max_epochs' : MAX_EPOCHS_L1,
                           'data_path' : data_path,
                           'grbm_path' : grbm_path,
                           'save_path' : train_path,
                           }
        
        if PRETRAINING:
        
            layer1_yaml = layer1_yaml % (hyper_params_l1)
            train = yaml_parse.load(layer1_yaml)
            
            print '\n-----------------------------------'
            print '     Unsupervised pre-training'
            print '-----------------------------------\n'
            
            print '\nPre-Training first layer...\n'
            train.main_loop()
    
        # load and train second layer
        train_yaml_path = os.path.join(train_path, 'dbm_cifar_l2.yaml')
        layer2_yaml = open(train_yaml_path, 'r').read()
        hyper_params_l2 = {'batch_size' : 100,
                           'monitoring_batches' : 5,
                           'nvis' : hyper_params_l1['nhid'],
                           'nhid' : N_HIDDEN_1,
                           'max_epochs' : MAX_EPOCHS_L2,
                           'data_path' : data_path,
                           'grbm_path' : grbm_path,
                           'save_path' : train_path,
                           }
        
        if PRETRAINING:
            
            layer2_yaml = layer2_yaml % (hyper_params_l2)
            train = yaml_parse.load(layer2_yaml)
            
            print '\n...Pre-training second layer...\n'
            train.main_loop()
        
        if TRAINING:
            
            # START TRAINING
            if SOFTMAX:
                train_yaml_path = os.path.join(train_path, 'dbm_cifar_softmax.yaml')
            else:
                train_yaml_path = os.path.join(train_path, 'dbm_cifar.yaml')
            yaml = open(train_yaml_path, 'r').read()
            hyper_params_dbm = {'batch_size' : 100,
                                'nvis' : NVIS,
                                'detector_layer_1_dim' : hyper_params_l1['nhid'],
                                'detector_layer_2_dim' : hyper_params_l2['nhid'],
                                'monitoring_batches' : 5,
                                'max_epochs' : MAX_EPOCHS_DBM,
                                'data_path' : data_path,
                                'grbm_path' : grbm_path,
                                'save_path' : train_path,
                                }
            
            yaml = yaml % (hyper_params_dbm)
            train = yaml_parse.load(yaml)

            rbm1 = serial.load(os.path.join(train_path, 'dbm_cifar_l1.pkl'))
            rbm2 = serial.load(os.path.join(train_path, 'dbm_cifar_l2.pkl'))
            pretrained_rbms = [rbm1, rbm2]
            
            # clamp pretrained weights into respective dbm layers
            for h, l in zip(train.model.hidden_layers, pretrained_rbms):
                h.set_weights(l.get_weights())
            
            # clamp pretrained biases into respective dbm layers
            bias_param = pretrained_rbms[0].get_params()[1]
            fun = function([], bias_param)
            cuda_bias = fun()
            bias = numpy.asarray(cuda_bias)
            train.model.visible_layer.set_biases(bias)
            bias_param = pretrained_rbms[-1].get_params()[1]
            fun = function([], bias_param)
            cuda_bias = fun()
            bias = numpy.asarray(cuda_bias)
            train.model.hidden_layers[0].set_biases(bias)
            bias_param = pretrained_rbms[-1].get_params()[2]
            fun = function([], bias_param)
            cuda_bias = fun()
            bias = numpy.asarray(cuda_bias)
            train.model.hidden_layers[1].set_biases(bias)
            
            print '\nAll layers weights and biases have been clamped to the respective layers of the DBM'

            print '\n-----------------------------------'
            print '     Unsupervised training'
            print '-----------------------------------\n'
            
            print '\nTraining phase...'
            train.main_loop()
            
    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()
