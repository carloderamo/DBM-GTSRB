import os
import numpy

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode
from pylearn2.datasets import mnist_augmented

MAX_EPOCHS_MPDBM = 100
MAX_EPOCHS_MLP = 500

FINETUNING = 1
MF_STEPS = 1
DROPOUT = 1

@no_debug_mode
def test_train_example():
    
    # path definition
    path = '/home/deramo/workspace/dbnsignsdetection_pl2/dbnsignsdetection_pl2/mnist'
    train_path = os.path.join(path, 'mpdbm')
    
    cwd = os.getcwd()
    try:
        os.chdir(train_path)
        
        # load and train first layer    
        train_yaml_path = os.path.join(train_path, 'mpdbm_mnist.yaml')
        mpdbm_yaml = open(train_yaml_path, 'r').read()
        hyper_params_mpdbm = {'train_stop' : 60000,
                           'valid_stop' : 60000,
                           'batch_size' : 100,
                           'detector_layer_1_dim' : 1000,
                           'detector_layer_2_dim' : 2000,
                           'max_epochs' : MAX_EPOCHS_MPDBM,
                           'save_path' : train_path}
        
        mpdbm_yaml = mpdbm_yaml % (hyper_params_mpdbm)
        train = yaml_parse.load(mpdbm_yaml)
        
        print '\nTraining...\n'
        train.main_loop()

        if FINETUNING:
        
            # START SUPERVISED TRAINING WITH BACKPROPAGATION
            print '\n-----------------------------------'
            print '       Supervised training'
            print '-----------------------------------'
            
            # load dbm as a mlp
            if DROPOUT:
                train_yaml_path = os.path.join(train_path, 'mpdbm_mnist_mlp_dropout.yaml')
            else:
                train_yaml_path = os.path.join(train_path, 'mpdbm_mnist_mlp.yaml')
            mlp_yaml = open(train_yaml_path, 'r').read()
            hyper_params_mlp = {'train_stop': 60000,
                                'valid_stop' : 60000,
                                'batch_size' : 5000,
                                'nvis' : 784 + hyper_params_mpdbm['detector_layer_2_dim'],
                                'n_h0' : hyper_params_mpdbm['detector_layer_1_dim'],
                                'n_h1' : hyper_params_mpdbm['detector_layer_2_dim'],
                                'max_epochs' : MAX_EPOCHS_MLP,
                                'save_path' : train_path
                                }
    
            mlp_yaml = mlp_yaml % (hyper_params_mlp)
            train = yaml_parse.load(mlp_yaml)
            
            dbm = serial.load(os.path.join(train_path,
                                           'mpdbm_mnist.pkl'))
            
            # dataset & monitoring dataset insertion without .yaml file to avoid problems (don't know how to pass 'model' hyperparameter)
            train.dataset = mnist_augmented.MNIST_AUGMENTED(dataset = train.dataset, which_set = 'train', one_hot = 1, model = dbm, start = 0, stop = hyper_params_mlp['train_stop'], mf_steps = MF_STEPS)
            train.algorithm.monitoring_dataset = {#'valid' : mnist_augmented.MNIST_AUGMENTED(dataset = train.algorithm.monitoring_dataset['valid'], which_set = 'train', one_hot = 1, model = dbm, start = hyper_params_mlp['train_stop'], stop = hyper_params_mlp['valid_stop'], mf_steps = mf_steps), 
                                                  'test' : mnist_augmented.MNIST_AUGMENTED(dataset = train.algorithm.monitoring_dataset['test'], which_set = 'test', one_hot = 1, model = dbm, mf_steps = MF_STEPS)}
            
            # DBM TRAINED WEIGHTS CLAMPED FOR FINETUNING AS EXPLAINED BY HINTON
            
            # concatenate weights between first and second hidden layer & weights between visible and first hidden layer
            train.model.layers[0].set_weights(numpy.concatenate((dbm.hidden_layers[1].get_weights().transpose(), dbm.hidden_layers[0].get_weights())))
            
            # then clamp all the others normally
            for l, h in zip(train.model.layers[1:], dbm.hidden_layers[1:]):
                l.set_weights(h.get_weights())
             
            # clamp biases       
            for l, h in zip(train.model.layers, dbm.hidden_layers):
                l.set_biases(h.get_biases())
            
            print '\nDBM trained weights and biases have been clamped in the MLP.'
            
            print '\n...Finetuning...\n'
            train.main_loop()
        
    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()
