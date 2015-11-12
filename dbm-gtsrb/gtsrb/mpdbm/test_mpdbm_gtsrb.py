import os
import numpy

from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode
from pylearn2.utils import serial
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.scripts.dbm.augment_input import augment_input

N_HIDDEN_0 = 1000
N_HIDDEN_1 = 2000

MAX_EPOCHS_MPDBM = 300
MAX_EPOCHS_MLP = 1500

TRAINING = 1
FINETUNING = 1

MF_STEPS = 1
AUGMENT_INPUT = 1
DROPOUT = 1

@no_debug_mode
def test_train_example():
    
    # path definition
    train_path = cwd = os.getcwd()
    data_path = os.path.join('${PYLEARN2_DATA_PATH}', 'gtsrb', 'preprocessed')
    data_path = serial.preprocess(data_path)
    NVIS = 32 * 32 * 3
    
    try:
        os.chdir(train_path)
        
        if TRAINING:
            # load and train first layer    
            train_yaml_path = os.path.join(train_path, 'mpdbm_gtsrb.yaml')
            layer1_yaml = open(train_yaml_path, 'r').read()
            hyper_params_mpdbm = {'batch_size' : 100,
                               'nvis' : NVIS,
                               'detector_layer_0_dim' : N_HIDDEN_0,
                               'detector_layer_1_dim' : N_HIDDEN_1,
                               'max_epochs' : MAX_EPOCHS_MPDBM,
                               'data_path' : data_path
                               }
            
            layer1_yaml = layer1_yaml % (hyper_params_mpdbm)
            train = yaml_parse.load(layer1_yaml)
            
            print '\nTraining...\n'
            train.main_loop()
        
        if FINETUNING:
            
            dbm = serial.load(os.path.join(train_path,
                                           'mpdbm_gtsrb.pkl'))
            if AUGMENT_INPUT:
                # data augmentation
                train_set = serial.load(os.path.join(data_path,
                                                     'preprocessed_train.pkl'))
                valid_set = serial.load(os.path.join(data_path,
                                                     'preprocessed_valid.pkl'))
                test_set = serial.load(os.path.join(data_path,
                                                    'preprocessed_test.pkl'))
                augmented_train = augment_input(X=train_set.X, model=dbm,
                                                mf_steps=MF_STEPS,
                                                which_set='training')
                augmented_valid = augment_input(X=valid_set.X, model=dbm,
                                                mf_steps=MF_STEPS,
                                                which_set='validation')
                augmented_test = augment_input(X=test_set.X, model=dbm,
                                               mf_steps=MF_STEPS,
                                               which_set='test')
                augmented_train_design_matrix = DenseDesignMatrix(
                    X=augmented_train,
                    y=train_set.y)
                augmented_valid_design_matrix = DenseDesignMatrix(
                    X=augmented_valid,
                    y=valid_set.y)
                augmented_test_design_matrix = DenseDesignMatrix(
                    X=augmented_test,
                    y=test_set.y)
                serial.save(os.path.join(data_path, 'augmented_train.pkl'),
                            augmented_train_design_matrix)
                serial.save(os.path.join(data_path, 'augmented_valid.pkl'),
                            augmented_valid_design_matrix)
                serial.save(os.path.join(data_path, 'augmented_test.pkl'),
                            augmented_test_design_matrix)

            # START SUPERVISED TRAINING WITH BACKPROPAGATION
            print("\n-----------------------------------"
                  "\n       Supervised training         "
                  "\n-----------------------------------\n")

            # load dbm as a mlp
            if DROPOUT:
                train_yaml_path = os.path.join(train_path,
                                               'mpdbm_gtsrb_mlp_dropout.yaml')
            else:
                train_yaml_path = os.path.join(train_path,
                                               'mpdbm_gtsrb_mlp.yaml')
                
            mlp_yaml = open(train_yaml_path, 'r').read()
            hyper_params_mlp = {'batch_size' : 100,
                                'nvis' : NVIS + hyper_params_mpdbm['detector_layer_1_dim'],
                                'n_h0' : hyper_params_mpdbm['detector_layer_0_dim'],
                                'n_h1' : hyper_params_mpdbm['detector_layer_1_dim'],
                                'max_epochs' : MAX_EPOCHS_MLP,
                                'data_path' : data_path,
                                'save_path' : train_path,
                                }

            mlp_yaml = mlp_yaml % (hyper_params_mlp)
            train = yaml_parse.load(mlp_yaml)

            # DBM TRAINED WEIGHTS CLAMPED FOR
            # FINETUNING AS EXPLAINED BY HINTON

            # concatenate weights between first and second hidden layer
            # & weights between visible and first hidden layer
            train.model.layers[0].set_weights(
                numpy.concatenate(
                    (dbm.hidden_layers[1].get_weights().transpose(),
                     dbm.hidden_layers[0].get_weights())
                )
            )

            # then clamp all the others normally
            for l, h in zip(train.model.layers[1:], dbm.hidden_layers[1:]):
                l.set_weights(h.get_weights())

            # clamp biases       
            for l, h in zip(train.model.layers, dbm.hidden_layers):
                l.set_biases(h.get_biases())

            print("\nDBM trained weights and biases have been clamped"
                  " in the MLP.")

            print("\n...Finetuning...\n")
            train.main_loop()
        
    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()
