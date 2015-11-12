import os
import numpy

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.scripts.dbm.augment_input import augment_input
from theano import function

##################
### PARAMETERS ###
##################

PRETRAINING_1 = 0
PRETRAINING_2 = 0
TRAINING = 0
FINETUNING = 0

N_HIDDEN_0 = 1000
N_HIDDEN_1 = 2000

# PRETRAINING
MAX_EPOCHS_L1 = 1000 
MAX_EPOCHS_L2 = 1000

# TRAINING
MAX_EPOCHS_DBM = 500 
SOFTMAX = 0

# FINETUNING
MAX_EPOCHS_MLP = 1500
DROPOUT = 1
AUGMENT_INPUT = 1
MF_STEPS = 1 # mf_steps for data augmentation


@no_debug_mode
def test_train_example():

    # path definition
    train_path = cwd = os.getcwd()
    data_path = os.path.join('${PYLEARN2_DATA_PATH}', 'gtsrb', 'preprocessed')
    data_path = serial.preprocess(data_path)
    grbm_path = os.path.join(train_path, '..', 'grbm')
    grbm = serial.load(os.path.join(grbm_path, 'grbm_gtsrb.pkl'))
    NVIS = grbm.nhid

    try:
        os.chdir(train_path)

        # START PRETRAINING
        # load and train first layer    
        train_yaml_path = os.path.join(train_path, 'dbm_gtsrb_l1.yaml')
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

        if PRETRAINING_1:

            layer1_yaml = layer1_yaml % (hyper_params_l1)
            train = yaml_parse.load(layer1_yaml)

            print("\n-----------------------------------"
                  "\n     Unsupervised pre-training     "
                  "\n-----------------------------------\n")

            print("\nPre-Training first layer...\n")
            train.main_loop()

        # load and train second layer
        train_yaml_path = os.path.join(train_path, 'dbm_gtsrb_l2.yaml')
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

        if PRETRAINING_2:

            layer2_yaml = layer2_yaml % (hyper_params_l2)
            train = yaml_parse.load(layer2_yaml)

            print("\n...Pre-training second layer...\n")
            train.main_loop()

        if TRAINING:

            # START TRAINING
            if SOFTMAX:
                train_yaml_path = os.path.join(train_path,
                                               'dbm_gtsrb_softmax.yaml')
            else:
                train_yaml_path = os.path.join(train_path, 'dbm_gtsrb.yaml')
            yaml = open(train_yaml_path, 'r').read()
            hyper_params_dbm = {
                'batch_size' : 100,
                'nvis' : NVIS,
                'detector_layer_1_dim' : hyper_params_l1['nhid'],
                'detector_layer_2_dim' : hyper_params_l2['nhid'],
                'max_epochs' : MAX_EPOCHS_DBM,
                'data_path' : data_path,
                'grbm_path' : grbm_path,
                'save_path' : train_path,
                }

            yaml = yaml % (hyper_params_dbm)
            train = yaml_parse.load(yaml)

            rbm1 = serial.load(os.path.join(train_path, 'dbm_gtsrb_l1.pkl'))
            rbm2 = serial.load(os.path.join(train_path, 'dbm_gtsrb_l2.pkl'))
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

            print("\nAll layers weights and biases have been clamped to"
                  " the respective layers of the DBM")

            print("\n-----------------------------------"
                  "\n     Unsupervised training         "
                  "\n-----------------------------------\n")

            print("\nTraining phase...")
            train.main_loop()

        if FINETUNING:
            if SOFTMAX:
                dbm = serial.load(os.path.join(train_path,
                                               'dbm_gtsrb_softmax.pkl'))
            else:
                dbm = serial.load(os.path.join(train_path,
                                               'dbm_gtsrb.pkl'))
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
                                                which_set='training',
                                                transformer=grbm)
                augmented_valid = augment_input(X=valid_set.X, model=dbm,
                                                mf_steps=MF_STEPS,
                                                which_set='validation',
                                                transformer=grbm)
                augmented_test = augment_input(X=test_set.X, model=dbm,
                                               mf_steps=MF_STEPS,
                                               which_set='test',
                                               transformer=grbm)
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
                                               'dbm_gtsrb_mlp_dropout.yaml')
            else:
                train_yaml_path = os.path.join(train_path,
                                               'dbm_gtsrb_mlp.yaml')
                
            mlp_yaml = open(train_yaml_path, 'r').read()
            hyper_params_mlp = {'batch_size' : 100,
                                'nvis' : NVIS + hyper_params_l2['nhid'],
                                'n_h0' : hyper_params_l1['nhid'],
                                'n_h1' : hyper_params_l2['nhid'],
                                'max_epochs' : MAX_EPOCHS_MLP,
                                'data_path' : data_path,
                                'grbm_path' : grbm_path,
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