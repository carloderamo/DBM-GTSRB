import os

from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode
from pylearn2.utils import serial

DROPOUT = 0

N_HIDDEN_0 = 1000
N_HIDDEN_1 = 2000
MAX_EPOCHS = 100

@no_debug_mode
def test_train_example():
    
    # path definition
    train_path = cwd = os.getcwd()
    data_path = os.path.join(train_path, '..', '..', '..', '..', 'datasets', 'gtsrb', 'preprocessed')
    grbm_path = os.path.join(train_path, '..', 'grbm', 'grbm_gtsrb.pkl')
    grbm = serial.load(grbm_path)
    NVIS = grbm.nhid
    
    try:
        os.chdir(train_path)
        
        # load and train first layer
        if DROPOUT:
            train_yaml_path = os.path.join(train_path, 'mlp_gtsrb_dropout.yaml')
        else:
            train_yaml_path = os.path.join(train_path, 'mlp_gtsrb.yaml')
        layer1_yaml = open(train_yaml_path, 'r').read()
        hyper_params_l1 = {'batch_size': 100,
                           'nvis': NVIS,
                           'n_h0': N_HIDDEN_0,
                           'n_h1': N_HIDDEN_1,
                           'max_epochs': MAX_EPOCHS,
                           'data_path' : data_path,
                           'grbm_path' : grbm_path,
                           }
        
        layer1_yaml = layer1_yaml % (hyper_params_l1)
        train = yaml_parse.load(layer1_yaml)
        
        print '\nTraining...\n'
        train.main_loop()
        
    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()
