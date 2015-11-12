import os
from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode
from pylearn2.utils import serial

MAX_EPOCHS = 100
NHIDDEN0 = 2500
NHIDDEN1 = 3000

@no_debug_mode
def test_train_example():
    
    path = cwd = os.getcwd()
    grbm_path = os.path.join(path, '..', 'dbm')
    grbm = serial.load(os.path.join(grbm_path, 'dbm_cifar_grbm.pkl'))
    data_path = os.path.join(path, '..', 'datasets')
    NVIS = grbm.nhid
    try:
        os.chdir(path)
        
        # load and train first layer    
        train_yaml_path = os.path.join(path, 'mlp_cifar.yaml')
        layer1_yaml = open(train_yaml_path, 'r').read()
        hyper_params_l1 = {'batch_size' : 100,
                           'monitoring_batches' : 5,
                           'nvis' : NVIS,
                           'dim_h0' : NHIDDEN0,
                           'dim_h1' : NHIDDEN1,
                           'max_epochs' : MAX_EPOCHS,
                           'data_path' : data_path,
                           'grbm_path' : grbm_path}
        
        layer1_yaml = layer1_yaml % (hyper_params_l1)
        train = yaml_parse.load(layer1_yaml)
        
        print '\nTraining...\n'
        train.main_loop()
        
    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()
