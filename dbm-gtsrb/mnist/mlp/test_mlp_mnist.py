import os
import time

from pylearn2.config import yaml_parse
from pylearn2.testing import no_debug_mode

MAX_EPOCHS = 5

@no_debug_mode
def test_train_example():
    
    total_time = 0.
    
    # path definition
    path = '/home/carlo/workspace/dbnsignsdetection_pl2/dbnsignsdetection_pl2/mnist'
    train_path = os.path.join(path, 'mlp')
    
    cwd = os.getcwd()
    try:
        os.chdir(train_path)
        
        # load and train first layer    
        train_yaml_path = os.path.join(train_path, 'mlp_mnist.yaml')
        layer1_yaml = open(train_yaml_path, 'r').read()
        hyper_params_l1 = {'train_stop' : 60000,
                           'batch_size' : 60000,
                           'monitoring_batches' : 1,
                           'dim_h0' : 500,
                           'max_epochs' : MAX_EPOCHS,
                           'save_path' : train_path}
        
        layer1_yaml = layer1_yaml % (hyper_params_l1)
        train = yaml_parse.load(layer1_yaml)
        
        print '\nTraining...\n'
        start_time = time.clock()
        train.main_loop()
        end_time = time.clock()
        
        total_time += (end_time - start_time) / 60.
        
        print '\nTotal time elapsed: %.2fm' % total_time
        
    finally:
        os.chdir(cwd)

if __name__ == '__main__':
    test_train_example()
