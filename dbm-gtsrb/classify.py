from pylearn2.utils import serial
from theano import tensor as T
from theano import function

def predict(model_name, model_path, x):
    
    print "loading model..."
    
    try:
        model = serial.load(model_path)
    except Exception, e:
        print "error loading {}:".format(model_path)
        print e
        quit(-1)
    
    print "setting up symbolic expressions..."
    X = model.get_input_space().make_theano_batch()
    if model_name == 'dbn':
        Y = model.fprop(X)
    else:
        ''' remember that if you use a dbm, it needs to have a softmax layer 
        and the supervised flag of the cost function set to true'''
        hidden_expectations = model.mf(X) 
        Y = hidden_expectations[-1]
    
    Y = T.argmax(Y, axis = 1)
    f = function([X], Y)
    
    y = f(x)
    
    print "writing predictions..."
    
    print y[0]