from pylearn2.utils import serial
import sys
from pylearn2.config import yaml_parse
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
from PIL import Image
from pylearn2.utils import safe_zip
from theano.compat import OrderedDict
from pylearn2.utils import sharedX

def make_state(layer, num_examples):

    empty_input = layer.h_space.get_origin_batch(num_examples)
    empty_output = layer.output_space.get_origin_batch(num_examples)
    
    empty_input[0][0:500] = empty_output[0][0:500] = 1

    h_state = sharedX(empty_input)
    p_state = sharedX(empty_output)

    p_state.name = 'p_sample_shared'
    h_state.name = 'h_sample_shared'

    return p_state, h_state

def make_layer_to_state(model, num_examples, rng=None):

    # Make a list of all layers
    layers = [model.visible_layer] + model.hidden_layers

    if rng is None:
        rng = model.rng

    states = [layer.make_state(num_examples, rng) for layer in layers]
    states[-1] = make_state(model.hidden_layers[-1], num_examples = 1)

    zipped = safe_zip(layers, states)

    def recurse_check(layer, state):
        if isinstance(state, (list, tuple)):
            for elem in state:
                recurse_check(layer, elem)
        else:
            val = state.get_value()
            m = val.shape[0]
            if m != num_examples:
                raise ValueError(layer.layer_name + " gave state with " +
                                 str(m) + " examples in some component."
                                 "We requested " + str(num_examples))

    for layer, state in zipped:
        recurse_check(layer, state)

    rval = OrderedDict(zipped)

    return rval

def show_reconstruction():

    _, model_path = sys.argv
    
    print 'Loading model...'
    model = serial.load(model_path)
    
    dataset_yaml_src = model.dataset_yaml_src
    
    print 'Loading data...'
    dataset = yaml_parse.load(dataset_yaml_src)
    
    layer_to_state = make_layer_to_state(model, num_examples = 1)
    theano_rng = RandomStreams(seed = 243)
    layer_to_clamp = {model.hidden_layers[-1] : True}
    layer_to_updated = model.sampling_procedure.sample(layer_to_state, theano_rng, layer_to_clamp, num_steps = 5)

    v = function([], layer_to_updated[model.visible_layer])
    v = v()
    v = v.reshape(28, 28)
    img = Image.fromarray(v, 'L')
    img = img.resize((200, 200), Image.ANTIALIAS)
    img.show()
    '''h1 = function([], layer_to_updated[model.hidden_layers[1]][0])
    h1 = h1()
    h1 = h1.reshape(25, 20)
    h0 = function([], layer_to_updated[model.hidden_layers[0]][0])
    h0 = h0()
    h0 = h0.reshape(25, 20)
    img = Image.fromarray(h1, 'L')
    img = img.resize((200, 200), Image.ANTIALIAS)
    img.show()
    img = Image.fromarray(h0, 'L')
    img = img.resize((200, 200), Image.ANTIALIAS)
    img.show()'''

if __name__ == '__main__':
    show_reconstruction()