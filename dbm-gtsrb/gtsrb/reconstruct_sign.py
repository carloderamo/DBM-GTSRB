from pylearn2.utils import serial
import os
from theano.compat.six.moves import input, xrange
from theano import function
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
import numpy
from PIL import Image

filename = ""
shape = [32, 32]

def split_rgb(X):
    ''' 
    modify the matrix in such a way that each image 
    is stored with a rgb configuration (all reds, 
    all greens and all blues)
    '''
    first = True
    for img in X:
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        if first == True:
            rgb = numpy.asarray([numpy.concatenate([r, g, b])])
            first = False
        else:
            rgb = numpy.append(rgb, [numpy.concatenate([r, g, b])], axis=0)

    return rgb

rows = cols = 1
m = 1
cwd = os.getcwd()
print('Loading model...')
grbm_path = cwd + "/grbm/grbm_gtsrb_dbm.pkl"
grbm = serial.load(grbm_path)
data_path = cwd + "/signs/" + filename

print('Loading data...')
img = Image.open(data_path)
X = numpy.asarray([numpy.uint8(img.getdata())])
X = numpy.cast['float32'](X)
X = split_rgb(X)
X = (X - numpy.mean(X)) / numpy.std(X)
temp = []
for i in xrange(X.shape[1] / 3):
    temp = numpy.append(temp, [X[0, i], X[0, i + shape[0] * shape[0]], X[0, i + shape[0] * shape[0] * 2]])
temp = temp.reshape(shape[0], shape[0], 3)
X = temp

dataset_yaml_src = grbm.dataset_yaml_src
dataset = yaml_parse.load(dataset_yaml_src)

vis_batch = dataset.get_batch_topo(m)

_, patch_rows, patch_cols, channels = vis_batch.shape

assert _ == m

mapback = hasattr(dataset, 'mapback_for_viewer')

actual_cols = 2 * cols * (1 + mapback) * (1 + (channels == 2))
pv = PatchViewer((rows, actual_cols), (patch_rows, patch_cols), is_color=(channels == 3))

batch = grbm.visible_layer.space.make_theano_batch(batch_size=1)
topo = batch.ndim > 2
reconstruction = grbm.reconstruct(batch)
recons_func = function([batch], reconstruction)


def show():
    vis_batch = ipt = numpy.asarray([X])
    if not topo:
        ipt = dataset.get_design_matrix(ipt)
    recons_batch = recons_func(ipt.astype(batch.dtype))
    if not topo:
        recons_batch = dataset.get_topological_view(recons_batch)
    if mapback:
        design_vis_batch = vis_batch
        if design_vis_batch.ndim != 2:
            design_vis_batch = dataset.get_design_matrix(design_vis_batch.copy())
        mapped_batch_design = dataset.mapback(design_vis_batch.copy())
        mapped_batch = dataset.get_topological_view(
                mapped_batch_design.copy())
        design_r_batch = recons_batch.copy()
        if design_r_batch.ndim != 2:
            design_r_batch = dataset.get_design_matrix(design_r_batch.copy())
        mapped_r_design = dataset.mapback(design_r_batch.copy())
        mapped_r_batch = dataset.get_topological_view(mapped_r_design.copy())
    for row in xrange(rows):
        row_start = cols * row
        for j in xrange(cols):
            vis_patch = vis_batch[row_start+j,:,:,:].copy()
            adjusted_vis_patch = dataset.adjust_for_viewer(vis_patch)
            if vis_patch.shape[-1] == 2:
                pv.add_patch(adjusted_vis_patch[:,:,1], rescale=False)
                pv.add_patch(adjusted_vis_patch[:,:,0], rescale=False)
            else:
                pv.add_patch(adjusted_vis_patch, rescale = False)
            r = vis_patch
            #print 'vis: '
            #for ch in xrange(3):
            #    chv = r[:,:,ch]
            #    print '\t',ch,(chv.min(),chv.mean(),chv.max())
            if mapback:
                pv.add_patch(dataset.adjust_for_viewer(
                    mapped_batch[row_start+j,:,:,:].copy()), rescale = False)
            if recons_batch.shape[-1] == 2:
                pv.add_patch(dataset.adjust_to_be_viewed_with(
                recons_batch[row_start+j,:,:,1].copy(),
                vis_patch), rescale = False)
                pv.add_patch(dataset.adjust_to_be_viewed_with(
                recons_batch[row_start+j,:,:,0].copy(),
                vis_patch), rescale = False)
            else:
                pv.add_patch(dataset.adjust_to_be_viewed_with(
                recons_batch[row_start+j,:,:,:].copy(),
                vis_patch), rescale = False)
            r = recons_batch[row_start+j,:,:,:]
            #print 'recons: '
            #for ch in xrange(3):
            #    chv = r[:,:,ch]
            #    print '\t',ch,(chv.min(),chv.mean(),chv.max())
            if mapback:
                pv.add_patch(dataset.adjust_to_be_viewed_with(
                    mapped_r_batch[row_start+j,:,:,:].copy(),
                    mapped_batch[row_start+j,:,:,:].copy()),rescale = False)
    pv.show()


if hasattr(grbm.visible_layer, 'beta'):
    beta = grbm.visible_layer.beta.get_value()
    #model.visible_layer.beta.set_value(beta * 100.)
    print('beta: ',(beta.min(), beta.mean(), beta.max()))

while True:
    show()
    print('Displaying reconstructions. (q to quit, ENTER = show more)')
    while True:
        x = input()
        if x == 'q':
            quit()
        if x == '':
            x = 1
            break
        else:
            print('Invalid input, try again')

    vis_batch = dataset.get_batch_topo(m)
