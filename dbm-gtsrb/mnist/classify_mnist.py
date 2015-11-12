import numpy
from PIL import Image
from dbnsignsdetection_pl2 import classify

reverse_colors = 1 # 1 to reverse colors, 0 otherwise

model = 'dbm'

model_path = "/home/carlo/workspace/dbnsignsdetection_pl2/dbnsignsdetection_pl2/mnist/dbm/dbm_mnist.pkl"

test_path = "/home/carlo/workspace/dbnsignsdetection_pl2/dbnsignsdetection_pl2/mnist/digits/prova-1-bis.png"

img = Image.open(test_path)
img = img.getdata()
img = img.convert('L')
x = numpy.asarray([img], numpy.float32)
if reverse_colors:
    x = 255. - x
x /= 255.

classify.predict(model, model_path, x)