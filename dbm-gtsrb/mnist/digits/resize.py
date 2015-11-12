from PIL import Image

path = "/prova-1-jacopo.jpg"
size = [28, 28]
name = "/prova-1-jacopo.png"

img = Image.open("/home/carlo/workspace/dbnsignsdetection_pl2/dbnsignsdetection_pl2/mnist/digits" + path)

img = img.resize(size, Image.ANTIALIAS)

img.save("/home/carlo/workspace/dbnsignsdetection_pl2/dbnsignsdetection_pl2/mnist/digits" + name)
