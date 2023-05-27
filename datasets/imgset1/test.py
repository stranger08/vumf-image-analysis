from libtiff import TIFF

filename = 'Kidney1.tif'
tifImg = TIFF.open(filename)

imageWidth = tifImg.GetField('Image Width')
print('Image Width', imageWidth)

tifImg.close()