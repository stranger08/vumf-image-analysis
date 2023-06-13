from libtiff import TIFF
import numpy as np
from matplotlib import pyplot as plt

def testDFT():
  tifImg = TIFF.open(IMG_SET_3_REL_PATH + 'Fig0342(a)(contact_lens_original).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  fig, ((ax1)) = plt.subplots(1, 1, figsize=(16,9))
  ax1.imshow(img, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  plt.show()
  tifImg.close()