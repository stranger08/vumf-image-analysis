
from libtiff import TIFF
import numpy as np
from matplotlib import pyplot as plt
from codeset1 import MAX_VAL
from codeset3 import to8bit, blockAverageFilter

IMG_SET_5_REL_PATH = './datasets/imgset5/'

def geometricMeanFilter(img, m, n, mirror=False):
  out = np.copy(img)
  mCenter = m // 2
  nCenter = n // 2

  for img_i, row in enumerate(img):
    for img_j, x in enumerate(row):

      acc = 1
      for filterOffsetI in range(-mCenter, mCenter + 1):
        for filterOffsetJ in range(-nCenter, nCenter + 1):
          # calc eff coordinates
          i = img_i + filterOffsetI
          j = img_j + filterOffsetJ
          # check if outside of an image
          if i < 0 or i >= img.shape[0]:
            if mirror:
              i = img_i - filterOffsetI
            else:
              continue
          if j < 0 or j >= img.shape[1]:
            if mirror:
              j = img_j - filterOffsetJ
            else:
              continue
          acc *= img[i, j]

      out[img_i, img_j] = acc**(1/(m*n))
  return out

def testGaussianNoiseRemoval():
  tifImg = TIFF.open(IMG_SET_5_REL_PATH + 'Fig0507(a)(ckt-board-orig).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)

  tifImgGN = TIFF.open(IMG_SET_5_REL_PATH + 'Fig0507(b)(ckt-board-gauss-var-400).tif', mode='r')
  imgGN = tifImgGN.read_image().astype(np.float64)

  imgAMF = blockAverageFilter(imgGN, 3)
  imgAMF8 = to8bit(imgAMF, mode='clip')

  imgGMF = geometricMeanFilter(imgGN, 3, 3, mirror=True)
  imgGMF8 = to8bit(imgGMF, mode='scale')

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9))
  ax1.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax1.set_title('Original')
  ax2.imshow(imgGN, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax2.set_title('Gaussian noise')
  ax3.imshow(imgAMF8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax3.set_title('Arithmetic mean filter 3x3')
  ax4.imshow(imgGMF8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax4.set_title('Geometric mean filter')
  plt.tight_layout()
  plt.show()
  tifImg.close()

testGaussianNoiseRemoval()

def testSaltNPepperNoiseRemoval():
  pass