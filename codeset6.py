import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt
from codeset1 import MAX_VAL
from codeset2 import intensityPowerlaw, thresholdingValue
from codeset3 import to8bit, applyLaplacianN8

def runStructuralElement(image, se, mode='erosion'):
  out = np.copy(image)
  seCenter = len(se) // 2

  for img_i, row in enumerate(image):
    for img_j, x in enumerate(row):

      if mode == 'erosion':
        out[img_i, img_j] = 1
      if mode == 'dilation':
        out[img_i, img_j] = 0
      for seOffsetI in range(-seCenter, seCenter + 1):
        for seOffsetJ in range(-seCenter, seCenter + 1):
          # calc eff coordinates
          i = img_i + seOffsetI
          j = img_j + seOffsetJ
          sei = seCenter + seOffsetI
          sej = seCenter + seOffsetJ
          # check if outside of an image
          if i < 0 or i >= image.shape[0]:
            continue
          if j < 0 or j >= image.shape[1]:
            continue

          if mode == 'erosion' and se[sei, sej] != 0 and image[i, j] == 0:
            out[img_i, img_j] = 0
          
          if mode == 'dilation' and se[sei, sej] != 0 and image[i, j] != 0:
            out[img_i, img_j] = 1

  return out

image = np.array([
  [0, 0, 0, 0, 0, 0], 
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0],
  [1, 1, 1, 0, 0, 0],
  [0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0]
])

SE = np.array([
  [0, 1, 0],
  [1, 1, 1],
  [0, 1, 0]
])

IMG_SET_6_REL_PATH = './datasets/imgset6/'

def test():
  tifImg = TIFF.open(IMG_SET_6_REL_PATH + 'Fig0914(a)(licoln from penny)-8bit.tiff', mode='r')
  img = tifImg.read_image().astype(np.float64)
  fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16,9))
  ax1.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  
  erosionR = runStructuralElement(img, SE, mode='erosion')
  outsideBorder = img - to8bit(erosionR, mode='scale')
  ax2.imshow(outsideBorder, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  plt.show()
  tifImg.close()

def cclN4(img):
  lm = np.reshape(np.arange(0, img.size), img.shape).astype(np.uint32)
  while True:
    hit = False
    for i, row in enumerate(image):
      for j, _ in enumerate(row):
        if i > 0 and img[i - 1, j] == img[i, j] and lm[i, j] < lm[i - 1, j]:
          lm[i, j] = lm[i - 1, j]
          hit = True
        if i < img.shape[0] - 1 and img[i + 1, j] == img[i, j] and lm[i, j] < lm[i + 1, j]:
          lm[i, j] = lm[i + 1, j]
          hit = True
        if j > 0 and img[i, j - 1] == img[i, j] and lm[i, j] < lm[i, j - 1]:
          lm[i, j] = lm[i, j - 1]
          hit = True
        if j < img.shape[1] - 1 and img[i, j + 1] == img[i, j] and lm[i, j] < lm[i, j + 1]:
          lm[i, j] = lm[i, j + 1]
          hit = True
    if not hit:
      break
  return lm
