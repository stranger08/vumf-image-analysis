
from libtiff import TIFF
import numpy as np
import math
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
          acc *= img[i, j] if img[i, j] != 0 else 1

      out[img_i, img_j] = acc**(1/(m*n))
  return out

def contraharmonicMeanFilter(img, m, n, q, mirror=False):
  out = np.copy(img)
  mCenter = m // 2
  nCenter = n // 2

  for img_i, row in enumerate(img):
    for img_j, x in enumerate(row):

      accQ0 = 0
      accQ1 = 0
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
          accQ0 += np.power(img[i, j], q) if img[i, j] != 0 else 1
          accQ1 += np.power(img[i, j], q + 1) if img[i, j] != 0 else 1

      if accQ0 != 0:
        out[img_i, img_j] = accQ1 / accQ0
  return out

def medianFilter(image, m, n, mirror=False):
  out = np.copy(image)
  mCenter = m // 2
  nCenter = n // 2

  for img_i, row in enumerate(image):
    for img_j, x in enumerate(row):
      mask = []
      for filterOffsetI in range(-mCenter, mCenter + 1):
        for filterOffsetJ in range(-nCenter, nCenter + 1):
          # calc eff coordinates
          i = img_i + filterOffsetI
          j = img_j + filterOffsetJ
          # check if outside of an image
          if i < 0 or i >= image.shape[0]:
            if mirror:
              i = img_i - filterOffsetI
            else:
              continue
          if j < 0 or j >= image.shape[1]:
            if mirror:
              j = img_j - filterOffsetJ
            else:
              continue
          mask.append(image[i, j])

      out[img_i, img_j] = np.median(np.array(mask))

  return out

def minFilter(image, m, n, mirror=False):
  out = np.copy(image)
  mCenter = m // 2
  nCenter = n // 2

  for img_i, row in enumerate(image):
    for img_j, x in enumerate(row):
      mask = []
      for filterOffsetI in range(-mCenter, mCenter + 1):
        for filterOffsetJ in range(-nCenter, nCenter + 1):
          # calc eff coordinates
          i = img_i + filterOffsetI
          j = img_j + filterOffsetJ
          # check if outside of an image
          if i < 0 or i >= image.shape[0]:
            if mirror:
              i = img_i - filterOffsetI
            else:
              continue
          if j < 0 or j >= image.shape[1]:
            if mirror:
              j = img_j - filterOffsetJ
            else:
              continue
          mask.append(image[i, j])

      out[img_i, img_j] = np.min(np.array(mask))

  return out

def maxFilter(image, m, n, mirror=False):
  out = np.copy(image)
  mCenter = m // 2
  nCenter = n // 2

  for img_i, row in enumerate(image):
    for img_j, x in enumerate(row):
      mask = []
      for filterOffsetI in range(-mCenter, mCenter + 1):
        for filterOffsetJ in range(-nCenter, nCenter + 1):
          # calc eff coordinates
          i = img_i + filterOffsetI
          j = img_j + filterOffsetJ
          # check if outside of an image
          if i < 0 or i >= image.shape[0]:
            if mirror:
              i = img_i - filterOffsetI
            else:
              continue
          if j < 0 or j >= image.shape[1]:
            if mirror:
              j = img_j - filterOffsetJ
            else:
              continue
          mask.append(image[i, j])

      out[img_i, img_j] = np.max(np.array(mask))

  return out

def midPointFilter(image, m, n, mirror=False):
  out = np.copy(image)
  mCenter = m // 2
  nCenter = n // 2

  for img_i, row in enumerate(image):
    for img_j, x in enumerate(row):
      mask = []
      for filterOffsetI in range(-mCenter, mCenter + 1):
        for filterOffsetJ in range(-nCenter, nCenter + 1):
          # calc eff coordinates
          i = img_i + filterOffsetI
          j = img_j + filterOffsetJ
          # check if outside of an image
          if i < 0 or i >= image.shape[0]:
            if mirror:
              i = img_i - filterOffsetI
            else:
              continue
          if j < 0 or j >= image.shape[1]:
            if mirror:
              j = img_j - filterOffsetJ
            else:
              continue
          mask.append(image[i, j])

      out[img_i, img_j] = np.floor(abs(np.max(np.array(mask)) + np.min(np.array(mask))) / 2)

  return out

def alphaTrimmedMeanFilter(image, m, n, d, mirror=False):
  out = np.copy(image)
  mCenter = m // 2
  nCenter = n // 2

  for img_i, row in enumerate(image):
    for img_j, x in enumerate(row):
      mask = []
      for filterOffsetI in range(-mCenter, mCenter + 1):
        for filterOffsetJ in range(-nCenter, nCenter + 1):
          # calc eff coordinates
          i = img_i + filterOffsetI
          j = img_j + filterOffsetJ
          # check if outside of an image
          if i < 0 or i >= image.shape[0]:
            if mirror:
              i = img_i - filterOffsetI
            else:
              continue
          if j < 0 or j >= image.shape[1]:
            if mirror:
              j = img_j - filterOffsetJ
            else:
              continue
          mask.append(image[i, j])

      out[img_i, img_j] = (1/(m*n - d)) * sum(mask)

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

#testGaussianNoiseRemoval()

def testSaltNPepperNoiseRemoval():
  tifImgPepperNoise = TIFF.open(IMG_SET_5_REL_PATH + 'Fig0508(a)(circuit-board-pepper-prob-pt1).tif', mode='r')
  imgPep = tifImgPepperNoise.read_image().astype(np.float64)

  tifImgSaltNoise = TIFF.open(IMG_SET_5_REL_PATH + 'Fig0508(b)(circuit-board-salt-prob-pt1).tif', mode='r')
  imgSalt = tifImgSaltNoise.read_image().astype(np.float64)

  imgPepRemoved = contraharmonicMeanFilter(imgPep, 3, 3, 1.5, mirror=False)
  imgPepRemoved8 = to8bit(imgPepRemoved, mode='scale')

  imgSaltRemoved = contraharmonicMeanFilter(imgSalt, 3, 3, -1.5, mirror=False)
  imgSaltRemoved8 = to8bit(imgSaltRemoved, mode='scale')

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9))
  ax1.imshow(imgPep, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax1.set_title('Pepper noise')
  ax2.imshow(imgSalt, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax2.set_title('Salt noise')
  ax3.imshow(imgPepRemoved8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax3.set_title('Contraharmonic Q=1.5 on above')
  ax4.imshow(imgSaltRemoved8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax4.set_title('Contraharmonic Q=-1.5 on above')
  plt.tight_layout()
  plt.show()
  tifImgPepperNoise.close()
  tifImgSaltNoise.close()

#testSaltNPepperNoiseRemoval()

def testMedianFilterIterative():
  tifImgSaltPepperNoise = TIFF.open(IMG_SET_5_REL_PATH + 'Fig0510(a)(ckt-board-saltpep-prob.pt05).tif', mode='r')
  imgSaltPep = tifImgSaltPepperNoise.read_image().astype(np.float64)

  imgMedian1 = medianFilter(imgSaltPep, 3, 3, mirror=False)
  imgMedian18 = to8bit(imgMedian1, mode='clip')

  imgMedian2 = medianFilter(imgMedian1, 3, 3, mirror=False)
  imgMedian28 = to8bit(imgMedian2, mode='clip')

  imgMedian3 = medianFilter(imgMedian2, 3, 3, mirror=False)
  imgMedian38 = to8bit(imgMedian3, mode='clip')

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9))
  ax1.imshow(imgSaltPep, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax1.set_title('Salt and Pepper noise')
  ax2.imshow(imgMedian18, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax2.set_title('Median 3x3 1st iteration')
  ax3.imshow(imgMedian28, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax3.set_title('Median 3x3 2nd iteration')
  ax4.imshow(imgMedian38, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax4.set_title('Median 3x3 3rd iteration')
  plt.tight_layout()
  plt.show()
  tifImgSaltPepperNoise.close()

#testMedianFilterIterative()

def testSaltNPepperNoiseRemovalByMaxMinFilters():
  tifImgPepperNoise = TIFF.open(IMG_SET_5_REL_PATH + 'Fig0508(a)(circuit-board-pepper-prob-pt1).tif', mode='r')
  imgPep = tifImgPepperNoise.read_image().astype(np.float64)

  tifImgSaltNoise = TIFF.open(IMG_SET_5_REL_PATH + 'Fig0508(b)(circuit-board-salt-prob-pt1).tif', mode='r')
  imgSalt = tifImgSaltNoise.read_image().astype(np.float64)

  imgPepRemoved = maxFilter(imgPep, 3, 3, mirror=False)
  imgPepRemoved8 = to8bit(imgPepRemoved, mode='clip')

  imgSaltRemoved = minFilter(imgSalt, 3, 3, mirror=False)
  imgSaltRemoved8 = to8bit(imgSaltRemoved, mode='clip')

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9))
  ax1.imshow(imgPep, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax1.set_title('Pepper noise')
  ax2.imshow(imgSalt, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax2.set_title('Salt noise')
  ax3.imshow(imgPepRemoved8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax3.set_title('Contraharmonic Q=1.5 on above')
  ax4.imshow(imgSaltRemoved8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax4.set_title('Contraharmonic Q=-1.5 on above')
  plt.tight_layout()
  plt.show()
  tifImgPepperNoise.close()
  tifImgSaltNoise.close()

#testSaltNPepperNoiseRemovalByMaxMinFilters()

def testUniformSaltPepperRestoration():
  tifImgUniformNoise = TIFF.open(IMG_SET_5_REL_PATH + 'Fig0512(a)(ckt-uniform-var-800).tif', mode='r')
  imgUnf = tifImgUniformNoise.read_image().astype(np.float64)

  tifImgAddSaltPepper = TIFF.open(IMG_SET_5_REL_PATH + 'Fig0512(b)(ckt-uniform-plus-saltpepr-prob-pt1).tif', mode='r')
  imgUSP = tifImgAddSaltPepper.read_image().astype(np.float64)

  imgAMF = blockAverageFilter(imgUSP, 5)
  imgAMF8 = to8bit(imgAMF, mode='clip')

  imgGMF = geometricMeanFilter(imgUSP, 5, 5, mirror=False)
  imgGMF8 = to8bit(imgGMF, mode='clip')

  imgMedian1 = medianFilter(imgUSP, 5, 5, mirror=False)
  imgMedian18 = to8bit(imgMedian1, mode='clip')

  imgAlphaTrimmed1 = alphaTrimmedMeanFilter(imgUSP, 5, 5, 5, mirror=False)
  imgAlphaTrimmed18 = to8bit(imgAlphaTrimmed1, mode='clip')

  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,9))
  ax1.imshow(imgUnf, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax1.set_title('Uniform noise')

  ax2.imshow(imgUSP, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax2.set_title('Additional salt n pepper')

  ax3.imshow(imgAMF8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax3.set_title('Arithmetic 5x5 mean filter')

  ax4.imshow(imgGMF8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax4.set_title('Geometric 5x5 mean filter')

  ax5.imshow(imgMedian18, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  ax5.set_title('Median filter')

  ax6.imshow(imgAlphaTrimmed18, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax6.set_axis_off()
  ax6.set_title('Alpha-trimmed mean filter d=5')
  plt.tight_layout()
  plt.show()
  tifImgUniformNoise.close()
  tifImgAddSaltPepper.close()

#testUniformSaltPepperRestoration()

def IBRF(img, C, W):
  out = np.copy(img)
  center = len(out[0]) // 2
  for i, row in enumerate(out):
    for j, col in enumerate(row):
      D = math.sqrt((i - center)**2 + (j - center)**2)
      if C - W / 2 <= D and D <= C + W / 2:
        out[i][j] = 0
      else:
        out[i][j] = 1
  return out

def GBRF(img, C, W):
  out = np.copy(img)
  center = len(out[0]) // 2
  for i, row in enumerate(out):
    for j, col in enumerate(row):
      D = math.sqrt((i - center)**2 + (j - center)**2)
      if D != 0:
        out[i][j] = 1 - np.exp(-(abs((D**2 - C**2)/(D*W))**2))
      else:
        out[i][j] = 1
  return out

def BBRF(img, C, W, n):
  out = np.copy(img)
  center = len(out[0]) // 2
  for i, row in enumerate(out):
    for j, col in enumerate(row):
      D = ((i - center)**2 + (j - center)**2)**(1/2)
      denominator = D**2 - C**2
      if denominator != 0:
        out[i][j] = 1 / (1 + (abs((D*W) / denominator))**(2*n))
      else:
        out[i][j] = 1 / (1 + (abs((D*W)))**(2*n))
  return out

def testBandFilters():
  img = np.full((512, 512), 0).astype(np.float64)
  img8 = to8bit(img, mode='clip')

  imgIBRF = IBRF(img, 128, 60)
  imgIBRF8 = to8bit(imgIBRF, mode='scale')

  imgGBRF = GBRF(img, 128, 60)
  imgGBRF8 = to8bit(imgGBRF, mode='scale')

  imgBBRF = BBRF(img, 128, 60, 1)
  imgBBRF8 = to8bit(imgBBRF, mode='scale')

  fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(16,9))
  ax1.imshow(imgIBRF8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(imgGBRF8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(imgBBRF8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  plt.tight_layout()
  plt.show()

testBandFilters()