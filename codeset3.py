from libtiff import TIFF
import numpy as np
from matplotlib import pyplot as plt
from codeset1 import plotImage, MAX_VAL
from codeset2 import intensityPowerlaw

def spatialFilter(image, filter, convolution=False, mirror=False):
  out = np.copy(image)
  filterCenter = len(filter) // 2

  for img_i, row in enumerate(image):
    for img_j, x in enumerate(row):

      acc = 0
      for filterOffsetI in range(-filterCenter, filterCenter + 1):
        for filterOffsetJ in range(-filterCenter, filterCenter + 1):
          # calc eff coordinates
          i = img_i + filterOffsetI if not convolution else img_i - filterOffsetI
          j = img_j + filterOffsetJ if not convolution else img_j - filterOffsetJ
          # check if outside of an image
          if i < 0 or i >= image.shape[0]:
            if mirror:
              i = img_i - filterOffsetI if not convolution else img_i + filterOffsetI
            else:
              continue
          if j < 0 or j >= image.shape[1]:
            if mirror:
              j = img_j - filterOffsetJ if not convolution else img_j + filterOffsetJ
            else:
              continue
          acc += image[i, j] * filter[filterCenter + filterOffsetI, filterCenter + filterOffsetJ]

      out[img_i, img_j] = acc

  return out

def testSpatialFilter():
  filter = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ])

  image = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
  ])

  res = spatialFilter(image, filter, convolution=True)
  print(res)

def gaussian(x, y, sigma):
  return np.exp(-((x**2 + y**2) / (2*sigma**2)))

def generateGaussianFilter(n, sigma):
  if n % 2 == 0:
    raise "Filter side length must be odd!"
  res = np.full((n, n), 0, dtype=np.float64)
  filterCenter = n // 2

  for filterOffsetI in range(-filterCenter, filterCenter + 1):
    for filterOffsetJ in range(-filterCenter, filterCenter + 1):
      res[filterCenter + filterOffsetI][filterCenter + filterOffsetJ] = gaussian(filterOffsetI, filterOffsetJ, sigma)

  return res

def testGaussianFilter():
  tifImg = TIFF.open(IMG_SET_3_REL_PATH + 'Fig0333(a)(test_pattern_blurring_orig).tif', mode='r')
  data = tifImg.read_image()
  filter = generateGaussianFilter(21, 3.5)
  filter /= np.sum(filter)
  res = spatialFilter(data.astype(np.float64), filter, mirror=True)
  fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16,9))
  ax1.imshow(data, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.imshow(np.floor(res).astype(np.uint8), interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  plt.show()
  tifImg.close()

def sobelX():
  return np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
  ])

def sobelY():
  return np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
  ])

def sobelGradient(img):
  filterX = sobelX()
  filterY = sobelY()
  resX = spatialFilter(img, filterX, convolution=True)
  resY = spatialFilter(img, filterY, convolution=True)
  return (resX**2 + resY**2)**(1/2)

def testSobel():
  tifImg = TIFF.open(IMG_SET_3_REL_PATH + 'Fig0342(a)(contact_lens_original).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  sobelImg = sobelGradient(img)
  fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16,9))
  ax1.imshow(img, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.imshow(to8bit(sobelImg, mode='clip'), cmap='gray', vmin=0, vmax=MAX_VAL)
  plt.show()
  tifImg.close()

IMG_SET_3_REL_PATH = './datasets/imgset3/'

def blockAverageFilter(img, n):
  filter = np.full((n, n), 1/(n**2))
  return spatialFilter(img, filter, mirror=True)

def testBlockAveraging():
  tifImg = TIFF.open(IMG_SET_3_REL_PATH + 'Fig0333(a)(test_pattern_blurring_orig).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  res = blockAverageFilter(img, 5)
  fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16,9))
  ax1.imshow(img, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.imshow(np.floor(res).astype(np.uint8), interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  plt.show()
  tifImg.close()

def testMedianFilter():
  # TODO -> test it for 3x3 average filter too.
  tifImg = TIFF.open(IMG_SET_3_REL_PATH + 'Fig0335(a)(ckt_board_saltpep_prob_pt05).tif', mode='r')
  data = tifImg.read_image()
  filterSize = 3
  filter = np.full((filterSize, filterSize), 0)
  res = spatialFilter(data.astype(np.float64), filter)
  print(np.unique(res))
  fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16,9))
  ax1.imshow(data, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.imshow(np.floor(res).astype(np.uint8), interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  plt.show()
  tifImg.close()

def applyLaplacianN4(img):
  filterN4 = np.array([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
  ], dtype=np.float64)
  return spatialFilter(img, filterN4, convolution=True)

def applyLaplacianN8(img):
  filterN8 = np.array([
    [ 1,  1,  1],
    [ 1, -8,  1],
    [ 1,  1,  1]
  ], dtype=np.float64)
  return spatialFilter(img, filterN8, convolution=True)

def testLaplacian():
  tifImg = TIFF.open(IMG_SET_3_REL_PATH + 'Fig0338(a)(blurry_moon).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  laplacian4img = applyLaplacianN4(img)
  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,9))
  ax1.imshow(img, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(to8bit(laplacian4img, mode='scale'), cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(to8bit(img - laplacian4img, mode='clip'), cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  laplacian8img = applyLaplacianN8(img)
  ax4.imshow(img, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax5.imshow(to8bit(laplacian8img, mode='scale'), cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  ax6.imshow(to8bit(img - laplacian8img, mode='clip'), cmap='gray', vmin=0, vmax=MAX_VAL)
  ax6.set_axis_off()
  plt.tight_layout()
  plt.show()
  tifImg.close()

def to8bit(img, mode='clip'):
  if mode == 'clip':
    res = np.where(img < 0, 0, img)
    res = np.where(res > MAX_VAL, MAX_VAL, res)
    return np.floor(res).astype(np.uint8)
  if mode == 'scale':
    res = np.interp(img, (np.min(img), np.max(img)), (0, MAX_VAL))
    return np.floor(res).astype(np.uint8)
  if mode == '8bitadd':
    return np.floor(np.interp(img, (0, MAX_VAL*2), (0, MAX_VAL))).astype(np.uint8)
  return img

def clip(data):
  res = np.where(data < 0, 0, data)
  res = np.where(res > MAX_VAL, MAX_VAL, res)
  return np.floor(res).astype(np.uint8)

def testHighboost():
  tifImg = TIFF.open(IMG_SET_3_REL_PATH + 'Fig0340(a)(dipxe_text).tif', mode='r')
  imgOriginal = tifImg.read_image().astype(np.float64)
  gaussianFilter = generateGaussianFilter(15, 3)
  gaussianFilter /= np.sum(gaussianFilter)
  imgBlurred = spatialFilter(imgOriginal, gaussianFilter, mirror=True)
  mask = imgOriginal - imgBlurred
  maskDisplay = np.floor(np.interp(mask, (-MAX_VAL, MAX_VAL), (0, MAX_VAL))).astype(np.uint8)
  unsharp = clip(imgOriginal + mask)
  highboost = clip(imgOriginal + 4*mask)
  fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16,9))
  ax1.imshow(imgOriginal, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(np.floor(imgBlurred), cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(maskDisplay, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax4.imshow(unsharp, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax5.imshow(highboost, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  plt.show()
  tifImg.close()

def testSkeleton():
  tifImg = TIFF.open(IMG_SET_3_REL_PATH + 'Fig0343(a)(skeleton_orig).tif', mode='r')
  imgOriginal = tifImg.read_image().astype(np.float64)
  laplacian4img = applyLaplacianN4(imgOriginal)
  sharpened = imgOriginal - laplacian4img
  sobelImg = sobelGradient(imgOriginal)
  sobelImg = to8bit(sobelImg, mode='clip')
  sobelAvgImg = blockAverageFilter(sobelImg, 5)
  sobelAvgImg = to8bit(sobelAvgImg, mode='scale')
  mask = to8bit(sobelAvgImg * sharpened, mode='scale')
  sharpaf = to8bit(imgOriginal + mask.astype(np.float64), mode='clip')
  powerLawImg = intensityPowerlaw(sharpaf.astype(np.float64), 0.5)
  fig, ((ax, bx, cx, dx), (ex, fx, gx, hx)) = plt.subplots(2, 4, figsize=(16,9))
  ax.imshow(imgOriginal.astype(np.uint8), cmap='gray', vmin=0, vmax=MAX_VAL)
  ax.set_axis_off()
  bx.imshow(to8bit(laplacian4img, mode='scale'), cmap='gray', vmin=0, vmax=MAX_VAL)
  bx.set_axis_off()
  cx.imshow(to8bit(sharpened, mode='scale'), cmap='gray', vmin=0, vmax=MAX_VAL)
  cx.set_axis_off()
  dx.imshow(to8bit(sobelImg, mode='clip'), cmap='gray', vmin=0, vmax=MAX_VAL)
  dx.set_axis_off()
  ex.imshow(sobelAvgImg, cmap='gray', vmin=0, vmax=MAX_VAL)
  ex.set_axis_off()
  fx.imshow(mask, cmap='gray', vmin=0, vmax=MAX_VAL)
  fx.set_axis_off()
  gx.imshow(sharpaf, cmap='gray', vmin=0, vmax=MAX_VAL)
  gx.set_axis_off()
  hx.imshow(powerLawImg, cmap='gray', vmin=0, vmax=MAX_VAL)
  hx.set_axis_off()
  plt.tight_layout()
  plt.show()
  tifImg.close()

#testSpatialFilter()
#testBlockAveraging()
#testMedianFilter()
#testLaplacian()
#testGaussianFilter()
#testSobel()
#testHighboost()
#testSkeleton()