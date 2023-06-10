from libtiff import TIFF, TIFFfile, TIFFimage
from matplotlib import pyplot as plt
import numpy as np
from functools import partial
#
# TODO add test for lookup tables for greyscale and negate transformations
# TODO add four images when displaying results for histogram equaling method:
#       : original image and its hist
#       : normalized image and its hist
#
from codeset1 import MAX_VAL, plotImage, circle

IMG_SET_2_REL_PATH = './datasets/imgset2/'

def duplicate(imageDataArray):
  return np.copy(imageDataArray)

def testAssignReference():
  img1 = circle()
  img2 = img1
  img1[:,:] = 0
  plotImage(img1)
  plotImage(img2)

def testDuplication():
  img1 = circle()
  img2 = duplicate(img1)
  img1[:,:] = 0
  plotImage(img1)
  plotImage(img2)

def intensityNegationGrayscale(imageDataArray):
  return MAX_VAL - imageDataArray;

def testIntensityNegationGrayscale():
  tifImg = TIFF.open(IMG_SET_2_REL_PATH + 'Fig0304(a)(breast_digital_Xray).tif', mode='r')
  data = tifImg.read_image()
  plotImage(intensityNegationGrayscale(data))
  plotImage(data)
  tifImg.close()

def intensityPowerlaw(imageDataArray, gamma):
  p = np.float_power(imageDataArray, gamma)
  pbyte = np.interp(p, (0, MAX_VAL**gamma), (0, MAX_VAL))
  return np.floor(pbyte).astype(np.uint8)

def testIntensityPowerlawUp():
  tifImg = TIFF.open(IMG_SET_2_REL_PATH + 'Fig0309(a)(washed_out_aerial_image).tif', mode='r')
  data = tifImg.read_image()
  plotImage(data)
  plotImage(intensityPowerlaw(data, 5))
  tifImg.close()

def testIntensityPowerlawDown():
  tifImg = TIFF.open(IMG_SET_2_REL_PATH + 'Fig0308(a)(fractured_spine).tif', mode='r')
  data = tifImg.read_image()
  plotImage(data)
  plotImage(intensityPowerlaw(data, 0.3))
  tifImg.close()

def thresholdingValue(arr, t):
  return np.array([[0 if x > t else 255 for x in row] for row in arr])

def thresholdingRange(arr, lower, higher):
  return np.array([[0 if x < lower or x > higher else 255 for x in row] for row in arr])

def testThreshholdingValue():
  tifImg = TIFF.open(IMG_SET_2_REL_PATH + 'Fig0310(b)(washed_out_pollen_image).tif', mode='r')
  data = tifImg.read_image()
  plotImage(data)
  plotImage(thresholdingValue(data, 110))
  tifImg.close()

def testThreshholdingRange():
  tifImg = TIFF.open(IMG_SET_2_REL_PATH + 'Fig0310(b)(washed_out_pollen_image).tif', mode='r')
  data = tifImg.read_image()
  plotImage(data)
  plotImage(thresholdingRange(data, 0, 100))
  tifImg.close()

class PltInterval:
  left_x = 0
  right_x = 0
  left_col = 0
  right_col = 0

  def __init__(self, lx, rx, lc, rc):
    self.left_x = lx
    self.right_x = rx
    self.left_col = lc
    self.right_col = rc

  def isApplicable(self, x):
    return x > self.left_x and x <= self.right_x

  def value(self, x):
    return np.interp(x, (self.left_x, self.right_x), (self.left_col, self.right_col))

def testIntervalInterpolation():
  PLT = [
    PltInterval(0, 96, 0, 32),
    PltInterval(96, 160, 32, 224),
    PltInterval(160, 255, 224, 255)
  ]
  tifImg = TIFF.open(IMG_SET_2_REL_PATH + 'Fig0310(b)(washed_out_pollen_image).tif', mode='r')
  data = tifImg.read_image()
  for i, row in enumerate(data):
    for j, x in enumerate(row):
      for p in PLT:
        if p.isApplicable(x):
          data[i, j] = p.value(x)
  plotImage(data)
  print(data[0:10,0:10])
  tifImg.close()

#
def hist(imageArr, intensity):
  return (imageArr == intensity).sum()

#
def pdf(histLookup, resolution, intensity):
  return histLookup[intensity] / resolution

#
def histEq(pdfLookup, i):
  s = (MAX_VAL - 1) * sum([pdfLookup[ij] for ij in range(0, i)])
  return np.floor(s).astype(np.uint8)

#
def collectLookup(pFunc):
  return [pFunc(i) for i in range(0, MAX_VAL)]

def testHistEq():
  tifImg = TIFF.open(IMG_SET_2_REL_PATH + 'Fig0310(b)(washed_out_pollen_image).tif', mode='r')
  data = tifImg.read_image()
  dataOrigin = duplicate(data)

  pHist = partial(hist, data)
  histLookup = collectLookup(pHist)

  pPdf = partial(pdf, histLookup, data.size)
  pdfLookup = collectLookup(pPdf)

  pHistEq = partial(histEq, pdfLookup)
  histEqLt = collectLookup(pHistEq)

  converted = np.array([[histEqLt[i] for i in row] for row in data])

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9))
  fig.suptitle('Sharing x per column, y per row')
  ax1.imshow(dataOrigin, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.hist(dataOrigin.flatten(), range=(0, MAX_VAL), bins=MAX_VAL)
  ax3.imshow(converted, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.hist(converted.flatten(), range=(0, MAX_VAL), bins=MAX_VAL)
  plt.show()
  tifImg.close()

#testAssignReference()
#testDuplication()
#testIntensityNegationGrayscale()
#testIntensityPowerlawUp()
#testIntensityPowerlawDown()
#testThreshholdingValue()
#testThreshholdingRange()
#testIntervalInterpolation()
#testHistEq()