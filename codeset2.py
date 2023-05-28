from libtiff import TIFF, TIFFfile, TIFFimage
from matplotlib import pyplot as plt
import numpy as np
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

def hn(dataImageArray, r):
  if r == 0:
    return dataImageArray.size - np.count_nonzero(dataImageArray)
  else:
    return np.count_nonzero(np.where(dataImageArray == r, dataImageArray, 0))

def collectIntensitiesHistogramValues(dataImageArray):
  return [hn(dataImageArray, r) for r in range(0, MAX_VAL)]

def testCollectIntensitiesHistogramValues():
  tifImg = TIFF.open(IMG_SET_2_REL_PATH + 'Fig0310(b)(washed_out_pollen_image).tif', mode='r')
  data = tifImg.read_image()
  plt.hist(data, range=(0, MAX_VAL))
  plt.show()
  tifImg.close()

def pr(dataImageArray, r):
  return hn(dataImageArray, r) / dataImageArray.size

def histEqIntTranformation(dataImageArray, r):
  return np.floor(MAX_VAL * sum([pr(dataImageArray, rj) for rj in range(0, r)])).astype(np.uint8)

def constructlookupTable(dataImageArray, func):
  return [func(dataImageArray, x) for x in range(0, MAX_VAL)]

def testHistEqIntTranformation():
  tifImg = TIFF.open(IMG_SET_2_REL_PATH + 'Fig0310(b)(washed_out_pollen_image).tif', mode='r')
  data = tifImg.read_image()
  lookupTable = constructlookupTable(data, histEqIntTranformation)
  for i, row in enumerate(data):
    for j, x in enumerate(row):
      data[i, j] = lookupTable[x]
  plotImage(data)
  plt.hist(data, range=(0, MAX_VAL))
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
testHistEqIntTranformation()
#testCollectIntensitiesHistogramValues()