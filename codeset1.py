from libtiff import TIFF, TIFFfile, TIFFimage
from matplotlib import pyplot as plt
import numpy as np
import math

tagNames = [
# baseline tag names
  'SUBFILETYPE',
  'OSUBFILETYPE',
  'IMAGEWIDTH',
  'IMAGELENGTH',
  'BITSPERSAMPLE',
  'COMPRESSION',
  'PHOTOMETRIC',
  'THRESHHOLDING',
  'CELLWIDTH',
  'CELLLENGTH',
  'FILLORDER',
  'IMAGEDESCRIPTION',
  'MAKE',
  'MODEL',
  'STRIPOFFSETS',
  'ORIENTATION',
  'SAMPLESPERPIXEL',
  'ROWSPERSTRIP',
  'STRIPBYTECOUNTS',
  'MINSAMPLEVALUE',
  'MAXSAMPLEVALUE',
  'XRESOLUTION',
  'YRESOLUTION',
  'PLANARCONFIG',
  'FREEOFFSETS',
  'FREEBYTECOUNTS',
  'GRAYRESPONSEUNIT',
  'GRAYRESPONSECURVE',
  'RESOLUTIONUNIT',
  'SOFTWARE',
  'DATETIME',
  'ARTIST',
  'HOSTCOMPUTER',
  'COLORMAP',
  'COPYRIGHT',
# extension tag names
  'DOCUMENTNAME',
  'PAGENAME',
  'XPOSITION',
  'YPOSITION',
  'T4OPTIONS',
  'T6OPTIONS',
  'PAGENUMBER',
  'TRANSFERFUNCTION',
  'PREDICTOR',
  'WHITEPOINT',
  'PRIMARYCHROMATICITIES',
  'HALFTONEHINTS',
  'TILEWIDTH',
  'TILELENGTH',
  'TILEOFFSETS',
  'TILEBYTECOUNTS',
  'BADFAXLINES',
  'CLEANFAXDATA',
  'CONSECUTIVEBADFAXLINES',
  'SUBIFD',
  'INKSET',
  'INKNAMES',
  'NUMBEROFINKS',
  'DOTRANGE',
  'TARGETPRINTER',
  'EXTRASAMPLES',
  'SAMPLEFORMAT',
  'SMINSAMPLEVALUE',
  'SMAXSAMPLEVALUE',
  'CLIPPATH',
  'XCLIPPATHUNITS',
  'YCLIPPATHUNITS',
  'INDEXED',
  'JPEGTABLES',
  'OPIPROXY',
  'GLOBALPARAMETERSIFD',
  'PROFILETYPE',
  'FAXPROFILE',
  'CODINGMETHODS',
  'VERSIONYEAR',
  'MODENUMBER',
  'DECODE',
  'IMAGEBASECOLOR',
  'T82OPTIONS',
  'JPEGPROC',
  'JPEGIFOFFSET',
  'JPEGIFBYTECOUNT',
  'JPEGRESTARTINTERVAL',
  'JPEGLOSSLESSPREDICTORS',
  'JPEGPOINTTRANSFORM',
  'JPEGQTABLES',
  'JPEGDCTABLES',
  'JPEGACTABLES',
  'YCBCRCOEFFICIENTS',
  'YCBCRSUBSAMPLING',
  'YCBCRPOSITIONING',
  'REFERENCEBLACKWHITE',
  'STRIPROWCOUNTS',
  'XMLPACKET',
  'OPIIMAGEID',
  'IMAGELAYER'
]

def printHeader(tifImg):
  print('-- -- --')
  for tagName in tagNames:
    tagValue = tifImg.GetField(tagName)
    if tagValue:
      print(tagName, tagValue)

def iterHeaders(tifImg):
  printHeader(tifImg)
  while not tifImg.LastDirectory():
    tifImg.ReadDirectory()
    printHeader(tifImg)
  tifImg.SetDirectory(0)

MAX_VAL = 255

grayscaleSideways = np.array([[MAX_VAL - x for x in range(MAX_VAL)] for y in range(MAX_VAL)], dtype=np.uint8)

grayscaleDiagonal = np.array([[MAX_VAL-(x//2+y//2) for x in range(MAX_VAL)] for y in range(MAX_VAL)], dtype=np.uint8)

def circle():
  C_MAX_VAL = 1024
  circle = np.zeros((C_MAX_VAL, C_MAX_VAL), dtype=np.uint8) + 255
  RADIUS = 255
  center = C_MAX_VAL // 2
  for i, row in enumerate(circle):
    for j, col in enumerate(row):
      if (i - center)**2 + (j - center)**2 < RADIUS**2:
        circle[i][j] = MAX_VAL - abs(math.sqrt((i - center)**2 + (j - center)**2))
  return circle
  
IMG_SET_REL_PATH = './datasets/imgset1/'
OUT_DIR_REL_PATH = './datasets/outimgset1/'
  
def saveImageToFile(imageDataArray, filename, isRBG):
  tifImg = TIFF.open(OUT_DIR_REL_PATH + filename, mode='w')
  tifImg.write_image(imageDataArray, write_rgb=isRBG)
  tifImg.close()

def plotImage(imageDataArray):
  plt.imshow(imageDataArray, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  plt.show()
  
def imageConstruction():
  #plotImage(grayscaleSideways)
  #plotImage(grayscaleDiagonal)
  saveImageToFile(grayscaleSideways, 'grayscaleSideways.tif', False)
  saveImageToFile(grayscaleDiagonal, 'grayscaleDiagonal.tif', False)
  circle = circle()
  #plotImage(circle)
  saveImageToFile(circle, 'circle.tif', False)

def combine():
  tifImgAcridine = TIFF.open(IMG_SET_REL_PATH + 'Region_001_FOV_00041_Acridine_Or_Gray.tif')
  tifImgDAPI = TIFF.open(IMG_SET_REL_PATH + 'Region_001_FOV_00041_DAPI_Gray.tif')
  tifImgFITC = TIFF.open(IMG_SET_REL_PATH + 'Region_001_FOV_00041_FITC_Gray.tif')
  dataAcridine = tifImgAcridine.read_image()
  dataDAPI = tifImgDAPI.read_image()
  dataFITC = tifImgFITC.read_image()
  combined = np.flip(np.stack([dataAcridine, dataDAPI, dataFITC], axis=-1, dtype=np.uint8), axis=0)
  plt.imshow(combined, interpolation='nearest')
  plt.show()
  saveImageToFile(combined, 'Region_001_FOV_00041_combined.tif', True)

combine()

