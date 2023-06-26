import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt
from codeset1 import MAX_VAL, combine
from codeset2 import intensityPowerlaw, thresholdingValue
from codeset3 import to8bit, applyLaplacianN8
from codeset6 import cclN4, runStructuralElement, SEL, getComponentsBox

def testCells():
  tifImg = TIFF.open('./data/BXY-ABCD_Region 002_FOV 00040_DAPI.tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  imgB = intensityPowerlaw(img, 0.1)
  imgT = thresholdingValue(imgB, 200)
  imgC = cclN4(imgT)
  cells = getComponentsBox(imgC)
  print('cell components identified')
  acredines, imgA = loadAcredine()
  print('acredine bodies identified')
  fitcs, imgF = loadFITC()
  print('FITC bodies identified')
  output = []
  for cellId, cell in cells.items():
    if cell[0] == 0 or cell[1] == img.shape[1] or cell[2] == 0 or cell[3] == img.shape[1]:
      continue
    countA = countObjectsInsideShape(cell, acredines)
    countF = countObjectsInsideShape(cell, fitcs)
    print('cell ', cellId, ' has ', countA, ' Acredines and ', countF, ' of FITCs.')
    if countA != 0 and countF != 0 and cellId != 0:
      output.append([cellId, cell[0], cell[1], cell[2], cell[3], countA, countF])
  fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16,9))
  imgD = combine(imgA, imgF, img.astype(np.uint8))
  ax1.imshow(imgD, interpolation='nearest')
  ax1.set_axis_off()
  ax2.set_axis_off()
  tableLabels = ("ID", "topX", "botX", "topY", "botY", "Acrds", "CTICs")
  the_table = ax2.table(output, colLabels=tableLabels, loc="center")
  the_table.auto_set_font_size(False)
  the_table.set_fontsize(8)
  the_table.scale(1.2, 1.4)
  plt.subplots_adjust(left=0.2, top=0.8)
  plt.show()
  tifImg.close()

def loadAcredine():
  tifAcredine = TIFF.open('./data/BXY-ABCD_Region 002_FOV 00040_Acridine.tif', mode='r')
  imgAcredine = tifAcredine.read_image()
  tifAcredine.close()
  imgAcredineB = intensityPowerlaw(imgAcredine, 0.5)
  imgAcredineT = thresholdingValue(imgAcredineB, 200)
  imgAcredineTC = cclN4(imgAcredineT)
  return getComponentsBox(imgAcredineTC), imgAcredine

def loadFITC():
  tifFITC = TIFF.open('./data/BXY-ABCD_Region 002_FOV 00040_FITC.tif', mode='r')
  imgFITC = tifFITC.read_image()
  tifFITC.close()
  imgFITCB = intensityPowerlaw(imgFITC, 0.5)
  imgFITCT = thresholdingValue(imgFITCB, 100)
  imgFITCTC = cclN4(imgFITCT)
  return getComponentsBox(imgFITCTC), imgFITC

def countObjectsInsideShape(shape, objects):
  out = 0
  sTopX = shape[0]
  sBotX = shape[1]
  sTopY = shape[2]
  sBotY = shape[3]

  for _, inner in objects.items():
    iTopX = inner[0]
    iBotX = inner[1]
    iTopY = inner[2]
    iBotY = inner[3]
    out += 1 if iTopX > sTopX and iBotX < sBotX and iTopY > sTopY and iBotY < sBotY else 0
  
  return out

#for l, v in labelLocations.items():
#   print(l, v)
testCells()
#signalCountsAcredine()
#signalCountsFITC()