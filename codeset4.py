from libtiff import TIFF
import numpy as np
from matplotlib import pyplot as plt
from codeset1 import MAX_VAL
from codeset3 import to8bit
from codeset2 import intensityPowerlaw

IMG_SET_4_REL_PATH = './datasets/imgset4/'

def pad(img):
  acc = np.full(img.shape, 0)
  res = np.concatenate((img, acc), axis=1)
  acc = np.concatenate((acc, acc), axis=1)
  return np.concatenate((res, acc), axis=0)

def testDFT():
  tifImg = TIFF.open(IMG_SET_4_REL_PATH + 'Fig0431(d)(blown_ic_crop).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  imgPadded = pad(img)
  dft0 = np.fft.fft2(imgPadded)
  dftS = np.fft.fftshift(dft0)
  dftA = abs(dftS) / dft0.size**(1/2)
  dftL = np.log(dftA + 1)
  dft8 = to8bit(dftL, mode='scale')

  isdftS = np.fft.ifftshift(dftS)
  isdftA = abs(isdftS) / isdftS.size**(1/2)
  isdftL = np.log(isdftA + 1)
  isdft8 = to8bit(isdftL, mode='scale')

  idft0 = np.fft.ifft2(isdftS)
  idftA = abs(idft0) / idft0.size**(1/2)
  idftL = np.log(idftA + 1)
  idft8 = to8bit(idftL, mode='scale')

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9))
  ax1.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(dft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(isdft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax4.imshow(idft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  plt.tight_layout()
  plt.show()
  tifImg.close()

testDFT()