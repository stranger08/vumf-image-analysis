from libtiff import TIFF
import numpy as np
import math
from matplotlib import pyplot as plt
from codeset1 import MAX_VAL
from codeset3 import to8bit, blockAverageFilter
from codeset4 import pad, shift, crop, BLPF
from codeset5 import notchBBRF, BBRF, notchIBRF, contraharmonicMeanFilter, medianFilter, geometricMeanFilter

IMG_DATA_REL_PATH = './data/'

def kidney1():
  tifkidneyNoise1 = TIFF.open(IMG_DATA_REL_PATH + 'Kidney256-Noise1.tif', mode='r')
  imgkidneyNoise1 = tifkidneyNoise1.read_image().astype(np.float64)
  print(imgkidneyNoise1.shape)

  imgPadded = pad(imgkidneyNoise1)
  imgShifted = shift(imgPadded)
  imgShifted8 = to8bit(imgShifted, mode='clip')

  dft0 = np.fft.fft2(imgShifted)
  dftA = abs(dft0) / dft0.size**(1/2)
  dftL = np.log(dftA + 1)
  dft8 = to8bit(dftL, mode='scale')

  dftBbrf = dft0 * BBRF(dft0, 192, 30, 1)
  dftBbrfA = abs(dftBbrf) / dftBbrf.size**(1/2)
  dftBbrfL = np.log(dftBbrfA + 1)
  dftBbrf8 = to8bit(dftBbrfL, mode='scale')

  idft0 = np.fft.ifft2(dftBbrf)
  idftR = np.real(idft0)
  idftS = shift(idftR)
  idft8 = to8bit(idftS, mode='clip')
  idftC = crop(idft8)

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9))
  ax1.imshow(imgkidneyNoise1, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax1.set_title('Kidney noise case 1')

  ax2.imshow(dft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax2.set_title('Spectrum')

  ax3.imshow(dftBbrf8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax3.set_title('Spectrum filtered by BRRF')

  ax4.imshow(idftC, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax4.set_title('Image Filtered with BRRF')

  plt.tight_layout()
  plt.show()
  tifkidneyNoise1.close()

def kidney2():
  tifkidney = TIFF.open(IMG_DATA_REL_PATH + 'Kidney256.tif', mode='r')
  imgkidney = tifkidney.read_image().astype(np.float64)
  imgkidneyPadded = pad(imgkidney)
  imgkidneyShifted = shift(imgkidneyPadded)
  imgkidneyShifted8 = to8bit(imgkidneyShifted, mode='clip')
  dftimgkidney0 = np.fft.fft2(imgkidneyShifted)
  dftimgkidneyA = abs(dftimgkidney0) / dftimgkidney0.size**(1/2)
  dftimgkidneyL = np.log(dftimgkidneyA + 1)
  dftimgkidney8 = to8bit(dftimgkidneyL, mode='scale')

  tifkidneyNoise2 = TIFF.open(IMG_DATA_REL_PATH + 'Kidney256-Noise2.tif', mode='r')
  imgkidneyNoise2 = tifkidneyNoise2.read_image().astype(np.float64)
  imgtifkidneyNoise2Padded = pad(imgkidneyNoise2)
  imgtifkidneyNoise2Shifted = shift(imgtifkidneyNoise2Padded)
  imgtifkidneyNoise2Shifted8 = to8bit(imgtifkidneyNoise2Shifted, mode='clip')

  dftimgkidneyNoise20 = np.fft.fft2(imgtifkidneyNoise2Shifted)
  dftimgkidneyNoise2A = abs(dftimgkidneyNoise20) / dftimgkidneyNoise20.size**(1/2)
  dftimgkidneyNoise2L = np.log(dftimgkidneyNoise2A + 1)
  dftimgkidneyNoise28 = to8bit(dftimgkidneyNoise2L, mode='scale')

  #dftimgkidneyNoise2Bbrf = dftimgkidneyNoise20 * notchIBRF(dftimgkidneyNoise20, 28, 28, 4) * notchIBRF(dftimgkidneyNoise20, 28, -28, 4) * notchIBRF(dftimgkidneyNoise20, 0, 40, 4) * notchIBRF(dftimgkidneyNoise20, 40, 0, 4)
  #dftimgkidneyNoise2Bbrf = dftimgkidneyNoise20 * notchBBRF(dftimgkidneyNoise20, 28, 28, 3, 1) * notchBBRF(dftimgkidneyNoise20, 28, -28, 3, 1) * notchBBRF(dftimgkidneyNoise20, 0, 40, 1, 1) * notchBBRF(dftimgkidneyNoise20, 40, 0, 1, 1)
  dftimgkidneyNoise2Bbrf = dftimgkidneyNoise20 * BBRF(dftimgkidneyNoise20, 40, 5, 2)
  dftimgkidneyNoise2BbrfA = abs(dftimgkidneyNoise2Bbrf) / dftimgkidneyNoise2Bbrf.size**(1/2)
  dftimgkidneyNoise2BbrfL = np.log(dftimgkidneyNoise2BbrfA + 1)
  dftimgkidneyNoise2Bbrf8 = to8bit(dftimgkidneyNoise2BbrfL, mode='scale')

  idft0Filtered = np.fft.ifft2(dftimgkidneyNoise2Bbrf)
  idftR = np.real(idft0Filtered)
  idftS = shift(idftR)
  idft8 = to8bit(idftS, mode='clip')
  idftC = crop(idft8)

  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,9))
  ax1.imshow(imgkidney, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax1.set_title('Kidney noise free')

  ax2.imshow(imgkidneyNoise2, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax2.set_title('Kidney noise case 2')

  ax3.imshow(dftimgkidneyNoise2Bbrf8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax3.set_title('Spectrum filtered by BRRF')

  ax4.imshow(dftimgkidney8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax4.set_title('Spectrum of noise free image')

  ax5.imshow(dftimgkidneyNoise28, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  ax5.set_title('Spectrum of noised image case 2')

  ax6.imshow(idftC, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax6.set_axis_off()
  ax6.set_title('Reconstructed image')

  plt.tight_layout()
  plt.show()
  tifkidney.close()
  tifkidneyNoise2.close()

def thorax():
  tifthorax = TIFF.open(IMG_DATA_REL_PATH + 'Thorax.tif', mode='r')
  imgthorax = tifthorax.read_image().astype(np.float64)

  tifthoraxNoise1 = TIFF.open(IMG_DATA_REL_PATH + 'Thorax-Noise1.tif', mode='r')
  imgthoraxNoise1 = tifthoraxNoise1.read_image().astype(np.float64)

  imgMedian1 = medianFilter(imgthoraxNoise1, 3, 3, mirror=False)
  imgMedian18 = to8bit(imgMedian1, mode='clip')

  imgMedian2 = medianFilter(imgMedian1, 3, 3, mirror=False)
  imgMedian28 = to8bit(imgMedian2, mode='clip')

  imgMedian3 = medianFilter(imgMedian2, 3, 3, mirror=False)
  imgMedian38 = to8bit(imgMedian3, mode='clip')

  tifthoraxNoise2 = TIFF.open(IMG_DATA_REL_PATH + 'Thorax-Noise2.tif', mode='r')
  imgthoraxNoise2 = tifthoraxNoise2.read_image().astype(np.float64)

  imgGMF = geometricMeanFilter(imgthoraxNoise2, 5, 5, mirror=False)
  imgGMF8 = to8bit(imgGMF, mode='clip')

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9))

  ax1.imshow(imgthoraxNoise1, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax1.set_title('Thorax noise case 1')

  ax2.imshow(imgthoraxNoise2, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax2.set_title('Thorax noise case 2')

  ax3.imshow(imgMedian38, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax3.set_title('3 iterations with median 3x3 filter')

  ax4.imshow(imgGMF8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax4.set_title('Geometric mean filter 5x5')

  plt.tight_layout()
  plt.show()
  tifthorax.close()
  tifthoraxNoise1.close()

kidney1()
kidney2()
thorax()