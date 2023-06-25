from libtiff import TIFF
import numpy as np
from matplotlib import pyplot as plt
import math
from codeset1 import MAX_VAL
from codeset3 import to8bit
from codeset2 import intensityPowerlaw

IMG_SET_4_REL_PATH = './datasets/imgset4/'

def pad(img):
  acc = np.full(img.shape, 0)
  res = np.concatenate((img, acc), axis=1)
  acc = np.concatenate((acc, acc), axis=1)
  return np.concatenate((res, acc), axis=0)

def crop(img):
  return img[0:(len(img) // 2) - 1, 0:(len(img[0]) // 2) - 1]

def shift(img):
  resImg = np.copy(img)
  for i, row in enumerate(resImg):
    for j, col in enumerate(row):
      resImg[i][j] = resImg[i][j] * ((-1)**(i + j))
  return resImg

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

def generateSin2D():
  img = np.full((256, 256), 0).astype(np.float64)
  
  for col in range(0, img.shape[1]):
    img[:,col] = math.sin(col)
  return img

def testSin2D():
  img = generateSin2D() #.T for vertical axis peaks

  dft0 = np.fft.fft2(img)
  dftS = np.fft.fftshift(dft0)
  dftA = abs(dftS) / dft0.size**(1/2)
  dftL = np.log(dftA + 1)
  dft8 = to8bit(dftL, mode='scale')

  nldft0 = np.fft.fft2(img)
  nldftS = np.fft.fftshift(nldft0)
  nldftA = abs(nldftS) / nldft0.size**(1/2)
  nldft8 = to8bit(nldftA, mode='scale')

  imgDisplay = to8bit(img, mode='scale')
  sdft0 = np.fft.fft2(imgDisplay)
  sdftS = np.fft.fftshift(sdft0)
  sdftA = abs(sdftS) / sdft0.size**(1/2)
  sdftL = np.log(sdftA + 1)
  sdft8 = to8bit(sdftL, mode='scale')

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,9))
  ax1.imshow(imgDisplay, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_title('sin(x), scaled [0, 255] image')
  ax1.set_axis_off()
  ax2.imshow(nldft8, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_title('sin(x) DFT of scaled [0, 255] image')
  ax2.set_axis_off()
  ax3.imshow(sdft8, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_title('sin(x) DFT, scaled [0, 255], log ')
  ax3.set_axis_off()
  ax4.imshow(dft8, interpolation='nearest', cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_title('sin(x) DFT, no image scale, log')
  ax4.set_axis_off()
  plt.tight_layout()
  plt.show()

#testDFT()
#testSin2D()

def ILPF(img, r):
  resImg = np.copy(img)
  center = len(resImg[0]) // 2
  for i, row in enumerate(resImg):
    for j, col in enumerate(row):
      if (i - center)**2 + (j - center)**2 > r**2:
        resImg[i][j] = 0
  return resImg

def testILPF():
  tifImg = TIFF.open(IMG_SET_4_REL_PATH + 'Fig0431(d)(blown_ic_crop).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  imgPadded = pad(img)
  imgShifted = shift(imgPadded)
  imgShifted8 = to8bit(imgShifted, mode='clip')

  dft0 = np.fft.fft2(imgShifted)
  dftA = abs(dft0) / dft0.size**(1/2)
  dftL = np.log(dftA + 1)
  dft8 = to8bit(dftL, mode='scale')

  dftIlpf = ILPF(dft0, 100)
  dftIlpfA = abs(dftIlpf) / dftIlpf.size**(1/2)
  dftIlpfL = np.log(dftIlpfA + 1)
  dftIlpf8 = to8bit(dftIlpfL, mode='scale')

  idft0 = np.fft.ifft2(dftIlpf)
  idftR = np.real(idft0)
  idftS = shift(idftR)
  idft8 = to8bit(idftS, mode='clip')
  idftC = crop(idft8)

  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,9))
  ax1.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(imgPadded, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(imgShifted8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax4.imshow(dft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax5.imshow(dftIlpf8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  ax6.imshow(idftC, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax6.set_axis_off()
  plt.tight_layout()
  plt.show()
  tifImg.close()

#testILPF()

def GLPF(img, sigma):
  resImg = np.copy(img)
  center = len(resImg[0]) // 2
  for i, row in enumerate(resImg):
    for j, col in enumerate(row):
      D = (i - center)**2 + (j - center)**2
      resImg[i][j] = math.exp(-D / (2*(sigma**2)))
  return resImg

def testGLPF():
  tifImg = TIFF.open(IMG_SET_4_REL_PATH + 'Fig0431(d)(blown_ic_crop).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  imgPadded = pad(img)
  imgShifted = shift(imgPadded)
  imgShifted8 = to8bit(imgShifted, mode='clip')

  dft0 = np.fft.fft2(imgShifted)
  dftA = abs(dft0) / dft0.size**(1/2)
  dftL = np.log(dftA + 1)
  dft8 = to8bit(dftL, mode='scale')

  dftGlpf = GLPF(dft0, 50) * dft0 # this does the trick
  dftGlpfA = abs(dftGlpf) / dftGlpf.size**(1/2)
  dftGlpfL = np.log(dftGlpfA + 1)
  dftGlpf8 = to8bit(dftGlpfL, mode='scale')

  idft0 = np.fft.ifft2(dftGlpf)
  idftR = np.real(idft0)
  idftS = shift(idftR)
  idft8 = to8bit(idftS, mode='clip')
  idftC = crop(idft8)

  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,9))
  ax1.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(imgPadded, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(imgShifted8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax4.imshow(dft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax5.imshow(dftGlpf8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  ax6.imshow(idftC, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax6.set_axis_off()
  plt.tight_layout()
  plt.show()
  tifImg.close()

#testGLPF()

def BLPF(img, r, n):
  out = np.copy(img)
  centerI = len(out) // 2
  centerJ = len(out[0]) // 2
  for i, row in enumerate(out):
    for j, col in enumerate(row):
      D = ((i - centerI)**2 + (j - centerJ)**2)**(1/2)
      out[i][j] = 1 / (1 + abs(D / r)**(2*n))
  return out

def testBLPF():
  tifImg = TIFF.open(IMG_SET_4_REL_PATH + 'Fig0441(a)(characters_test_pattern).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  imgPadded = pad(img)
  imgShifted = shift(imgPadded)
  imgShifted8 = to8bit(imgShifted, mode='clip')

  dft0 = np.fft.fft2(imgShifted)
  dftA = abs(dft0) / dft0.size**(1/2)
  dftL = np.log(dftA + 1)
  dft8 = to8bit(dftL, mode='scale')

  dftBlpf = BLPF(dft0, 50, 1) * dft0 # this does the trick
  dftBlpfA = abs(dftBlpf) / dftBlpf.size**(1/2)
  dftBlpfL = np.log(dftBlpfA + 1)
  dftBlpf8 = to8bit(dftBlpfL, mode='scale')

  idft0 = np.fft.ifft2(dftBlpf)
  idftR = np.real(idft0)
  idftS = shift(idftR)
  idft8 = to8bit(idftS, mode='clip')
  idftC = crop(idft8)

  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,9))
  ax1.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(imgPadded, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(imgShifted8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax4.imshow(dft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax5.imshow(dftBlpf8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  ax6.imshow(idftC, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax6.set_axis_off()
  plt.tight_layout()
  plt.show()
  tifImg.close()

#testBLPF()

def IHPF(img, r):
  resImg = np.copy(img)
  center = len(resImg[0]) // 2
  for i, row in enumerate(resImg):
    for j, col in enumerate(row):
      if (i - center)**2 + (j - center)**2 > r**2:
        resImg[i][j] = 1
      else:
        resImg[i][j] = 0
  return resImg

def testIHPF():
  tifImg = TIFF.open(IMG_SET_4_REL_PATH + 'Fig0441(a)(characters_test_pattern).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  imgPadded = pad(img)
  imgShifted = shift(imgPadded)
  imgShifted8 = to8bit(imgShifted, mode='clip')

  dft0 = np.fft.fft2(imgShifted)
  dftA = abs(dft0) / dft0.size**(1/2)
  dftL = np.log(dftA + 1)
  dft8 = to8bit(dftL, mode='scale')

  dftIlpf = IHPF(dft0, 160) * dft0
  dftIlpfA = abs(dftIlpf) / dftIlpf.size**(1/2)
  dftIlpfL = np.log(dftIlpfA + 1)
  dftIlpf8 = to8bit(dftIlpfL, mode='scale')

  idft0 = np.fft.ifft2(dftIlpf)
  idftR = np.real(idft0)
  idftS = shift(idftR)
  idft8 = to8bit(idftS, mode='clip')
  idftC = crop(idft8)

  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,9))
  ax1.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(imgPadded, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(imgShifted8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax4.imshow(dft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax5.imshow(dftIlpf8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  ax6.imshow(idftC, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax6.set_axis_off()
  plt.tight_layout()
  plt.show()
  tifImg.close()

#testIHPF()

def BHPF(img, r, n):
  resImg = np.copy(img)
  center = len(resImg[0]) // 2
  for i, row in enumerate(resImg):
    for j, col in enumerate(row):
      D = ((i - center)**2 + (j - center)**2)**(1/2)
      if D == 0:
        resImg[i][j] = 0
      else:
        resImg[i][j] = 1 / (1 + abs(r / D)**(2*n))
  return resImg

def testBHPF():
  tifImg = TIFF.open(IMG_SET_4_REL_PATH + 'Fig0441(a)(characters_test_pattern).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  imgPadded = pad(img)
  imgShifted = shift(imgPadded)
  imgShifted8 = to8bit(imgShifted, mode='clip')

  dft0 = np.fft.fft2(imgShifted)
  dftA = abs(dft0) / dft0.size**(1/2)
  dftL = np.log(dftA + 1)
  dft8 = to8bit(dftL, mode='scale')

  dftBlpf = BHPF(dft0, 160, 2) * dft0 # this does the trick
  dftBlpfA = abs(dftBlpf) / dftBlpf.size**(1/2)
  dftBlpfL = np.log(dftBlpfA + 1)
  dftBlpf8 = to8bit(dftBlpfL, mode='scale')

  idft0 = np.fft.ifft2(dftBlpf)
  idftR = np.real(idft0)
  idftS = shift(idftR)
  idft8 = to8bit(idftS, mode='clip')
  idftC = crop(idft8)

  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,9))
  ax1.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(imgPadded, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(imgShifted8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax4.imshow(dft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax5.imshow(dftBlpf8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  ax6.imshow(idftC, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax6.set_axis_off()
  plt.tight_layout()
  plt.show()
  tifImg.close()

#testBHPF()

def GHPF(img, sigma):
  resImg = np.copy(img)
  center = len(resImg[0]) // 2
  for i, row in enumerate(resImg):
    for j, col in enumerate(row):
      D = (i - center)**2 + (j - center)**2
      resImg[i][j] = 1 - math.exp(-D / (2*(sigma**2)))
  return resImg

def testGHPF():
  tifImg = TIFF.open(IMG_SET_4_REL_PATH + 'Fig0441(a)(characters_test_pattern).tif', mode='r')
  img = tifImg.read_image().astype(np.float64)
  imgPadded = pad(img)
  imgShifted = shift(imgPadded)
  imgShifted8 = to8bit(imgShifted, mode='clip')

  dft0 = np.fft.fft2(imgShifted)
  dftA = abs(dft0) / dft0.size**(1/2)
  dftL = np.log(dftA + 1)
  dft8 = to8bit(dftL, mode='scale')

  dftGlpf = GHPF(dft0, 160) * dft0 # this does the trick
  dftGlpfA = abs(dftGlpf) / dftGlpf.size**(1/2)
  dftGlpfL = np.log(dftGlpfA + 1)
  dftGlpf8 = to8bit(dftGlpfL, mode='scale')

  idft0 = np.fft.ifft2(dftGlpf)
  idftR = np.real(idft0)
  idftS = shift(idftR)
  idft8 = to8bit(idftS, mode='clip')
  idftC = crop(idft8)

  fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,9))
  ax1.imshow(img, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax1.set_axis_off()
  ax2.imshow(imgPadded, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax2.set_axis_off()
  ax3.imshow(imgShifted8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax3.set_axis_off()
  ax4.imshow(dft8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax4.set_axis_off()
  ax5.imshow(dftGlpf8, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax5.set_axis_off()
  ax6.imshow(idftC, cmap='gray', vmin=0, vmax=MAX_VAL)
  ax6.set_axis_off()
  plt.tight_layout()
  plt.show()
  tifImg.close()

#testGHPF()

# FIXME -> use correct H and L for high low filter namings respectively