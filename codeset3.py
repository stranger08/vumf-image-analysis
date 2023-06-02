import numpy as np

def spatialFilter(image, filter, convolution=False):
  out = np.copy(image)
  relIdxMaxOffset = len(filter) // 2

  for img_i, row in enumerate(image):
    for img_j, x in enumerate(row):

      acc = 0

      for fil_i in range(-relIdxMaxOffset, relIdxMaxOffset + 1):
        for fil_j in range(-relIdxMaxOffset, relIdxMaxOffset + 1):
          # calc eff coordinates
          i = img_i + fil_i if not convolution else img_i - fil_i
          j = img_j + fil_j if not convolution else img_j - fil_j
          # check if outside of an image
          if i < 0 or j < 0:
            continue
          if i >= image.shape[0] or j >= image.shape[1]:
            continue
          if image[i, j] == 0:
            continue
          acc += image[i, j] * filter[relIdxMaxOffset + fil_i, fil_j + relIdxMaxOffset]

      out[img_i, img_j] = acc

  return out

def testSpacialFilter():
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

  res = spatialFilter(image, filter, convolution=False)
  print(res)

testSpacialFilter()