import numpy as np
import skimage
from skimage import io
from skimage import transform
a = io.imread('image_1.png',as_grey=True)
b = io.imread('image_2.png',as_grey=True)
c = np.abs(a-b)
io.imshow(c)
io.show()
io.imsave('diff_image.png',c)
print(c.shape)

resized_diff_left = transform.pyramid_reduce(c[:,:1024],downscale=32)
io.imshow(resized_diff_left)
io.show()
