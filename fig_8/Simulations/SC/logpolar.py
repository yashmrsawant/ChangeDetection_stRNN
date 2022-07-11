#!/usr/bin/env python

'''
plots image as logPolar and linearPolar

Usage:
    logpolar.py

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function
from skimage import io
import cv2
import matplotlib.pyplot as plt
if __name__ == '__main__':
    print(__doc__)

    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = '../data/fruits.jpg'

    img = cv2.imread(fn)
    b,g,r = cv2.split(img)       # get b,g,r
    img = cv2.merge([r,g,b])     # switch it to rgb
	

    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)
	
    img2 = cv2.logPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, cv2.WARP_FILL_OUTLIERS)
    img3 = cv2.linearPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, cv2.WARP_FILL_OUTLIERS)

    plt.imshow(img)
    plt.show()	
    plt.imshow(img2)
    plt.show()	
    plt.imshow(img3)
    plt.show()		
