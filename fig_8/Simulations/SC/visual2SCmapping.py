# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Wahiba Taouali (Wahiba.Taouali@inria.fr)
#               Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
import os
import numpy as np

from helper import *
from graphics import *
from projections import *
from stimulus import *
from skimage import io
from skimage.transform import resize
import cv2


if __name__ == '__main__':
  import matplotlib
  import matplotlib.pyplot as plt
  from matplotlib.patches import Polygon
  from mpl_toolkits.axes_grid1 import ImageGrid
  from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
  from mpl_toolkits.axes_grid1.inset_locator import mark_inset
  font = {'size'   : 18}
  matplotlib.rc('font', **font)
  P,IP = retina_projection()
  #print('p:',P.shape)
  #print('retina:',retina_shape)
  img = io.imread('input.png',as_grey=True)		
  #img_b = io.imread('bin_image_2.png',as_grey=True)		
  #img = np.abs(img_a - img_b)	
  img_height = img.shape[0] 
  img_width = img.shape[1]
  #print('img:',img_height,img_width)
  vf_height = 1600
  vf_half_width = 1050
  sc_height = 1600
  sc_half_width = 1050
  delta_x = img_width/8
  delta_y = img_height/8
  for i in range(3,4):
   for j in range(3,4):	 
    #print('itr:',i,j)
    plt.axis('off')
    R = np.zeros((int(vf_height),2*int(vf_half_width)))
    #print('R',R.shape)
    fcy = (j+1)*delta_y # fixation center x
    fcx = (i+1)*delta_x # fixation center y
    R[int(vf_height/2)-fcy:int(vf_height/2)+(img_height-fcy),int(vf_half_width)-fcx:int(vf_half_width)+(img_width-fcx)]=img	
    plt.imshow(R,cmap='gray')
    plt.show()	
    print('original')	 
    #plt.savefig('./results_bin_image_2/%d_%d_visual_image.png'%(i,j))
    #io.imsave('./results1k_bin_image_1/%d_%d_visual_image.png'%(i,j),R)
    # Take half-retina
    R_Left = R[:,int(retina_shape[1]):]
    #plt.imshow(R_Left)
    #plt.show()
    #print('retina_left')
    # Project to colliculus
    SC_Left = R_Left[P[...,0], P[...,1]]
    plt.imshow(SC_Left)
    plt.show()	
    print('SC_left')		
    #Inv_SC_Left = SC_Left[IP[...,0],IP[...,1]]
    #plt.imshow(Inv_SC_Left)	
    #plt.show()	
    #print('invert SC Left')	
    R = np.fliplr(R)		 
    # Take half-retina
    R_Right = R[:,int(retina_shape[1]):]
    # Project to colliculus
    SC_Right = R_Right[P[...,0], P[...,1]]
    SC_Right = np.fliplr(SC_Right)	    
    plt.imshow(SC_Right)
    plt.show()	
    print('SC_Right')		
    SC = np.zeros((sc_height,2*sc_half_width))
    SC[:,:sc_half_width]=SC_Right
    SC[:,sc_half_width:]=SC_Left
    plt.axis('off')	
    plt.imshow(SC,cmap='gray')
    plt.show()
    print('SC')
    VF_Left = SC_Left[IP[...,0],IP[...,1]]
    plt.imshow(VF_Left,cmap='gray')
    plt.show()
    print('vf_left')	
    SC = np.fliplr(SC)
    SC_Right = SC[:,sc_half_width:]
    VF_Right = SC_Right[IP[...,0],IP[...,1]]
    VF_Right = np.fliplr(VF_Right)
    plt.imshow(VF_Right,cmap='gray')
    plt.show()
    print('vf_right')	

    VF = np.zeros((vf_height,2*vf_half_width))
    VF[:,:vf_half_width]=VF_Right
    VF[:,vf_half_width:]=VF_Left

    #print('difference in VF:')
    plt.imshow(VF,cmap='gray')
    plt.show()
    print('inverted')	
	
    #plt.savefig('./results_bin_image_2/%d_%d_SC_image.png'%(i,j))		   
    #io.imsave('./results1k_bin_image_1/%d_%d_SC_image.png'%(i,j),SC)		   
    plt.close('all')	 	  	
