#!/usr/bin/env python
import math
import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt
from PIL import Image
'''
img_grid = io.imread('grid.png',as_grey=False)
img_grid = img_grid[1:197, :,:]
img_grid = cv2.resize(img_grid, dsize=(1024,768), interpolation = cv2.INTER_CUBIC)
img1, img2, img3 = img_grid, img_grid, img_grid[:,:,0]
'''
#alpha_cvr, sfx, sfy = 0.05, 200, 200
alpha_cvr_x, alpha_cvr_y = 0.02, 0.02
#x_res, y_res = 2000, 1600
############################################
#img1, img2, img3, mid_x, mid_y = mrp_blur[0], mrp_blur[1], ior_map, 2*fix_x, 2*fix_y
def cvr_transform(img1, img2, img3, mid_x, mid_y, x_res, y_res):
    width, height = img1.shape[1], img1.shape[0]
    ######################### modified
    ## (vx, vy) -> (x, y)
    mapping_mat = np.zeros((y_res, x_res, 2))
    ######## doubt
    sfx_ = (1.*x_res) / (np.log(alpha_cvr_x*(width - mid_x) + 1) + np.log(alpha_cvr_x*mid_x + 1))
    sfy_ = (1.*y_res) / (np.log(alpha_cvr_y*(height - mid_y) + 1) + np.log(alpha_cvr_y*mid_y + 1))

    #sfx_, sfy_ = sfx_ / x_res, sfy_ / y_res

    mid_y_n = sfy_ * np.log(alpha_cvr_y*(1.0*mid_y )+ 1) # / y_res
    mid_x_n = sfx_ * np.log(alpha_cvr_x*(1.0*mid_x )+ 1) # / x_res
    x_grid, y_grid = np.meshgrid(np.linspace(0,1,x_res)*(x_res-1)-mid_x_n, np.linspace(0,1,y_res)*(y_res-1)-mid_y_n )

    x_grid_m = np.sign(x_grid)*(np.exp( abs(x_grid)/sfx_ ) - 1 )/alpha_cvr_x
    y_grid_m = np.sign(y_grid)*(np.exp( abs(y_grid)/sfy_ ) - 1 )/alpha_cvr_y

    x_grid_mod = x_grid_m - x_grid_m.min()
    y_grid_mod = y_grid_m - y_grid_m.min()

    ############################
    mapping_mat = np.zeros((y_res, x_res, 2))
    mapping_mat[:,:,0], mapping_mat[:,:,1] = x_grid_mod, y_grid_mod 
    width_n, height_n = mapping_mat.shape[1], mapping_mat.shape[0]
    dst_pix1 = np.zeros([height_n,width_n, 3])
    dst_pix2 = np.zeros([height_n,width_n, 3])
    dst_pix3 = np.zeros([height_n,width_n])
    for w in range(width_n):
        for h in range(height_n):
            x, y = round(x_grid_mod[h, w]), round(y_grid_mod[h, w])
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            dst_pix1[h, w] = tuple(img1[int(y), int(x)])
            dst_pix2[h, w] = tuple(img2[int(y), int(x)])
            dst_pix3[h, w] = img3[int(y), int(x)]

    return dst_pix1, dst_pix2, dst_pix3, mapping_mat #mid_x_n, mid_y_n


def reverse_cvr_transform(mapping_mat_r, vx_, vy_):
    xx, yy = round(mapping_mat_r[vy_, vx_, 0]), round(mapping_mat_r[vy_, vx_, 1])
    xx = min(max(xx, 0), 1024. - 1)
    yy = min(max(yy, 0), 768. - 1)
    return int(round(xx)), int(round(yy))

def cvr_transform_4(img1, img2, img3, img4, mid_x, mid_y, x_res, y_res):
    width, height = img1.shape[1], img1.shape[0]
    ######################### modified
    ## (vx, vy) -> (x, y)
    mapping_mat = np.zeros((y_res, x_res, 2))
    ######## doubt
    sfx_ = (1.*x_res) / (np.log(alpha_cvr_x*(width - mid_x) + 1) + np.log(alpha_cvr_x*mid_x + 1))
    sfy_ = (1.*y_res) / (np.log(alpha_cvr_y*(height - mid_y) + 1) + np.log(alpha_cvr_y*mid_y + 1))

    #sfx_, sfy_ = sfx_ / x_res, sfy_ / y_res

    mid_y_n = sfy_ * np.log(alpha_cvr_y*(1.0*mid_y )+ 1) # / y_res
    mid_x_n = sfx_ * np.log(alpha_cvr_x*(1.0*mid_x )+ 1) # / x_res
    x_grid, y_grid = np.meshgrid(np.linspace(0,1,x_res)*(x_res-1)-mid_x_n, np.linspace(0,1,y_res)*(y_res-1)-mid_y_n )

    x_grid_m = np.sign(x_grid)*(np.exp( abs(x_grid)/sfx_ ) - 1 )/alpha_cvr_x
    y_grid_m = np.sign(y_grid)*(np.exp( abs(y_grid)/sfy_ ) - 1 )/alpha_cvr_y

    x_grid_mod = x_grid_m - x_grid_m.min()
    y_grid_mod = y_grid_m - y_grid_m.min()

    ############################
    mapping_mat = np.zeros((y_res, x_res, 2))
    mapping_mat[:,:,0], mapping_mat[:,:,1] = x_grid_mod, y_grid_mod 
    width_n, height_n = mapping_mat.shape[1], mapping_mat.shape[0]
    dst_pix1 = np.zeros([height_n,width_n, 3])
    dst_pix2 = np.zeros([height_n,width_n, 3])
    dst_pix3 = np.zeros([height_n,width_n])
    dst_pix4 = np.zeros([height_n,width_n])
    for w in range(width_n):
        for h in range(height_n):
            x, y = round(x_grid_mod[h, w]), round(y_grid_mod[h, w])
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            dst_pix1[h, w] = tuple(img1[int(y), int(x)])
            dst_pix2[h, w] = tuple(img2[int(y), int(x)])
            dst_pix3[h, w] = img3[int(y), int(x)]
            dst_pix4[h, w] = img4[int(y), int(x)]

    return dst_pix1, dst_pix2, dst_pix3, dst_pix4, mapping_mat #mid_x_n, mid_y_n




'''
#plt.imshow(dst_pix1.astype(np.uint8))
#plt.show()

mid_x, mid_y = 860, 544
img_f = spherize(img_grid, mid_x, mid_y)
plt.imshow(img_f.astype(np.uint8))
plt.show()

img = reverse_spherize(img_f, mid_x, mid_y)
plt.imshow(dst_pix.astype(np.uint8))
plt.show()
'''


'''   
def reverse_cvr_transform(img, mid_x_n, mid_y_n, vx_, vy_):
    ########################
    ## (vx, vy) -> (x, y)
    dvy = 0.0 - mid_y_n
    dy = np.sign(dvy)*(np.exp( abs(1.0*dvy)/sfy ) - 1 )/alpha_cvr
    miny = mid_y_n + dy 
        ######
    dvy = vy_ - mid_y_n
    dy = np.sign(dvy)*(np.exp( abs(1.0*dvy)/sfy ) - 1 )/alpha_cvr
    yy = mid_y_n + dy - miny
    ###########################
    dvx = 0.0 - mid_x_n
    dx = np.sign(dvx)*(np.exp( abs(1.0*dvx)/sfx ) - 1 )/alpha_cvr
    minx = mid_x_n + dx 
    ######
    dvx = vx_ - mid_x_n
    dx = np.sign(dvx)*(np.exp( abs(1.0*dvx)/sfx ) - 1 )/alpha_cvr
    xx = mid_x_n + dx - minx
    return int(round(xx)), int(round(yy))

############################################
def cvr_transform(img1, img2, img3, mid_x, mid_y):
    width, height = img1.shape[1], img1.shape[0]
    #mid_x =  508 #508-1
    #mid_y =  400 #378-1 
    #width_n, height_n = 1313, 1203

    ########################
    ## (x, y) ---> (vx, vy)
    mapping_mat_ = np.zeros((height, width, 2))
    for y in [0, mid_y, height-1]:
        dy = y - mid_y
        if y==0: miny = mid_y + np.sign(dy)*sfy*np.log(alpha_cvr*abs(1.0*dy) + 1)
        mapping_mat_[y, :, 1] = mid_y + np.sign(dy)*sfy*np.log(alpha_cvr*abs(1.0*dy) + 1) - miny +1

    for x in [0, mid_x, width-1]:
        dx = x - mid_x
        if x==0: minx = mid_x + np.sign(dx)*sfx*np.log(alpha_cvr*abs(1.0*dx) + 1)
        mapping_mat_[:, x, 0] = mid_x + np.sign(dx)*sfx*np.log(alpha_cvr*abs(1.0*dx) + 1) - minx +1

    width_n, height_n = mapping_mat_[height-1, width-1].tolist()
    width_n, height_n = int(round(width_n)), int(round(height_n))
    mid_x_n, mid_y_n = mapping_mat_[mid_y, mid_x].tolist()
    mid_x_n, mid_y_n = int(round(mid_x_n)), int(round(mid_y_n))

    #########################
    ## (vx, vy) -> (x, y)
    mapping_mat = np.zeros((height_n, width_n, 2))
    for vy in range(0, height_n):
        dvy = vy - mid_y_n
        dy = np.sign(dvy)*(np.exp( abs(1.0*dvy)/sfy ) - 1 )/alpha_cvr
        if vy==0: miny = mid_y_n + dy
        #print mid_y_n + dy - miny
        mapping_mat[vy, :, 1] = mid_y_n + dy - miny

    for vx in range(0, width_n):
        dvx = vx - mid_x_n
        dx = np.sign(dvx)*(np.exp( abs(1.0*dvx)/sfx ) - 1 )/alpha_cvr
        if vx==0: minx = mid_x_n + dx
        #print mid_x_n + dx - minx
        mapping_mat[:, vx, 0] = mid_x_n + dx - minx

    ############################
    mapping_mat_r = cv2.resize(mapping_mat , (width, height), interpolation = cv2.INTER_NEAREST)
    width_n, height_n = mapping_mat_r.shape[1], mapping_mat_r.shape[0]
    dst_pix1 = np.zeros([height_n,width_n, 3])
    dst_pix2 = np.zeros([height_n,width_n, 3])
    dst_pix3 = np.zeros([height_n,width_n])
    for w in range(width_n):
        for h in range(height_n):
            x, y = round(mapping_mat_r[h, w, 0]), round(mapping_mat_r[h, w, 1])
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            dst_pix1[h, w] = tuple(img1[int(y), int(x)])
            dst_pix2[h, w] = tuple(img2[int(y), int(x)])
            dst_pix3[h, w] = img3[int(y), int(x)]

    return dst_pix1, dst_pix2, dst_pix3, mapping_mat_r #mid_x_n, mid_y_n
'''