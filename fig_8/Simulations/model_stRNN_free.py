from skimage import io
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("./Saliency/itti-koch/pySaliencyMap")
import pySaliencyMap
import cv2
sys.path.append("./SC")
from projections import *
import scipy.ndimage
import time
from collections import deque
from STRNN import *
import os
import scipy.ndimage as ndimage
execfile('cvr_transform.py')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        #print '%r %2.2f sec' % (method.__name__, te-ts)
        return result
    return timed

# These are pre-requisites to compute visual to SC mapping
retina_shape     = np.array([1600,1050]).astype(int)
projection_shape = np.array([1600,1050]).astype(int)
n = 128
colliculus_shape = np.array([n,n]).astype(int)
stimulus_size      = 1.5 # in degrees
stimulus_intensity = 1.5
second       = 1.0
millisecond  = 0.001
dt           = 5*millisecond
duration     = 10*second
noise        = 0.01
sigma_e  = 0.10
A_e      = 1.30
sigma_i  = 1.00
A_i      = 0.65
alpha    = 12.5
tau      = 10*millisecond
scale    = 40.0*40.0/(n*n)
P, IP = retina_projection()

plot_counter = 0

# The Winner-Take-All Network
s = 50
Wg = np.zeros((s*s,s*s),dtype = np.float32)
for i in range(s*s):
	for j in range(s*s):
		Wg[i,j] = 0.001 / (s * s)
		if i==j:
			Wg[i,j] = 0.0


# read trained weights for ST-RNN cell
'''
Wx1 = np.loadtxt('./weights/new_weights/Wx297_per_g.csv').astype(np.float32)
Wg1 = np.loadtxt('./weights/new_weights/Wg297_per_g.csv').astype(np.float32)
Wh1 = np.loadtxt('./weights/new_weights/Wh297_per_g.csv').astype(np.float32)
Bi1 = np.loadtxt('./weights/new_weights/bi297_per_g.csv').astype(np.float32)
Wo1 = np.loadtxt('./weights/new_weights/Wo297_per_g.csv').astype(np.float32)
Bo1 = np.loadtxt('./weights/new_weights/bo297_per_g.csv').astype(np.float32)
'''
Wx2 = np.loadtxt('./weights/Wx9999_cd.csv').astype(np.float32)
Wh2 = np.loadtxt('./weights/Wh9999_cd.csv').astype(np.float32)
Bi2 = np.loadtxt('./weights/bi9999_cd.csv').astype(np.float32)
Wo2 = np.loadtxt('./weights/Wo9999_cd.csv').astype(np.float32)
Bo2 = np.loadtxt('./weights/bo9999_cd.csv').astype(np.float32)

# Models ### update
index_weight = 35
Wg1_g = np.loadtxt('./weights/Wg%d.csv'%(index_weight)).astype(np.float32)
Wx1_g = np.loadtxt('./weights/Wx%d.csv'%(index_weight)).astype(np.float32)
Wh1_g = np.loadtxt('./weights/Wh%d.csv'%(index_weight)).astype(np.float32)
Bi1_g = np.loadtxt('./weights/bi%d.csv'%(index_weight)).astype(np.float32)
Wo1_g = np.loadtxt('./weights/Wo%d.csv'%(index_weight)).astype(np.float32)
Bo1_g = np.loadtxt('./weights/bo%d.csv'%(index_weight)).astype(np.float32)
'''
Wx1_g = np.loadtxt('./weights/new_weights/strnn_g/Wx50.csv').astype(np.float32)
Wg1_g = np.loadtxt('./weights/new_weights/strnn_g/Wg50.csv').astype(np.float32)
Wh1_g = np.loadtxt('./weights/new_weights/strnn_g/Wh50.csv').astype(np.float32)
Bi1_g = np.loadtxt('./weights/new_weights/strnn_g/bi50.csv').astype(np.float32)
Wo1_g = np.loadtxt('./weights/new_weights/strnn_g/Wo50.csv').astype(np.float32)
Bo1_g = np.loadtxt('./weights/new_weights/strnn_g/bo50.csv').astype(np.float32)
'''
# create tensorflow computational graph
hidden_layer_size = 256+64
input_size, input_g_size =  64, 80
target_size = 64

rnn_1 = RNN_cell_1_g(input_size, hidden_layer_size, target_size, input_g_size, Wx1_g, Wg1_g, Wh1_g,Wo1_g,Bi1_g,Bo1_g)
p_outputs = rnn_1.get_outputs()
p_h_states  = rnn_1.get_states()

rnn_2 = RNN_cell_2(input_size, hidden_layer_size, target_size, Wx2, Wh2, Wo2, Bi2, Bo2)
outputs = rnn_2.get_outputs()
h_states  = rnn_2.get_states()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

@timeit
def create_batch_input(data):
	############ to create global feature ##############
	sample_size = 128*96 ## 50000
	BX = np.zeros((2,64))
	M_1_g = np.zeros((sample_size,data.shape[0],80))
	M_1 = np.zeros((sample_size,data.shape[0],64))
	M_2 = np.zeros((sample_size,data.shape[0],64))
	M_3 = np.zeros((sample_size,data.shape[0],64))
	M_4 = np.zeros((sample_size,data.shape[0],64))
	###***************************************###
	for i in range(data.shape[0]):
		M_1_g[:,i, :] = np.reshape( cv2.resize(data[i] , (10, 8), interpolation = cv2.INTER_NEAREST) > 0.1 ,(1,80))
	counter = 0
	for i in range(96-1):
		for j in range(128-1):
			t_1 = data[:,   i*8:(i+1)*8,           j*8:(j+1)*8]
			t_2 = data[:,   4+ i*8: 4+ (i+1)*8,    j*8:(j+1)*8]
			t_3 = data[:,   i*8:(i+1)*8,       4+j*8:4+(j+1)*8]
			t_4 = data[:,   4+i*8:4+(i+1)*8,    4+j*8:4+(j+1)*8]
			for k in range(data.shape[0]):
				M_1[counter,k,:] = np.reshape(t_1[k,:,:],(1,64))
				M_2[counter,k,:] = np.reshape(t_2[k,:,:],(1,64))
				M_3[counter,k,:] = np.reshape(t_3[k,:,:],(1,64))
				M_4[counter,k,:] = np.reshape(t_4[k,:,:],(1,64))
			counter += 1
	return M_1.copy().astype(np.float32), M_2.copy().astype(np.float32), M_3.copy().astype(np.float32), M_4.copy().astype(np.float32), M_1_g.copy().astype(np.float32)

#  The mapping from visual space to SC space [ottes et al]
@timeit
def visual2SC(img,x,y):
	grey_blr_img = img
	R = np.zeros((1600,2100))
	img_height = img.shape[0]
	img_width = img.shape[1]
	vf_height = 1600
	vf_half_width = 1050
	sc_height = 1600
	sc_half_width = 1050
	R = np.zeros((int(vf_height),2*int(vf_half_width)))
	fcy = y # fixation center x
	fcx = x # fixation center y
	R[int(vf_height/2)-fcy:int(vf_height/2)+(img_height-fcy),
	int(vf_half_width)-fcx:int(vf_half_width)+(img_width-fcx)] = grey_blr_img
	R_Left = R[:,int(retina_shape[1]):]
	SC_Left = R_Left[P[...,0], P[...,1]]
	R = np.fliplr(R)
	R_Right = R[:,int(retina_shape[1]):]
	SC_Right = R_Right[P[...,0], P[...,1]]
	SC_Right = np.fliplr(SC_Right)
	SC = np.zeros((sc_height,2*sc_half_width))
	SC[:,:sc_half_width]=SC_Right
	SC[:,sc_half_width:]=SC_Left
	return SC

# []
@timeit
def SC2Visual(SC_diff,x,y):
	vf_height = 1600
	vf_half_width = 1050
	sc_height = 1600
	sc_half_width = 1050
	img_height = SC_diff.shape[0]
	img_width = SC_diff.shape[1]
	SC_diff_left = SC_diff[:,int(retina_shape[1]):]
	VF_diff_left = SC_diff_left[IP[...,0],IP[...,1]] 
	SC_diff = np.fliplr(SC_diff)
	SC_diff_right = SC_diff[:,int(retina_shape[1]):]
	VF_diff_right = SC_diff_right[IP[...,0],IP[...,1]]
	VF_diff_right = np.fliplr(VF_diff_right)
	VF_diff = np.zeros((vf_height,2*vf_half_width))
	VF_diff[:,:vf_half_width]=VF_diff_right
	VF_diff[:,vf_half_width:]=VF_diff_left
	return VF_diff

# []
@timeit
def read_image_io(img_id):
	a = io.imread('./assets/sprites/%d_a.jpg'%img_id,as_grey=False)
	b = io.imread('./assets/sprites/%d_b.jpg'%img_id,as_grey=False)
	diff = io.imread('./assets/sprites/%d_rew.jpg'%img_id,as_grey=False)
	diff = diff > 0
	return a,b,diff

# []
@timeit
def read_image_cv(img_id):
	a = cv2.imread('./Images/test_images/%d_a.jpg'%img_id,1)
	b = cv2.imread('./Images/test_images/%d_b.jpg'%img_id,1)        
#	a = np.zeros((768,1024,3),dtype=np.uint8)
#	b = np.zeros((768,1024,3),dtype=np.uint8)
#	a[50:200,50:200,:] = 255
#	b[50:200,50:200,:] = 255
#	a[300:400,400:600,:] = 255
#	b[300:400,400:600,:] = 255
#	b[600:700,600:1000,:] = 255
	diff = cv2.imread('./Images/test_images/%d_rew.jpg'%img_id,0)
	bu,gr,re = cv2.split(a) 
	a = cv2.merge([re,gr,bu])   
	bu,gr,re = cv2.split(b) 
	b = cv2.merge([re,gr,bu])  
	diff = diff > 0
	return a,b,diff

@timeit
def get_diff(img_id):
	diff = cv2.imread('./Images/test_images/%d_rew_.jpg'%img_id,0)
	diff = diff > 0
	return diff

@timeit
def read_saliency(img_id):
	a = io.imread('./Images/test_images/%d_a_map.png'%img_id,as_grey=True)
	b = io.imread('./Images/test_images/%d_b_map.png'%img_id,as_grey=True)
	return a.astype(np.float32),b.astype(np.float32)
	

# Multi-Resolution Pyramid [Burt and Adelson paper] 
@timeit
def blur_image(img,x,y):
	kernel25 = np.ones((10,10),np.float32)/100
	kernel16 = np.ones((7,7),np.float32)/49
	kernel9  = np.ones((4,4),np.float32)/16
	img25 = cv2.filter2D(img,-1,kernel25)
	img16 = cv2.filter2D(img,-1,kernel16)
	img9 = cv2.filter2D(img,-1,kernel9)
	blr_img = np.zeros((img.shape[0],img.shape[1],3))
	maxd = 0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			dist = np.linalg.norm((x-i,y-j))
			maxd = max(dist,maxd)
	 		if (dist/50)>=3:
				blr_img[i,j,:] = img25[i,j,:]
			elif (dist/50)>=2:
				blr_img[i,j,:] = img16[i,j,:]
			elif (dist/50)>=1:
				blr_img[i,j,:] = img9[i,j,:]
			else:	
				blr_img[i,j,:] = img[i,j,:]
	return blr_img.astype(np.uint8)

@timeit
def blur_image_approx(img,x,y):
	kernel25 = np.ones((16,16),np.float32)/256
	kernel16 = np.ones((10,10),np.float32)/100
	kernel9  = np.ones((4,4),np.float32)/16
	img25 = cv2.filter2D(img,-1,kernel25)
	img16 = cv2.filter2D(img,-1,kernel16)
	img9 = cv2.filter2D(img,-1,kernel9)
	blr_img = np.zeros((img.shape[0],img.shape[1],3))
	xmin = 0
	xmax = 1024
	ymin = 0
	ymax = 768
	blr_img = img25.copy()
	r1 = 100
	r2 = 200
	r3 = 300
	blr_img[max(ymin,y-r3):min(ymax,y+r3),max(xmin,x-r3):min(xmax,x+r3),:]=img16[max(ymin,y-r3):min(ymax,y+r3),max(xmin,x-r3):min(xmax,x+r3),:]
	blr_img[max(ymin,y-r2):min(ymax,y+r2),max(xmin,x-r2):min(xmax,x+r2),:]=img9[max(ymin,y-r2):min(ymax,y+r2),max(xmin,x-r2):min(xmax,x+r2),:]
	blr_img[max(ymin,y-r1):min(ymax,y+r1),max(xmin,x-r1):min(xmax,x+r1),:]=img[max(ymin,y-r1):min(ymax,y+r1),max(xmin,x-r1):min(xmax,x+r1),:]
	return blr_img.astype(np.uint8)
		

# itti-koch implementation
@timeit
def computesaliency(img, binarize = True):
	sm = pySaliencyMap.pySaliencyMap2(img.shape[1], img.shape[0])
	aa, bb = sm.SMGetBothSM(img)
	return aa, bb

# winner take all computation
@timeit
def apply_WTA(img,Wg,s):
	img = cv2.resize(img,None,fx=0.02, fy=0.02, interpolation = cv2.INTER_CUBIC)	
	for k in range(9):
		i_t = img.copy()
		i_t = np.reshape(i_t,(50*50,1))
		for i in range((k+1)*100):
			i_t = -1 * Wg.dot(i_t) + i_t
		mask = i_t > 0.0
                i_t = i_t * mask
	img = np.reshape(i_t,(50,50))
	img = cv2.resize(img,None,fx=50, fy=50, interpolation = cv2.INTER_CUBIC)	
	return img


@timeit
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

# []
@timeit
def gaussian_blur(img,sigma):
	blr=scipy.ndimage.filters.gaussian_filter(img, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
	return blr 

@timeit
def scale(img,out_min,out_max):
	in_min = np.min(img[:])
	in_max = np.max(img[:])
	res = out_min + img * ((out_max - out_min)/(in_max - in_min)) - in_min * ((out_max - out_min)/(in_max - in_min))
	return res


#@timeit
#def create_mudsplash(a,b,diff,num_mudsplashes):
#	am = a
#	bm = b
#	ms = diff
#	count = 0
#	x = []
#	y = []
#	radius = []
#	while count < num_mudsplashes:
#		y = np.random.randint(200,460)
#		x = np.random.randint(200,800)
#		radius = np.random.randint(100,200)
#		new = np.zeros((768,1024))
#		new[y:y+radius,xx]
#		if(intersect(ms,new)):
#							
#		else:
#			count += 1
#			ms += new		
#	return am,bm


@timeit
def input_generator(img_id=134,distractor='blank',num_blanks=1,num_mudsplashes=3):
	a,b,diff = read_image_cv(img_id)
	blank = np.zeros([768,1024,3])
	size = num_blanks*2 + 2
	cd_blank_q = deque([],maxlen=size)
	cd_blank_q.append(a)
	for i in range(num_blanks):
		cd_blank_q.append(blank)
	cd_blank_q.append(b)
	for i in range(num_blanks):
		cd_blank_q.append(blank)
#	am,bm = create_mudsplash(a,b,diff,num_mudsplashes)
#	cd_ms_q = deque([],maxlen=size)
#	cd_ms_q.append(am)
#	for i in range(num_blanks):
#		cd_ms_q.append(a)
#	cd_ms_q.append(bm)
#	for i in range(num_blanks):
#		cd_ms_q.append(b)
#	cd_q = {'blank':cd_blank_q,'mudsplash':cd_ms_q}[distractor]
	while True:
		cd_blank_q.rotate(0)
		yield cd_blank_q

@timeit
def calc_saccade_vector(result):
	norm = np.sum(result)
	result = result/norm
	point = np.random.choice(1024*768, p = result.flatten())
	#point = np.argmax(result)
	y_t = point / 1024
	x_t = point % 1024
	return x_t,y_t
# []
@timeit
def plot(data,cm=plt.cm.RdYlBu):
	global plot_counter 
	plt.imshow(data,cmap=cm)
	plot_counter += 1
	#plt.savefig(str(plot_counter)+'.png')
	plt.show()


@timeit
def tiled_STRNN_direct(M_1, M_2, M_3, M_4, M_g):
	#################################################################
	p_predict_1 = sess.run(p_outputs, feed_dict={rnn_1._inputs: (M_1).astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
	p_predict_2 = sess.run(p_outputs, feed_dict={rnn_1._inputs: (M_2).astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
	p_predict_3 = sess.run(p_outputs, feed_dict={rnn_1._inputs: (M_3).astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
	p_predict_4 = sess.run(p_outputs, feed_dict={rnn_1._inputs: (M_4).astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
	#p_state = sess.run(p_h_states, feed_dict={rnn_1._inputs: X_test, y : y_test})
	p_predict_1 = np.float32(p_predict_1 >0.1) ############ update  > 0.5
	p_predict_2 = np.float32(p_predict_2 >0.1) ############ update  > 0.5
	p_predict_3 = np.float32(p_predict_3 >0.1) ############ update  > 0.5
	p_predict_4 = np.float32(p_predict_4 >0.1) ############ update  > 0.5
	#input_cd = np.swapaxes(p_predict,0,1)
	predict_1 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_1,0,1)})
	predict_2 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_2,0,1)})
	predict_3 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_3,0,1)})
	predict_4 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_4,0,1)})
	#state = sess.run(h_states, feed_dict={rnn_2._inputs: input_cd})
	z = tf.placeholder(tf.float32,shape=[None, 64])
	q = tf.reshape(z,(128*96,8,8))
	result1_a, result2_a = np.zeros((4,768,1024)), np.zeros((4,768,1024))
	result1_b, result2_b = np.zeros((4,768,1024)), np.zeros((4,768,1024))
	result1_c, result2_c = np.zeros((4,768,1024)), np.zeros((4,768,1024))
	result1_d, result2_d = np.zeros((4,768,1024)), np.zeros((4,768,1024))
	for itr in range(4):
	    out1_a, out2_a = sess.run(q, feed_dict={z:p_predict_1[itr,:,:]}), sess.run(q, feed_dict={z:predict_1[itr,:,:]})
	    out1_b, out2_b = sess.run(q, feed_dict={z:p_predict_2[itr,:,:]}), sess.run(q, feed_dict={z:predict_2[itr,:,:]})
	    out1_c, out2_c = sess.run(q, feed_dict={z:p_predict_3[itr,:,:]}), sess.run(q, feed_dict={z:predict_3[itr,:,:]})
	    out1_d, out2_d = sess.run(q, feed_dict={z:p_predict_4[itr,:,:]}), sess.run(q, feed_dict={z:predict_4[itr,:,:]}) 
	    for i in range(12064):
	        m = i / (128-1)
	        n = i % (128-1)
	        result1_a[itr,  m*8:(m+1)*8,   n*8:(n+1)*8]= out1_a[i,:,:] ### update np.abs
	        result2_a[itr,  m*8:(m+1)*8,   n*8:(n+1)*8]= out2_a[i,:,:] ### update np.abs
	        ##
	        result1_b[itr, 4+m*8:4+(m+1)*8, n*8:(n+1)*8]= out1_b[i,:,:] ### update np.abs
	        result2_b[itr, 4+m*8:4+(m+1)*8, n*8:(n+1)*8]= out2_b[i,:,:] ### update np.abs
	        ##
	        result1_c[itr,  m*8:(m+1)*8, 4+n*8:4+(n+1)*8]= out1_c[i,:,:] ### update np.abs
	        result2_c[itr,  m*8:(m+1)*8, 4+n*8:4+(n+1)*8]= out2_c[i,:,:] ### update np.abs
	        ##
	        result1_d[itr, 4+m*8:4+(m+1)*8, 4+n*8:4+(n+1)*8]= out1_d[i,:,:] ### update np.abs
	        result2_d[itr, 4+m*8:4+(m+1)*8, 4+n*8:4+(n+1)*8]= out2_d[i,:,:] ### update np.abs
	STRNN_output_per = ( (result1_a + result1_b +result1_c + result1_d ) / 4.0  )
	STRNN_output_cd = ( (result2_a + result2_b +result2_c + result2_d ) / 4.0  )
	return STRNN_output_per, STRNN_output_cd

# implementation for inhibition of return
size = 100
sigma_x = .8
sigma_y = .8
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
x, y = np.meshgrid(x, y)
zz = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
     + y**2/(2*sigma_y**2))))
zz = (zz/zz[49,49]) * 1
#plt.imshow(z)
#plt.show()
@timeit
def update_ior_map(x,y):
	ior_map_pad[50+y-50:50+y+50, 50+x-50:50+x+50] += zz
	ior_map = ior_map_pad[50:-50,50:-50]

########################### jnk update ######################################
#############################################################################
#stop = stop+1
for trial_id in range(0,40):
	for num in (np.array([0,1,2])).tolist():
		##********************************************##
		print('################ started doing the task ####################')
		#num=7
		ior_map = np.zeros((768,1024),dtype=np.float32)
		saccade_path = np.zeros((400,2))
		num_saccades = np.zeros(25)
		#cb_net(num,512,384,num_distractors = 1)
		#plt.show()
		image, start_x,start_y, num_distractors = num, 512, 384, 2
		#@timeit
		#def cb_net(image,start_x,start_y,distractor='blank',num_distractors=0):
		found = False
		over = False
		counter = 0
		fix_x = start_x
		fix_y = start_y
		dfa,dfb = read_saliency(image)
		dfa = scale(dfa,0,1)
		ig = input_generator(img_id=image,distractor='blank',num_blanks=num_distractors)
		# Inhibition Of Return Map ## updated
		ior_map = np.zeros((768,1024),dtype=np.float32)
		ior_map_pad = np.zeros((768+100,1024+100),dtype=np.float32) #ior_map_pad[50:-50,50:-50]
		#####################################
		input_seq = next(ig)
		diff = get_diff(image)
		size = len(input_seq)
		#### Dilation of the gt chnage ########
		diff = cv2.dilate(diff.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1) > 0.5
		cv2.imwrite('saccade_paths/gt_image_%d.png'%image, (255*diff.astype(np.float32)).astype(np.uint8))
		while(found == False and over == False):
			cv2.imwrite('saccade_paths/ior_image_%d.png'%image, 255*ior_map)
			#fix_x, fix_y = 860-40, 544-40 ### update for image:0
			saccade_path[counter,0] =  fix_x #860
			saccade_path[counter,1] =  fix_y #544
			## ********** update ior *********************************
			zz_current = zz * 1 #np.tanh(0.05*counter)
			ior_map_pad[50+fix_y-50:50+fix_y+50, 50+fix_x-50:50+fix_x+50] += zz_current
			ior_map = ior_map_pad[50:-50,50:-50]
			#update_ior_map(fix_x,fix_y)
			found = True if diff[fix_y,fix_x] else False
			over = True if counter >= (120+1) else False
			print(('saccade:',counter,'loc:',fix_x,fix_y,'found ?:',found,'over ?:',over))
			##********** Compute saliency in polar coordinate ********
			vpi_sal = np.zeros((size, 768, 1024))
			vpi_sal_real = np.zeros((size, 768, 1024))
			mrp_blur, jj = [], []
			for j in range(size):
				if input_seq[j].max() != 0:
					mrp_blur.append(input_seq[j].astype(np.uint8)) 
					#mrp_blur.append(blur_image_approx(input_seq[j],fix_x,fix_y))  
					jj.append(j)
			mrp_blur_sc1, mrp_blur_sc2, SC_ior_map, mapping_mat_r = cvr_transform(mrp_blur[0], mrp_blur[1], ior_map, fix_x, fix_y, 1024, 768)
			cv2.imwrite('saccade_paths/current_fovea_%d.png'%image, mrp_blur_sc1.astype(np.uint8)[:,:,::-1])
			#plt.imshow(mrp_blur_sc1.astype(np.uint8))
			#plt.show()
			if (counter%2):
				vpi_sal_real[jj[0]], vpi_sal[jj[0]] = computesaliency(mrp_blur_sc1.astype(np.uint8))
				vpi_sal_real[jj[1]], vpi_sal[jj[1]] = computesaliency(mrp_blur_sc2.astype(np.uint8))
			else:
				vpi_sal_real[jj[0]], vpi_sal[jj[0]] = computesaliency(mrp_blur_sc2.astype(np.uint8))
				vpi_sal_real[jj[1]], vpi_sal[jj[1]] = computesaliency(mrp_blur_sc1.astype(np.uint8))
			#vpi_sal[jj[0]] = gaussian_blur(computesaliency(mrp_blur_sc1), 12)
			#vpi_sal[jj[1]] = gaussian_blur(computesaliency(mrp_blur_sc2), 12)
			data = scale(np.asarray(vpi_sal),0,1).astype(np.float32)
			M_1, M_2, M_3, M_4, M_g = create_batch_input(data)
			##********** execution of STRNN ****************
			STRNN_output_per, STRNN_output_cd = tiled_STRNN_direct(M_1, M_2, M_3, M_4, M_g)
			#####
			'''
			plt.matshow(STRNN_output_cd[0],cmap=plt.cm.RdYlBu,vmin=-1.0,vmax = 1.0)
			plt.show()
			plt.matshow(STRNN_output_cd[1],cmap=plt.cm.RdYlBu,vmin=-1.0,vmax = 1.0)
			plt.show()
			plt.matshow(STRNN_output_cd[2],cmap=plt.cm.RdYlBu,vmin=-1.0,vmax = 1.0)
			plt.show()
			plt.matshow(STRNN_output_cd[3],cmap=plt.cm.RdYlBu,vmin=-1.0,vmax = 1.0)
			plt.show()
			'''
			SC_diff	= STRNN_output_cd[num_distractors+1]
			strength_cd, strength_ior = 0.72, 0.7
			#SC_saliency = scale(vpi_sal_real[jj[1]] * (SC_diff*strength_cd + (1-strength_cd)), 0, 1)
			#SC_saliency =  scale( ((SC_diff<0.45)*vpi_sal_real[jj[1]] + (SC_diff>0.45)*0.95) , 0, 1)
			SC_saliency = scale(vpi_sal_real[jj[1]] + vpi_sal_real[jj[0]], 0, 1)
			result = SC_saliency*( (1-SC_ior_map)  )
			result[result<0] = 0
			## Remove noisy prediction ## updated
			#aa = cv2.erode((SC_diff>0.75).astype(np.float32), np.ones((5,5), np.uint8), iterations=2)
			#bb = cv2.dilate(aa, np.ones((5,5), np.uint8), iterations=3)
			#SC_diff = bb
			#SC_saliency = scale((vpi_sal[jj[0]]+vpi_sal[jj[1]]).clip(max=255.0), 0, 1)
			#result = (SC_saliency - 2*SC_ior_map).clip(min=0.0)
			#result = result + SC_diff 
			#plt.imshow(result)
			#plt.savefig('result_pdf_%d.png'%image)
			#tmp = apply_WTA(pad_diff,Wg,2500)
			counter += 1 
			x_t,y_t = calc_saccade_vector(result)
			fix_x, fix_y = reverse_cvr_transform(mapping_mat_r, x_t, y_t)
		np.savetxt('saccade_paths/cartesian_global_new_free_120_fixations/sc_trial_%d_image_%d.csv'%(trial_id, num), saccade_path[0:counter])
