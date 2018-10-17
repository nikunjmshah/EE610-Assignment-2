''' 

User info: All part of code is self-written and original by Nikunj Shah. 
Multiple sources from web were referenced for understanding dft, idft, fft functions
but nothing is copied as given.
Funtions itself were written from scratch wherever necessary.


'''

# USAGE
# python demo.py
# for gamma, K and radius values can be given as text input in gui


# import the necessary packages
from Tkinter import *
import tkMessageBox
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from numpy import linalg as LA
from skimage.measure import compare_ssim as ssim

# initialize global variables
imagefft_blur_b = None
imagefft_blur_g = None
imagefft_blur_r = None
kernel = None
kernelfft = None
sign = None
radius = 5
K = 1000000
gamma = 1000000
result = 0


image_orig = None
image_curr = None
image_prev = None

radius_in = None # variable to capture input gamma value
K_in = None # variable to capture input blurriness value
gamma_in = None # variable to capture input sharpness value
path = None # path of input image
# creating panels for displaying original and edited image
panelA = None
panelB = None


def iexp(n):
    '''
    Converts phase to a complex quantity.
    '''
    return complex(math.cos(n), math.sin(n))

def is_pow2(n):
    '''
    Checks if a number is a power of 2
    '''
    return False if ((n & 1) == 1 and n != 1) else (n == 1 or is_pow2(n >> 1))

def dft(x):
    '''
    Naive implementation of FFT of a sequence x
    '''
    n = len(x)
    X = [0+0j]*n
    for k in range(n):
        X[k] = sum( x[i] * iexp(-2 * math.pi * k * i / n) for i in range(n))
    return X

def idft(X):
    '''
    Naive implementation of inverse FFT of a sequence X
    '''
    n = len(X)
    x = [0+0j]*n
    for k in range(n):
        x[k] = sum( X[i] * iexp(2 * math.pi * k * i / n) for i in range(n)) / n
    return x

def fft(x):
    '''
    Recursive implementation of FFT of a sequence x
    '''
    n = len(x)
    if is_pow2(n) == False:
        raise ValueError("Size of x must be a power of 2")
    elif n == 1:
        return [x[0]]
    else:
        # Splitting in even-odd sequences according to Tukey-cooley algoritm
        x = fft(x[::2]) + fft(x[1::2])
        for i in range(n//2):
            e = iexp(-2 * math.pi * i / n)
            x_i = x[i]
            x[i] = x_i + e * x[i + n//2]
            x[n//2 + i] = x_i - e * x[i + n//2]
        return x
        
def ifft(X):
    '''
    Recursive implementation of inverse FFT of a sequence X
    '''
    n = len(X)
    if is_pow2(n) == False:
        raise ValueError("Size of x must be a power of 2")
    elif n == 1:
        return [X[0]]
    else:
        # Splitting in even-odd sequences according to Tukey-cooley algoritm
        X = ifft(X[::2]) + ifft(X[1::2])
        for i in range(n//2):
            e = iexp(2 * math.pi * i / n)
            X_i = X[i]
            X[i] = X_i + e * X[i + n//2]
            X[n//2 + i] = X_i - e * X[i + n//2]
        return [val/2 for val in X]

def pad_image_2(image, pad_height, pad_width):
    '''
    Given an image, this function will pad it with zeros to 
    image.shape + (pad_height, pad_width) + (nearest power of 2)
    '''
    P = pow2_ceil(image.shape[0] + pad_height)
    Q = pow2_ceil(image.shape[1] + pad_width)
    image_padded = np.zeros((P,Q))
    image_padded[:image.shape[0], :image.shape[1]] = image
    return image_padded

def dft_2d(image):

    # Initialize DFT in matrix form of image
    # The image is also padded to the nearest power of 2 for speedy computations
    image = pad_image_2(image, 0, 0)
    imWidth = int(image.shape[1])
    imHeight = int(image.shape[0])
    image_fft = np.zeros((imHeight, imWidth), dtype=np.complex_)
    
    # Row-wise DFT calculation
    for x in range(imHeight):
        image_fft[x] = fft(image[x])
        
    # Column-wise DFT calculation
    image_fft_t = image_fft.transpose()
    for y in range(imWidth):
        image_fft_t[y] = fft(image_fft_t[y])
    image_fft = image_fft_t.transpose()
    
    return image_fft

def idft_2d(image_fft):

    # Initialize IDFT in matrix form of image
    imWidth = int(image_fft.shape[1])
    imHeight = int(image_fft.shape[0])
    image = np.zeros((imHeight, imWidth), dtype=np.complex_)

    # Row-wise IDFT calculation
    for x in range(imHeight):
        image[x] = ifft(image_fft[x])

    # Column-wise IDFT calculation
    image_t = image.transpose()
    for y in range(imWidth):
        image_t[y] = ifft(image_t[y])
    image = image_t.transpose()

    return image

def pow2_ceil(x):
    return 2 ** int(np.ceil(np.log2(x)))

def shift_dft(image_fft):
    '''
    Shift the fourier transform so that F(0,0) is in the center.
    '''
    # Typecasting as array
    y = np.asarray(image_fft)  
    for k, n in enumerate(image_fft.shape):
        mid = (n + 1) / 2
        indices = np.concatenate((np.arange(mid, n), np.arange(mid)))
        y = np.take(y, indices, k) 
    return y

def filter1():
    global imagefft_blur_b, imagefft_blur_g, imagefft_blur_r, kernelfft, sign, image_curr, image_prev, kernel, result

    image_prev = image_curr

    imagefinal = image_curr.copy()
    data = (image_curr[:,:,0]).copy()

    # plt.imshow(image_curr[:,:,0], cmap='gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    M, N = data.shape
    # print(M)
    # print(N)
    m, n = kernel.shape
    P, Q = (pow2_ceil(m + M - 1), pow2_ceil(n + N - 1))

    X, Y = np.meshgrid(np.arange(Q), np.arange(P))
    sign = np.power(-1, X + Y)

    fpad = pad_image_2(data, m-1, n-1)
    fpad = sign * fpad

    kpad = pad_image_2(kernel, M-1, N-1)
    kpad = sign * kpad

    fstar = dft_2d(fpad)
    kernelfft = np.abs(dft_2d(kpad))

    kstar = kernelfft
    #plot_DFT(fstar)
    #plot_DFT(kstar)
    #print(result)
    if result == 'no':
        imagefinal1 = (np.array(idft_2d(fstar * kstar)).real * sign)[:M,:N]
        fpad = pad_image_2(imagefinal1, m-1, n-1)
        imagefft_blur_b = dft_2d(sign * fpad)
        imagefinal1 = np.abs(imagefinal1)
        imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255  
        imagefinal[:,:,0] = imagefinal1.copy()
    else:
        imagefft_blur_b = fstar.copy()


    #for green channel
    data = (image_curr[:,:,1]).copy()
    fpad = pad_image_2(data, m-1, n-1)
    fpad = sign * fpad
    fstar = dft_2d(fpad)
    if result == 'no':
        imagefinal1 = (np.array(idft_2d(fstar * kstar)).real * sign)[:M,:N]
        fpad = pad_image_2(imagefinal1, m-1, n-1)
        imagefft_blur_g = dft_2d(sign * fpad)
        imagefinal1 = np.abs(imagefinal1)
        imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255    
        imagefinal[:,:,1] = imagefinal1.copy()
    else:
        imagefft_blur_g = fstar.copy()
    # plt.imshow((imagefinal[:,:,1]), cmap='gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    #for red channel
    data = (image_curr[:,:,2]).copy()
    fpad = pad_image_2(data, m-1, n-1)
    fpad = sign * fpad
    fstar = dft_2d(fpad)
    if result == 'no':
        imagefinal1 = (np.array(idft_2d(fstar * kstar)).real * sign)[:M,:N]
        fpad = pad_image_2(imagefinal1, m-1, n-1)
        imagefft_blur_r = dft_2d(sign * fpad)
        imagefinal1 = np.abs(imagefinal1)
        imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255 
        imagefinal[:,:,2] = imagefinal1.copy()
    else:
        imagefft_blur_r = fstar.copy()
    print('done')

    #print(imagefinal)

    # print(imagefinal.shape)
    # print(max(imagefinal.flatten()))
    image_curr = imagefinal
    imagefinal = Image.fromarray(imagefinal)
    #print(imagefinal)
    imagefinal = ImageTk.PhotoImage(imagefinal)
    panelA.configure(image=imagefinal)
    panelA.image = imagefinal


def inv_filter():
    global kernelfft, sign, image_curr, image_prev, imagefft_blur_b, imagefft_blur_g, imagefft_blur_r, image_orig

    image_prev = image_curr

    M, N, C = image_curr.shape
    imagefinal = image_curr.copy()
    kstar = kernelfft

    # for blue
    fstar = imagefft_blur_b.copy()    
    imagefinal1 = (np.array(idft_2d(np.divide(fstar, kstar))).real * sign)[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,0] = imagefinal1.copy()

    # for green
    fstar = imagefft_blur_g.copy()    
    imagefinal1 = (np.array(idft_2d(np.divide(fstar, kstar))).real * sign)[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,1] = imagefinal1.copy()

    # for red
    fstar = imagefft_blur_r.copy()    
    imagefinal1 = (np.array(idft_2d(np.divide(fstar, kstar))).real * sign)[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,2] = imagefinal1.copy()

    print('PSNR:')
    print(PSNR(image_orig,imagefinal))
    print('SSIM:')
    print(ssim(image_orig,imagefinal, data_range = image_orig.max() - image_orig.min(), multichannel = True))

    image_curr = imagefinal
    imagefinal = Image.fromarray(imagefinal)
    imagefinal = ImageTk.PhotoImage(imagefinal)
    panelA.configure(image=imagefinal)
    panelA.image = imagefinal

def trunc_inv():
    # grab required global references
    global imagefft_blur_b, imagefft_blur_g, imagefft_blur_r, kernelfft, sign, image_curr, image_prev, radius_in, radius, image_orig

    # update prev image
    image_prev = image_curr

    # check for input
    if radius_in.get() == "":
    	# display error if nothing given
    	tkMessageBox.showwarning("Error","Type input value in required field")
    else:
    	# update gamma if there is any input
    	radius = float(radius_in.get())

    M, N, C = image_curr.shape
    m, n = kernel.shape
    P, Q = (pow2_ceil(m + M - 1), pow2_ceil(n + N - 1))

    x = np.arange(0, P)
    y = np.arange(0, Q)
    mask = (x[np.newaxis,:]-(P/2))**2 + (y[:,np.newaxis]-(Q/2))**2 < radius**2

    imagefinal = image_curr.copy()
    kstar = kernelfft

    # for blue
    fstar = imagefft_blur_b.copy() 
    imagefinal1 = fstar
    imagefinal1[mask] = np.divide((fstar[mask]),(kstar[mask]))   
    imagefinal1 = (np.array(idft_2d(imagefinal1).real * sign))[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,0] = imagefinal1.copy()

    # for green
    fstar = imagefft_blur_g.copy()    
    imagefinal1 = fstar
    imagefinal1[mask] = np.divide((fstar[mask]),(kstar[mask]))   
    imagefinal1 = (np.array(idft_2d(imagefinal1).real * sign))[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,1] = imagefinal1.copy()

    # for red
    fstar = imagefft_blur_r.copy()    
    imagefinal1 = fstar
    imagefinal1[mask] = np.divide((fstar[mask]),(kstar[mask]))   
    imagefinal1 = (np.array(idft_2d(imagefinal1).real * sign))[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,2] = imagefinal1.copy() 

    print('PSNR:')
    print(PSNR(image_orig,imagefinal))
    print('SSIM:')
    print(ssim(image_orig,imagefinal, data_range = image_orig.max() - image_orig.min(), multichannel = True))

    image_curr = imagefinal
    imagefinal = Image.fromarray(imagefinal)
    imagefinal = ImageTk.PhotoImage(imagefinal)
    panelA.configure(image=imagefinal)
    panelA.image = imagefinal

def weiner():
    # grab required global references
    global imagefft_blur_b, imagefft_blur_g, imagefft_blur_r, kernelfft, sign, image_curr, image_prev, K_in, K, image_orig

    # update prev image
    image_prev = image_curr

    # check for input
    if K_in.get() == "":
    	# display error if nothing given
    	tkMessageBox.showwarning("Error","Type input value in required field")
    else:
    	# update gamma if there is any input
    	K = float(K_in.get())

    M, N, C = image_curr.shape
    imagefinal = image_curr.copy()
    kstar = kernelfft
    kstar_abs = np.abs(kernelfft)
    coeff = np.divide(kstar_abs**2,(kstar_abs**2 + K))

    # for blue
    fstar = imagefft_blur_b    
    imagefinal1 = (np.array(idft_2d(coeff * np.divide(fstar, kstar)).real * sign))[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,0] = imagefinal1.copy()

    # for green
    fstar = imagefft_blur_g    
    imagefinal1 = (np.array(idft_2d(coeff * np.divide(fstar, kstar)).real * sign))[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,1] = imagefinal1.copy()

    # for red
    fstar = imagefft_blur_r    
    imagefinal1 = (np.array(idft_2d(coeff * np.divide(fstar, kstar)).real * sign))[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,2] = imagefinal1.copy() 

    print('PSNR:')
    print(PSNR(image_orig,imagefinal))
    print('SSIM:')
    print(ssim(image_orig,imagefinal, data_range = image_orig.max() - image_orig.min(), multichannel = True))

    image_curr = imagefinal
    imagefinal = Image.fromarray(imagefinal)
    imagefinal = ImageTk.PhotoImage(imagefinal)
    panelA.configure(image=imagefinal)
    panelA.image = imagefinal

def constrained_ls():

    # grab required global references
    global kernelfft, sign, image_curr, image_prev, imagefft_blur_b, imagefft_blur_g, imagefft_blur_r, gamma_in, gamma, image_orig

    # update prev image
    image_prev = image_curr

    # check for input
    if gamma_in.get() == "":
    	# display error if nothing given
    	tkMessageBox.showwarning("Error","Type input value in required field")
    else:
    	# update gamma if there is any input
    	gamma = float(gamma_in.get())

    M, N, C = image_curr.shape
    m, n = kernel.shape
    P, Q = (pow2_ceil(m + M - 1), pow2_ceil(n + N - 1))

    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplac_pad = pad_image_2(laplacian, P-3, Q-3)
    laplac_pad = laplac_pad * sign

    imagefinal = image_curr.copy()
    kstar = kernelfft
    lstar = dft_2d(laplac_pad)
    coeff = np.divide(np.conjugate(kstar), (np.abs(kstar)**2 + gamma * np.abs(lstar)**2))

    # for blue
    fstar = imagefft_blur_b    
    imagefinal1 = (np.array(idft_2d(np.multiply(fstar, coeff))).real * sign)[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,0] = imagefinal1.copy()

    # for green
    fstar = imagefft_blur_g    
    imagefinal1 = (np.array(idft_2d(np.multiply(fstar, coeff))).real * sign)[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,1] = imagefinal1.copy()

    # for red
    fstar = imagefft_blur_r    
    imagefinal1 = (np.array(idft_2d(np.multiply(fstar, coeff))).real * sign)[:M, :N]
    imagefinal1 = np.abs(imagefinal1)
    imagefinal1 = (imagefinal1/(max(imagefinal1.flatten())))*255.0
    imagefinal[:,:,2] = imagefinal1.copy() 

    print('PSNR:')
    print(PSNR(image_orig,imagefinal))
    print('SSIM:')
    print(ssim(image_orig,imagefinal, data_range = image_orig.max() - image_orig.min(), multichannel = True))

    image_curr = imagefinal
    imagefinal = Image.fromarray(imagefinal)
    imagefinal = ImageTk.PhotoImage(imagefinal)
    panelA.configure(image=imagefinal)
    panelA.image = imagefinal


def MSE(image, image_rec):
    return ((LA.norm(image - image_rec)))/(image.shape[0] * image.shape[1] * 3)

def PSNR(image, image_rec):
    return 10 * np.log10(255**2/MSE(image, image_rec))

def plot_DFT(image_fft):
    '''
    Surface plot of the magnitude spectrum of given image.
    '''
    x = np.arange(0, image_fft.shape[0], 1)
    y = np.arange(0, image_fft.shape[1], 1)
    X, Y = np.meshgrid(x, y)    

    fig = plt.figure()
    
    magnitude_spectrum = 20*np.log(np.abs(image_fft))
    Z = magnitude_spectrum.reshape(X.shape)
    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    phase_spectrum = np.angle(image_fft)
    Z = phase_spectrum.reshape(X.shape)
    ax = fig.add_subplot(222, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 2D plot of magnitude and phase spectrum of 2D-DFT
    plt.subplot(223),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum (Log scale)')#, plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(phase_spectrum, cmap = 'gray')
    plt.title('Phase Spectrum (Log scale)')#, plt.xticks([]), plt.yticks([])

    plt.show()

# function to select image file
def select_image():
	# grab a reference to required global variables
    global panelA, panelB, image_orig, image_curr, image_prev, path, result

    result = tkMessageBox.askquestion("Input type", "Is input already blurred?", icon='warning')

    # open a file chooser dialog and allow the user to select an input image
    path = tkFileDialog.askopenfilename()

    # ensure a file path was selected
    if (len(path)) > 0:

        # note: below method handles grayscale images too as opencv imread copies 
        # grayscale image image in all 3 channel by default and reads them as 3 channel image

        # load the input image and logo from disk, abstract value channel
        image = cv2.imread(path)

        # converting to hsv format and abstracting channel 'V'
        image_orig = image

        # setting global variables appropriately
        image_prev = image_orig
        image_curr = image_orig
        
        # imagefft_b = dft_2d(b)
        # imagefft_g = dft_2d(g)
        # imagefft_r = dft_2d(r)
        # convert the images to PIL format
        image = Image.fromarray(image_curr)
        # convert to ImageTk format to put in tkinter window
        image = ImageTk.PhotoImage(image)

        # display/update images in appropriate panels
        panelA.configure(image=image)
        panelA.image = image
        panelB.configure(image=image)
        panelB.image = image



def select_kernel():
    # grab a reference to required global variables
    global kernel

    # open a file chooser dialog and allow the user to select an input image
    path = tkFileDialog.askopenfilename()

    # ensure a file path was selected
    if (len(path)) > 0:

        # note: below method handles grayscale images too as opencv imread copies 
        # grayscale image image in all 3 channel by default and reads them as 3 channel image

        # load the input image and logo from disk, abstract value channel
        kernel = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# single undo function
def undo():
	# grab required global references
	global panelA, image_curr, image_prev

	# copy previous image and make it final
	imagefinal = image_prev

	# update and display 'edited' image
	image_curr = imagefinal
	imagefinal = Image.fromarray(imagefinal)
	imagefinal = ImageTk.PhotoImage(imagefinal)
	panelA.configure(image=imagefinal)
	panelA.image = imagefinal

# function to revert to original image
def undo_all():
	# grab required global references
	global panelA, image_curr, image_prev, imageorig

	# copy original image and make it final
	imagefinal = image_orig

	# update and display 'edited' image
	image_curr = imagefinal
	image_prev = imagefinal
	imagefinal = Image.fromarray(imagefinal)
	imagefinal = ImageTk.PhotoImage(imagefinal)
	panelA.configure(image=imagefinal)
	panelA.image = imagefinal

# function to save image
def save():
	# grab required global references
	global image_curr

	# use cv2 function to write current image
	cv2.imwrite(path.replace(".","_edited."), image_curr)



# initialize the window toolkit 
root = Tk()
root.title("My editor")

# display tip
# tkMessageBox.showinfo("Tip","Please type necessary input values before \
# 	pressing the related transformation button")


#  create blank image to display as default in both panels
image = np.zeros((400,400), np.uint8)

# modify image to tkinter format
image = Image.fromarray(image)
image = ImageTk.PhotoImage(image)

# initialize panel positions, labels and display image for both panels
panelA = Label(text ="Edited",image=image, compound=TOP)
panelA.image = image
panelA.pack(side="right", padx=10, pady=10)
panelB = Label(text ="Original",image=image, compound=TOP)
panelB.image = image
panelB.pack(side="left", padx=10, pady=10)

# creating necessary buttons (names are informative)
# first line creates the button and assigns the callback function as passed in
# 'command' parameter second line positions the button appropraitely in GUI

btn_save = Button(root, text="Save image", command=save)
btn_save.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")

btn_select = Button(root, text="Select an image", command=select_image)
btn_select.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")

btn_selectk = Button(root, text="Select kernel", command=select_kernel)
btn_selectk.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")

btn_his = Button(root, text="Full inverse", command=inv_filter)
btn_his.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")

btn_gamma = Button(root, text="Truncated inverse", command=trunc_inv)
btn_gamma.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")

btn_log = Button(root, text="Weiner filter", command=weiner)
btn_log.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")

btn_blur = Button(root, text="Initialize Image", command=filter1)
btn_blur.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")

btn_sharp = Button(root, text="Constrained LS", command=constrained_ls)
btn_sharp.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")

btn_undo = Button(root, text="Undo", command=undo)
btn_undo.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")

btn_undoall = Button(root, text="Undo all", command=undo_all)
btn_undoall.pack(side="bottom", fill="both", expand="yes", padx="10", pady="2")


# creating text inputs and labels for values to be taken from user as input

radius_lab = Label(root, text="Truncation")
radius_lab.pack(side="top", fill="both", expand="yes", padx="10", pady="1")

radius_in = Entry(root)
radius_in.pack(side="top", fill="both", expand="yes", padx="10", pady="1")

K_lab = Label(root, text="K")
K_lab.pack(side="top", fill="both", expand="yes", padx="10", pady="1")

K_in = Entry(root)
K_in.pack(side="top", fill="both", expand="yes", padx="10", pady="1")

gamma_lab = Label(root, text="gamma")
gamma_lab.pack(side="top", fill="both", expand="yes", padx="10", pady="1")

gamma_in = Entry(root)
gamma_in.pack(side="top", fill="both", expand="yes", padx="10", pady="1")

# kick off the GUI
root.mainloop()
