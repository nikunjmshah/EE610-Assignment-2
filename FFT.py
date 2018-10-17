import math
import numpy as np
import pylab as py
from scipy import fftpack
import scipy.misc
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from numpy import linalg as LA
from skimage.measure import compare_ssim as ssim

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
    ''' 
    2D FFT: Implementation of FFT in 2D is analogous to the 1D case
    as 2D-DFT is separable transformation.
    Note that the dimensions of image have to be a multiple of 2 individually.
    Also note, here centered DFT of given image is computed. 
    '''
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
    '''
    2D Inverse FFT: Implementation of iFFT in 2D is analogous to the 1D case
    as 2D-iDFT is separable transformation.
    Note that the dimensions of image have to be a multiple of 2 individually.
    '''
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

def filter(data, kernel):
    M, N = data.shape
    m, n = kernel.shape
    P, Q = (pow2_ceil(m + M - 1), pow2_ceil(n + N - 1))

    X, Y = np.meshgrid(np.arange(Q), np.arange(P))
    sign = np.power(-1, X + Y)

    fpad = pad_image_2(data, m-1, n-1)
    fpad = sign * fpad

    kpad = pad_image_2(kernel, M-1, N-1)
    kpad = sign * kpad

    fstar = dft_2d(fpad)
    kstar = np.abs(dft_2d(kpad))
    # plot_DFT(fstar)
    # plot_DFT(kstar)
    return (np.array(idft_2d(fstar * kstar)).real * sign)[:M, :N]

def inv_filter(data, kernel):
    M, N = data.shape
    m, n = kernel.shape
    P, Q = (pow2_ceil(m + M - 1), pow2_ceil(n + N - 1))

    X, Y = np.meshgrid(np.arange(Q), np.arange(P))
    sign = np.power(-1, X + Y)

    fpad = pad_image_2(data, m-1, n-1)
    fpad = sign * fpad

    kpad = pad_image_2(kernel, M-1, N-1)
    kpad = sign * kpad

    fstar = dft_2d(fpad)
    kstar = np.abs(dft_2d(kpad))
    return (np.array(idft_2d(np.divide(fstar, kstar))).real * sign)[:M, :N]

def trunc_inv(data, kernel, radius):
    M, N = data.shape
    m, n = kernel.shape
    P, Q = (pow2_ceil(m + M - 1), pow2_ceil(n + N - 1))

    X, Y = np.meshgrid(np.arange(Q), np.arange(P))
    sign = np.power(-1, X + Y)

    fpad = pad_image_2(data, m-1, n-1)
    fpad = sign * fpad

    kpad = pad_image_2(kernel, M-1, N-1)
    kpad = sign * kpad

    x = np.arange(0, P)
    y = np.arange(0, Q)
    mask = (x[np.newaxis,:]-(P/2))**2 + (y[:,np.newaxis]-(Q/2))**2 < radius**2

    fstar = dft_2d(fpad)
    kstar = np.abs(dft_2d(kpad))
    fstar_final = fstar
  
    fstar_final[mask] = np.divide((fstar[mask]),(kstar[mask]))
    return (np.array(idft_2d(fstar_final).real * sign))[:M, :N]


def weiner(data, kernel, K):
    M, N = data.shape
    m, n = kernel.shape
    P, Q = (pow2_ceil(m + M - 1), pow2_ceil(n + N - 1))

    X, Y = np.meshgrid(np.arange(Q), np.arange(P))
    sign = np.power(-1, X + Y)

    fpad = pad_image_2(data, m-1, n-1)
    fpad = sign * fpad

    kpad = pad_image_2(kernel, M-1, N-1)
    kpad = sign * kpad

    fstar = dft_2d(fpad)
    kstar = dft_2d(kpad)
    kstar_abs = np.abs(kstar)
    coeff = np.divide(kstar_abs**2,(kstar_abs**2 + K))
    return (np.array(idft_2d(coeff * np.divide(fstar, kstar_abs)).real * sign))[:M, :N]

def constrained_ls(data, kernel, gamma):
    M, N = data.shape
    m, n = kernel.shape
    P, Q = (pow2_ceil(m + M - 1), pow2_ceil(n + N - 1))

    X, Y = np.meshgrid(np.arange(Q), np.arange(P))
    sign = np.power(-1, X + Y)

    fpad = pad_image_2(data, m-1, n-1)
    fpad = sign * fpad

    kpad = pad_image_2(kernel, M-1, N-1)
    kpad = sign * kpad

    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplac_pad = pad_image_2(laplacian, P-3, Q-3)
    laplac_pad = laplac_pad * sign

    fstar = dft_2d(fpad)
    kstar = (dft_2d(kpad))
    lstar = dft_2d(laplac_pad)
    coeff = np.divide(np.abs(kstar), (np.abs(kstar)**2 + gamma * np.abs(lstar)**2))
    return (np.array(idft_2d(np.multiply(fstar, coeff))).real * sign)[:M, :N]

def MSE(image, image_rec):
    return (LA.norm(image - image_rec))/(image.shape[0] * image.shape[1])

def PSNR(image, image_rec):
    return 10 * np.log10(255**2/MSE(image, image_rec))

def SSIM(image, image_rec):
    return ssim(image, image_rec, data_range=image.max() - image.min())

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



# image = np.array([[1,2,3, 5], [4,6,7,9], [8,1,7,2], [1,1,1,1]])
# print(image)
# print('--------')
# print(dft_2d(image))
# print('--------')
# f = np.fft.fft2(image)
# print(f)
# print('--------')
# # print('_______________________')
# # print(fft(image[0]))
# # print(fft(image[1]))
# # print(fft(image[2]))
# # print(fft(image[3]))
# # print('_______________________')
# print('--------')
# a = iDFT_2d(dft_2d(image))
# # for i in range(image.shape[0]):
# #     for j in range(image.shape[1]):
# #         a[i, j] = a[i,j]* ((-1)**(i+j))
# print([v.real for v in a])

# def test(data, my_func, lib_func, all_right, name):
#     """Check if my implementation is close to the one in the library."""
#     my_result, lib_result = my_func(data), lib_func(data)
#     error = np.abs(lib_result - my_result)
#     error_range = (np.min(error), np.max(error))
#     if all_right(lib_result, my_result):
#         print ("[PASSED] %s, " % name)
#         print ("Error in (%.6e, %.6e)" % error_range)
#     else:
#         print ("[FAILED] %s" % name)

# if __name__ == "__main__":
#     lena = np.reshape(misc.lena(), (1024, 256))
#     data = np.random.rand(1024)

#     # test(lena, shift_dft, fftpack.fftshift, np.array_equal, 'Shift')
#     test(lena, dft_2d, fftpack.fft2, np.allclose, '2D-DFT')
#     test(lena, idft_2d, fftpack.ifft2, np.allclose, '2D-IDFT')
#     # test(lena, get_fft, fftpack.fft2, np.allclose, '2D-FFT')
#     # test(lena, get_ifft, fftpack.ifft2, np.allclose, '2D-IFFT')

# data = np.random.rand(1024, 1000)
# ker = np.random.rand(200, 200)

# a = filter(data, ker)
# print(a)

image = cv2.imread('/home/suyashbagad/Desktop/Sem 7/EE 610/Assignment 2/src/test.jpeg',  cv2.IMREAD_GRAYSCALE)
kernel = cv2.imread('/home/suyashbagad/Desktop/Sem 7/EE 610/Assignment 2/src/Cho_Deblur/Blurry1_9_result_Cho_1.png.psf.png', cv2.IMREAD_GRAYSCALE)
X, Y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
sign = np.power(-1, X + Y)

# P, Q = (pow2_ceil(kernel.shape[0]), pow2_ceil(kernel.shape[1]))
# ker = np.zeros((P,Q))
# ker[:kernel.shape[0], :kernel.shape[1]] = kernel

'''
# Machaya hai bhai nacho bc
image = image * sign
image_dft = dft_2d(image)
plot_DFT(image_dft)

image_rec = np.array(idft_2d(image_dft)).real
image_rec = image * sign
plt.imshow(image_rec, cmap='gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()
'''
# X, Y = np.meshgrid(np.arange(kernel.shape[1]), np.arange(kernel.shape[0]))
# print(X)
# sign = np.power(-1, X + Y)
# print(kernel.shape)
# print(sign.shape)
# print(kernel[200])
# kernel1 = np.multiply(sign,kernel)
# print(kernel[200])

# ker_dft = dft_2d(kernel1)
# plot_DFT(ker_dft)
t = time.clock()
image_fil = filter(image, kernel)
print(time.clock() - t)
# plt.imshow(image_fil, cmap='gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])
# plt.show()

image_fulli = inv_filter(image_fil, kernel) 

image_res = trunc_inv(image_fil, kernel,8)

image_res_weiner = weiner(image_fil, kernel, 3000000)

image_ls = constrained_ls(image_fil, kernel, 1000000)

fig = plt.figure(figsize=(20, 60))
ax = fig.add_subplot(221)
plt.imshow(image_fil, cmap='gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])

ax = fig.add_subplot(222)
plt.imshow(image_fulli, cmap='gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])

ax = fig.add_subplot(223)
plt.imshow(image_ls, cmap='gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])

ax = fig.add_subplot(224)
plt.imshow(image_res_weiner, cmap='gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()



