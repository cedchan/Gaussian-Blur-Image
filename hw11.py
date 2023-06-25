import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.fft import fft, ifft


img = image.imread('pic.jpeg')

img = img / 0xff
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]


def gauss(n, sigma):
    r = np.arange(-int(n / 2), int(n / 2) + 1)
    return 1 / (sigma * math.sqrt(2 * math.pi)) * np.exp(-r ** 2 / (2 * sigma ** 2))


def conv(a, b):
    n = max(len(a), len(b))
    a1 = np.zeros(n)
    a1[:len(a)] = a
    b1 = np.zeros(n)
    b1[:len(b)] = b
    return np.real(ifft(fft(a1) * fft(b1)))


def blur(r, g, b, n, sigma):
    filt = gauss(n, sigma)
    r1 = conv(r.ravel(), filt).reshape(r.shape)
    g1 = conv(g.ravel(), filt).reshape(g.shape)
    b1 = conv(b.ravel(), filt).reshape(b.shape)
    return r1, g1, b1


def blur_im(r, g, b, n, sigma):
    r1, g1, b1 = blur(r, g, b, n, sigma)
    r2, g2, b2 = blur(r1.T, g1.T, b1.T, n, sigma)
    return np.stack((r2.T, g2.T, b2.T), axis=2)


plt.subplot(1, 3, 1)
plt.imshow(blur_im(R, G, B, 100, 5))
plt.subplot(1, 3, 2)
plt.imshow(blur_im(R, G, B, 100, 10))
plt.subplot(1, 3, 3)
plt.imshow(blur_im(R, G, B, 100, 50))

plt.show()

# I use sigma 5, 10, then 50, as displayed above. As sigma becomes larger, the image becomes more blurred, and at the
# end appears to get darker as well. The greater blur makes sense because sigma increases the spread of the gaussian
# function, meaning that in a given range the values will be more distributed, resulting in blur. The color change
# could be from a similar reason, perhaps that having overlapped copies of same pixel because of the flat Gaussian,
# the composite color will become dark.
