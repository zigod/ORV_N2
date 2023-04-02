import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
#plt.ion()

def my_roberts(slika):
    kernelv = np.array( [[1, 0 ], [0,-1 ]] )
    kernelh = np.array( [[ 0, 1 ], [ -1, 0 ]] )

    vertical = cv2.filter2D( slika, -1, kernelv )
    horizontal = cv2.filter2D( slika, -1, kernelh )

    slika_robov = cv2.add(vertical, horizontal)
    slika_robov*=3
    
    return slika_robov

def my_prewitt(slika):
    kernelv = np.array( [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] )
    kernelh = np.array( [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] )

    vertical = cv2.filter2D(slika, -1, kernelv)
    horizontal = cv2.filter2D(slika, -1, kernelh)

    slika_robov = cv2.add(vertical, horizontal)
    #slika_robov*=3

    return slika_robov

def my_sobel(slika):
    kernelv = np.array( [[-1, 0, 1], [-2, 0, -2], [-1, 0, -1]] )
    kernelh = np.array( [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] )

    vertical = cv2.filter2D(slika, -1, kernelv)
    horizontal = cv2.filter2D(slika, -1, kernelh)

    slika_robov = cv2.add(vertical, horizontal)
    #slika_robov*=3
    
    return slika_robov

def canny(slika, sp_prag, zg_prag):
    slika_robov = cv2.Canny(slika, sp_prag, zg_prag)
    slika_robov = cv2.cvtColor(slika_robov, cv2.COLOR_BGR2RGB)
    return slika_robov 

def spremeni_kontrast(slika, alfa, beta):
    x, y, _ = slika.shape
    for i in range(0, x, 1):
        for j in range(0, y, 1):
            slika[i, j] = alfa * slika[i, j] + beta
    return slika

fig = plt.figure(figsize =(10, 7))
rows = 2
columns = 4

bgr_slika = cv2.imread('image.jpg',0)
slika = cv2.cvtColor(bgr_slika, cv2.COLOR_BGR2RGB)

global roberts_slika
roberts_slika = my_roberts(slika)
prewitt_slika = my_prewitt(slika)
sobel_slika = my_prewitt(slika)
canny_slika = canny(slika, 100, 200)

cv2.imwrite('Roberts.jpg', roberts_slika)
cv2.imwrite('Prewitt.jpg', prewitt_slika)
cv2.imwrite('Sobel.jpg', sobel_slika)
cv2.imwrite('Canny.jpg', canny_slika)

#1 2 3 4
#5 6 7 8
fig.add_subplot(rows, columns, 1)
plt.imshow(roberts_slika)
plt.axis('off')
plt.title("Roberts slika")

fig.add_subplot(rows, columns, 2)
plt.imshow(prewitt_slika)
plt.axis('off')
plt.title("Prewitt slika")

fig.add_subplot(rows, columns, 5)
plt.imshow(sobel_slika)
plt.axis('off')
plt.title("Sobel slika")

fig.add_subplot(rows, columns, 6)
plt.imshow(canny_slika)
plt.axis('off')
plt.title("Canny slika")

fig.add_subplot(rows, columns, (3, 8))
plt.imshow(slika)
plt.axis('off')
plt.title("Originalna slika")