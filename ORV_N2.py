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

