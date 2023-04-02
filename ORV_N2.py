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