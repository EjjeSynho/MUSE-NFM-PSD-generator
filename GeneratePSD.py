from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colors

import os
from tqdm import tqdm

import cupy as cp
from cupyx.scipy.fft import get_fft_plan
import cupyx.scipy.fft as gfft

class GeneratePSD:
    def __init__(self):
        self.use_GPU = True
        
