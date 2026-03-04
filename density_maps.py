from PIL import Image, ImageSequence
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from collections import defaultdict

