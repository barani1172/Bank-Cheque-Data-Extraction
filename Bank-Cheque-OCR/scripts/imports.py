import cv2
import shutil
import imutils
from imutils import contours
from PIL import Image
import numpy as np
import os
# import tesserocr as tr
import pytesseract as tr 
import re 
import os 
import io 
import vision 
from matplotlib import pyplot as plt 
import pandas as pd 
from skimage.segmentation import clear_border
from imutils import contours
from keras.models import load_model

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\scripts\\__pycache__\\vision-91b4ebf7783a.json'
tr.pytesseract.tesseract_cmd = r'D:\\Tesseract\\tesseract.exe'
charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "T", "U", "A", "D"]
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
mnist_model = load_model('C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\mnist_GC_v1.h5')

