import os
import shutil
import cv2
import numpy as np
import imutils

from imports import *
from preprocess import pad_img
from vision import vision_api

def ext_amount(image, template):
    amount_path = 'C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\feilds\\Amount\\'
    if os.path.exists(amount_path) and os.path.isdir(amount_path):
        shutil.rmtree(amount_path)

    img_bkp = np.copy(image)
    if img_bkp.ndim == 2:
        img_bkp = cv2.cvtColor(img_bkp, cv2.COLOR_GRAY2RGB)

    if not os.path.exists(amount_path):
        os.mkdir(amount_path)

    # Convert template to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the grayscale template
    _, template_thresholded = cv2.threshold(template_gray, 0, 255, cv2.THRESH_OTSU)

    template = cv2.Canny(template_thresholded, 50, 200)
    (tH, tW) = template.shape[:2]

    found = None
    count = 0
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(image, width=int(image.shape[1] * scale))
        r = image.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                      (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        count += 1
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    cv2.rectangle(img_bkp, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imwrite('./../final_templ.jpg', img_bkp)

    amt_x1 = endX + 10
    amt_y1 = startY - 10
    amt_x2 = endX + 30
    amt_y2 = endY + 10
    h = abs(amt_y2 - amt_y1)
    y = amt_y1
    x = amt_x2
    w = abs(x - image.shape[1])
    amount = image[y:y+h, x:x+w]
    padded_img, padded_img_bkp = pad_img(amount, amount)
    padded_img = cv2.copyMakeBorder(padded_img, top=5, bottom=5, left=5, right=5, borderType=cv2.BORDER_CONSTANT, value=[255, 255,255])
    padded_img_bkp = cv2.copyMakeBorder(padded_img_bkp, top=5, bottom=5, left=5, right=5, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    cv2.imwrite('C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\feilds\\Amount\\padded_amount.jpg', padded_img)
    amount = vision_api('C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\scripts\\__pycache__\\vision-91b4ebf7783a.json')
    amount = "".join(amount)
    return amount