from imports import *
charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "T", "U", "A", "D"]
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
    charIter = charCnts.__iter__()
    rois = []
    locs = []
    while True:
        try:
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None
            if cW >= minW and cH >= minH:
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            else:
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf, -np.inf)
                for p in parts:
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))
        except StopIteration:
            break
    return rois, locs
def find_ref_micr_data():
    image = cv2.imread('C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\reference_micr.png')
    ref, refCnts = find_ref_micr_contours(image)
    refROIs = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)[0]
    chars = {}
    for (name, roi) in zip(charNames, refROIs):
        roi = cv2.resize(roi, (30, 30))
        chars[name] = roi
    return chars


def find_ref_micr_contours(image):
    ref = imutils.resize(image, width=400)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    _, ref_thresh = cv2.threshold(ref_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours = cv2.findContours(ref_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        raise Exception("No contours found")

    refCnts = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return ref, refCnts
def extract_blackhat(image): 
    (h, w,) = image.shape[:2]
    delta = int(h - (h * 0.15))
    bottom = image[delta:h, 0:w]
    gray = np.copy(bottom)
    cv2.imwrite('bottom.jpg', gray)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    return blackhat, gray, delta


def find_group_contours(image):
    blackhat = extract_blackhat(image=image)[0]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    groupCnts = imutils.grab_contours(groupCnts)
    return groupCnts


def group_locations(image):
    groupCnts = find_group_contours(image=image)
    groupLocs = []
    for (i, c) in enumerate(groupCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 50 and h > 15:
            groupLocs.append((x, y, w, h))
    groupLocs = sorted(groupLocs, key=lambda x: x[0])
    return groupLocs


def extract_micr(image):
    blackhat, gray, delta = extract_blackhat(image=image)
    groupLocs = group_locations(image=image)
    chars = find_ref_micr_data()
    output = []
    for (gX, gY, gW, gH) in groupLocs:
        groupOutput = []
        group = gray[gY - 2:gY + gH + 2, gX - 2:gX + gW + 2]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        charCnts = imutils.grab_contours(charCnts)
        charCnts = contours.sort_contours(charCnts, method="left-to-right")[0]
        (rois, locs) = extract_digits_and_symbols(group, charCnts)
        for roi in rois:
            scores = []
            roi = cv2.resize(roi, (36, 36))
            for charName in charNames:
                result = cv2.matchTemplate(roi, chars[charName], cv2.TM_CCORR)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            groupOutput.append(charNames[np.argmax(scores)])
        cv2.rectangle(image, (gX - 10, gY + delta - 10), (gX + gW + 10, gY + gY + delta), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput),
                    (gX - 10, gY + delta - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
        output.append("".join(groupOutput))
    output = " ".join(output)
    return output, image
