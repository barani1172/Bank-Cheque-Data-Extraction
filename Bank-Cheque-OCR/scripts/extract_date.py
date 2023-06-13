from imports import *
from preprocess import pad_img
from vision import vision_api

mnist_model = load_model('C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\mnist_GC_v1.h5')

def ext_date(img, mask):

    date_path = 'C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\feilds\\Date\\'
    if os.path.exists(date_path) and os.path.isdir(date_path):
        shutil.rmtree(date_path)

    if not os.path.exists(date_path):
        os.mkdir(date_path)
    rows,cols = img.shape
    x = 0 # row
    y = cols - 700 # col
    w = 250
    h = 700

    date_img_bkp = img[x:x+w, y:y+h]
    date_img_mask = mask[x:x+w, y:y+h]
    padded_img, padded_img_bkp = pad_img(date_img_mask, date_img_bkp)
    if padded_img_bkp.ndim == 2:
        date_img_color = cv2.cvtColor(padded_img_bkp, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('./../init_date_mask.jpg', padded_img)
    new_mask = np.ones_like(padded_img) * 255
    dateContours, hier = cv2.findContours(padded_img_bkp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if dateContours is not None and len(dateContours) > 0:
        count = 0
        for contour in dateContours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 100:  
                cv2.rectangle(date_img_color, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(new_mask, (x, y), (x + w, y + h), (0, 0, 255), -1)

        finalDateContours = cv2.findContours(new_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        if finalDateContours is not None and len(finalDateContours) > 0:
            finalDateContours = contours.sort_contours(finalDateContours, method='right-to-left')[0]
            contour_counts = 0
            date = []
            for cont in finalDateContours:
                x, y, w, h = cv2.boundingRect(cont)
                if (7 < w < 100) and h > 10 and contour_counts < 8:
                    new_img = padded_img_bkp[y - 5:y+h+5, x-5:x+w+5]
                    date.append(new_img)
                    cv2.imwrite('C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\feilds\\Date\\date_img_'+str(count)+'.jpg', new_img)
                    count += 1
                    contour_counts += 1
            f_date = []
            digits = [file for file in os.listdir('C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\feilds\\Date\\')]
            digits = digits[::-1]
            for d in digits:
                img = cv2.imread('C:\\Users\\baran\\Downloads\\vgts2\\Bank-Cheque-OCR\\feilds\\Date\\'+d, 0)
                img = cv2.resize(img, (28, 28))
                img = cv2.threshold

