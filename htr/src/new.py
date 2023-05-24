import cv2
import os

pparabig = r'D:/Useful/Python/Projects/BEProject/htr/data/parabig.jpg'

def captch_ex(file_name):
    img = cv2.imread(file_name)
    cv2.imshow('img',img)
    cv2.waitKey()
    alpha = 1.5 # Contrast control
    beta = 10 # Brightness control

    # call convertScaleAbs function
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if not cv2.imwrite(r'D:\Useful\Python\Projects\BEProject\htr\src\1.jpg', img):
        raise Exception("Couldnt write image")
    cv2.imshow('img',img)
    cv2.waitKey()

    img_final = cv2.imread(file_name)
    cv2.imshow('img_final',img_final)
    if not cv2.imwrite(r'D:\Useful\Python\Projects\BEProject\htr\src\2.jpg', img_final):
        raise Exception("Couldnt write image")

    cv2.waitKey()
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img2gray',img2gray)
    if not cv2.imwrite(r'D:\Useful\Python\Projects\BEProject\htr\src\3.jpg', img2gray):
        raise Exception("Couldnt write image")

    cv2.waitKey()
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imshow('mask',mask)
    cv2.waitKey()
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    cv2.imshow('image_final',image_final)
    if not cv2.imwrite(r'D:\Useful\Python\Projects\BEProject\htr\src\4.jpg', image_final):
        raise Exception("Couldnt write image")

    cv2.waitKey()
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    cv2.imshow('new_img',new_img)
    if not cv2.imwrite(r'D:\Useful\Python\Projects\BEProject\htr\src\5.jpg', new_img):
        raise Exception("Couldnt write image")

    cv2.waitKey()
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    cv2.imshow('kernel',kernel)
    if not cv2.imwrite(r'D:\Useful\Python\Projects\BEProject\htr\src\6.jpg', kernel):
        raise Exception("Couldnt write image")

    cv2.waitKey()
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation
    cv2.imshow('dilated',dilated)
    if not cv2.imwrite(r'D:\Useful\Python\Projects\BEProject\htr\src\7.jpg', dilated):
        raise Exception("Couldnt write image")

    cv2.waitKey()

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours
    index = 0

    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)


        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg'
        if not cv2.imwrite(r'D:\Useful\Python\Projects\BEProject\htr\src' + str(index) + r'.jpg', cropped):
            raise Exception("Couldnt write image")
        index = index + 1

    # write original image with added contours to disk
    cv2.imshow('captcha_result',img)
    cv2.waitKey()

captch_ex(pparabig)

