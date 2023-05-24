# Program to extract images from paragraph or line and save those under extracted folder.

import cv2
import os

def captch_ex(file_name,saveFolder ):

    img = cv2.imread(file_name)
    alpha = 1.5 # Contrast control
    beta = 10 # Brightness control

    # call convertScaleAbs function
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6,
                                                         6))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=6)  # dilate , more the iteration more the dilation


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

        cropped = img_final[y+1 :y +  h +1, x +1: x + w+1]

        s = saveFolder + '/crop_' + str(index) + '.jpg'
        print(s)
        # cv2.imshow(s,cropped)
        cv2.waitKey()
        cv2.imwrite(s, cropped)
        index = index + 1

    # write original image with added contours to disk


file_name = r'D:/Useful/Python/Projects/BEProject/htr/data/page.jpg'
saveFolder = r'D:/Useful/Python/Projects/BEProject/htr/extractedPage'

captch_ex(file_name, saveFolder)

# directory = 'the/directory/you/want/to/use'

# for filename in os.listdir(savefilename):
#     if filename.endswith(".jpg"):
#         main()
#     else:
#         continue