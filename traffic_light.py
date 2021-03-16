import os
import cv2
import argparse
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX

font = cv2.FONT_HERSHEY_SIMPLEX

thresholds = {
    'RED': (100,222),
    'YELLOW':(50,float('inf')),
    'GREEN':(145,148)
}

def is_good_circle(mask,center,radius,threshold):
    R = int(radius)
    if center[0] > mask.shape[1] or center[1] > mask.shape[0]:
        return False
   
    # total = np.sum(mask[center[1]-R:center[1]+R,center[0]-R:center[0]+R])
    mean =  np.mean(mask[center[1]-R:center[1]+R,center[0]-R:center[0]+R])
    std =  np.std(mask[center[1]-R:center[1]+R,center[0]-R:center[0]+R])
    count = 4*radius*radius
    t = mean
    # print(std/t, t,R,mean)
    return (count>0 and threshold[1] > t > threshold[0], t)

def add_to_output(circles, mask, text, output):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for each in circles[0,:]:
            center = each[:-1]
            radius = each[-1]
            test, t = is_good_circle(mask, center,radius,thresholds[text])
            if test:
                print(text)
                cv2.circle(output, tuple(center), radius, (0, 255, 0), 2)
                cv2.putText(output,f'{text}',tuple(center), font, 0.5,(255,0,0),1,cv2.LINE_AA)

def recognize(input_file):
    MIN_RADIUS = 0
    MAX_RADIUS = 22
    P1 = 50
    P2 = 10
    img = cv2.imread(input_file)
    output = img[:]
    # convert to hsv color mode -> easier to pick colors
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
    lower_red1 = np.array([0,100,100])
    upper_red1= np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])

    #https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image
    lower_green = np.array([40,40,40])
    upper_green = np.array([70,255,255])

    # https://stackoverflow.com/questions/9179189/detect-yellow-color-in-opencv/19488733
    lower_yellow = np.array([20,150,150])
    upper_yellow = np.array([30,255,255])

    #Threshold masks
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1),cv2.inRange(hsv, lower_red2, upper_red2))

    #Detect circiles
    # maskr = cv2.medianBlur(maskr,5)
    # maskg = cv2.medianBlur(maskg,5)
    # masky = cv2.medianBlur(masky,5)
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                            param1=P1, param2=P2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 80,
                                param1=P1, param2=P2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 80,
                                param1=P1, param2=P2, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    add_to_output(r_circles, maskr, "RED", output)
    add_to_output(g_circles, maskg, "GREEN", output)
    add_to_output(y_circles, masky, "YELLOW", output)
    cv2.imwrite('output.jpg',output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic light detector')
    parser.add_argument('input_path', help='path of the input image')
    args = parser.parse_args()
    recognize(args.input_path)