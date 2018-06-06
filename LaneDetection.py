import numpy as np
import cv2
from math import atan

def calculateDistance(x1, y1, x2, y2):
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])  # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)

def midpoint(p1, p2):
    x = (p1[0]+p2[0])/2
    y = (p1[1]+p2[1])/2
    p = [x, y]
    return p

cap = cv2.VideoCapture('test.mp4')

while (True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.imread('test.jpeg')

    height = np.size(frame, 0)
    width = np.size(frame, 1)

    edges = cv2.Canny(frame, 100, 200)

    rows, cols = edges.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.9]
    top_left = [cols * 0.25, rows * 0.55]
    bottom_right = [cols * 0.9, rows * 0.9]
    top_right = [cols * 0.75, rows * 0.55]

    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cropped_edges = filter_region(edges, vertices)

    #INSERT CODE HERE
    # Find all lines in the image
    hough_lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=50, maxLineGap=50)

    if hough_lines is not None:
        list_of_lines = list(hough_lines)

    totalM = 0
    m = 0

    xCenter = width / 2
    yCenter = height*0.75

    leftLane = [-1,-1,-1,-1]
    rightLane = [-1,-1,-1,-1]

    for i in list_of_lines:
        arr = i[0]
        x1 = arr[0]
        y1 = arr[1]
        x2 = arr[2]
        y2 = arr[3]
        #Check if the line you are looking at has a slope near 0
        isNotFlatStraight = True
        if abs(float(x2) - float(x1)) < 50:
            isNotFlatStraight = False
            #if lane line is on the left side
        if (x1 < xCenter and x2 < xCenter) and isNotFlatStraight:
            if calculateDistance(leftLane[0], leftLane[1], leftLane[2], leftLane[3]) < calculateDistance(x1, y1, x2, y2):
                leftLane = [x1, y1, x2, y2]
            # if lane line is on the right side
        elif (x1 > xCenter and x2 > xCenter) and isNotFlatStraight:
            if calculateDistance(rightLane[0], rightLane[1], rightLane[2], rightLane[3]) < calculateDistance(x1, y1, x2, y2):
                rightLane = [x1, y1, x2, y2]

    cv2.line(frame, (rightLane[0], rightLane[1]), (rightLane[2], rightLane[3]), (255, 0, 0), 5)
    cv2.line(frame, (leftLane[0], leftLane[1]), (leftLane[2], leftLane[3]), (255, 0, 0), 5)

    angle = 0

    #check if right lane is valid
    if rightLane[0] != -1 and (float(rightLane[2]) - float(rightLane[0])) != 0:
        #right lane is valid
        totalM += (float(rightLane[1]) - float(rightLane[3])) / (float(rightLane[2]) - float(rightLane[0]))
        m+=1
    if leftLane[0] != -1 and (float(leftLane[2]) - float(leftLane[0])) != 0:
        #right lane is valid
        totalM += (float(leftLane[1]) - float(leftLane[3])) / (float(leftLane[2]) - float(leftLane[0]))
        m+=1

    run = 100
    rise = 100

    if m != 0:
        rise = 100
        run = rise * totalM / m
        angle = atan(run / rise) * 57.2958
    else:
        angle = 0

    # angle --> The degrees to turn in either direction
    # confidence --> confidence of making the turn at that angle, 0 to 10


    print(angle)

    cv2.line(frame, (width / 2, int(height * 0.75)), (int(width / 2 + run), int(height * 0.75 - rise)), (0, 255, 0), 5)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()