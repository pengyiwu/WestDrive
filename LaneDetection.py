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

cap = cv2.VideoCapture('test.mp4')

while (True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.imread('test.jpeg')

    height = np.size(frame, 0)
    width = np.size(frame, 1)

    #frame = cv2.blur(frame, (2, 2))

    edges = cv2.Canny(frame, 100, 200)

    rows, cols = edges.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.9]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.9]
    top_right = [cols * 0.6, rows * 0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cropped_edges = filter_region(edges, vertices)

    hough_lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=50,maxLineGap=50)

    if hough_lines is None:
        continue

    list_of_lines = list(hough_lines)

    totalM = 0
    m = 3

    longestLeftLine = [0,0,0,0]
    longestRightLine = [0,0,0,0]

    xCenter = width/2

    for i in list_of_lines:
        arr = i[0]
        lineThickness = 1
        x1 = arr[0]
        y1 = arr[1]
        x2 = arr[2]
        y2 = arr[3]

        #on the left
        if x1 < xCenter:
            if calculateDistance(x1, y1, x2, y2) > calculateDistance(longestLeftLine[0],longestLeftLine[1],longestLeftLine[2],longestLeftLine[3]):
                longestLeftLine = [x1, y1, x2, y2]
        else:
            if calculateDistance(x1, y1, x2, y2) > calculateDistance(longestRightLine[0],longestRightLine[1],longestRightLine[2],longestRightLine[3]):
                longestRightLine = [x1, y1, x2, y2]



    cv2.line(frame, (longestLeftLine[0],longestLeftLine[1]), (longestLeftLine[2],longestLeftLine[3]), (255, 0, 0), 10)
    cv2.line(frame, (longestRightLine[0],longestRightLine[1]), (longestRightLine[2],longestRightLine[3]), (255, 0, 0), 10)


    if (float(longestLeftLine[2]) - float(longestLeftLine[0])) != 0:
        totalM += (float(longestLeftLine[1]) - float(longestLeftLine[3])) / (float(longestLeftLine[2]) - float(longestLeftLine[0]))

    if (float(longestRightLine[2]) - float(longestRightLine[0])) != 0:
        totalM += (float(longestRightLine[1]) - float(longestRightLine[3])) / (float(longestRightLine[2]) - float(longestRightLine[0]))

    if calculateDistance(longestLeftLine[0],longestLeftLine[1],longestLeftLine[2],longestLeftLine[3]) == 0 or calculateDistance(longestRightLine[0],longestRightLine[1],longestRightLine[2],longestRightLine[3]) == 0:
        confidence = 0
        m = 15
    else:
        confidence = 10

    rise = 100
    run = rise * totalM / m

    angle = atan(run/rise)*57.2958

    #angle --> The degrees to turn in either direction
    #confidence --> confidence of making the turn at that angle, 0 to 10

    print(angle)
    print(confidence)

    cv2.line(frame, (width / 2, int(height * 0.75)), (int(width / 2 + run), int(height * 0.75 - rise)), (0, 255, 0), 5)

# Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()