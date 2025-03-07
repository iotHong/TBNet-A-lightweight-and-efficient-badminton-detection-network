import cv2
import datetime
import numpy as np
from collections import defaultdict
# from centroidtracker import CentroidTracker
import pandas as pd
import imutils

land_Video_Xy=[
[[639,695]],
# [[231,586]],
# [[825,304]],
# [[353,383]],
# [[765,269]],
# [[692,518]],
# [[105,675]],

# [[60,549]],
# [[875,289]],
# [[739,279]]
]

pts_src = np.array([[392, 252], [379, 268], [300, 361], [200, 478], [73, 626], [44, 660],
                    [435, 255], [422, 271], [348, 364], [254, 482], [135, 630], [108, 664],
                    [650, 271], [642, 287], [592, 381], [529, 501], [450, 652], [431, 687],
                    [866, 286], [862, 302], [836, 399], [804, 521], [764, 674], [755, 709],
                    [908, 289], [905, 305], [884, 402], [858, 524], [825, 679], [818, 714]])
# Take points from the frame as reference and give the same point coordinates on the picture for a transformation
pts_dst = np.array([[0,0], [0,76], [0,468], [0,868], [0,1264],[ 0,1340],
                    [50,0], [50,76], [50,468],[ 50 ,868],[ 50,1264],[50, 1340],
                    [305,0], [305,76], [305,468],[305,868],[305,1264],[305, 1340],
                    [560,0], [560,76], [560,468],[ 560 ,868],[ 560,1264],[560, 1340],
                    [610,0], [610,76], [610,468],[ 610 ,868],[ 610,1264],[610, 1340]])
add_pts_dst=pts_dst*(1/1.65) + [42,62]

# calculate matrix H
h, status = cv2.findHomography(pts_src, pts_dst)
h1, status1= cv2.findHomography(pts_src, add_pts_dst)
real_xy=[]
# for i in range(0,len(land_Video_Xy)):
#
#     a = np.array(land_Video_Xy[i], dtype='float32')
#     a = np.array([a])
#
#     # finally, get the mapping
#     pointsOut = cv2.perspectiveTransform(a, h)
#     pointsOut = pointsOut.astype(int)
#     print(pointsOut)
print("++++++++++++++++++++++++++++++++++++++++")
for i in range(0,len(land_Video_Xy)):

    a = np.array(land_Video_Xy[i], dtype='float32')
    a = np.array([a])

    # finally, get the mapping
    pointsOut = cv2.perspectiveTransform(a, h1)
    pointsOut = pointsOut.astype(int).tolist()
    # print(type(pointsOut))
    # print(pointsOut)
    print("in video:",a[0][0])
    print("real:",pointsOut[0][0])
    outXY=tuple(pointsOut[0][0])
    real_xy.append(outXY)

# print(h)
# print(add_pts_dst)
print(real_xy)

filename = './badminton_court.jpg'
img = cv2.imread(filename)
img_output = img.copy()
close_LF=0
close_LM=0
close_LB=0
close_RF=0
close_RM=0
close_RB=0
close_hit_out=0

far_LF=0
far_LM=0
far_LB=0
far_RF=0
far_RM=0
far_RB=0
far_hit_out=0

print("+++++++++++++++++++++++++++++++++++++++++++")
for project_point in real_xy:
    # new_xy=tuple(map(tuple(int,project_point)))
    #out>>red______in>>blue
    if(project_point[1]>65 and project_point[1]<870 and project_point[0]>75 and project_point[0]<385):
        if(project_point[1]>465 and project_point[1]<582 and project_point[0]<225):
            close_LF = close_LF +1
        if (project_point[1] > 465 and project_point[1] < 582 and project_point[0] > 225):
            close_RF = close_RF + 1
        if (project_point[1] > 582 and project_point[1] < 720 and project_point[0] < 225):
            close_LM = close_LM + 1
        if (project_point[1] > 582 and project_point[1] < 720 and project_point[0] > 225):
            close_RM = close_RM + 1
        if (project_point[1] > 720 and project_point[1] < 870 and project_point[0] < 225):
            close_LB = close_LB + 1
        if (project_point[1] > 720 and project_point[1] < 870 and project_point[0] > 225):
            close_RB = close_RB + 1

        if (project_point[1] > 65 and project_point[1] < 210 and project_point[0] < 225):
            far_RB = far_RB + 1
        if (project_point[1] > 65 and project_point[1] < 210 and project_point[0] > 225):
            far_LB = close_LB + 1
        if (project_point[1] > 210 and project_point[1] < 348 and project_point[0] < 225):
            far_RM = far_RM + 1
        if (project_point[1] > 210 and project_point[1] < 348 and project_point[0] > 225):
            far_LM = close_LM + 1
        if (project_point[1] > 348 and project_point[1] < 465 and project_point[0] < 225):
            far_RF = far_RF + 1
        if (project_point[1] > 348 and project_point[1] < 465 and project_point[0] > 225):
            far_LF = close_LF + 1
        cv2.circle(img_output, project_point, 3, (255, 0,0 ), -1)
        # cv2.circle(img_output, project_point, 6, (255, 0, 0), -1)


    else:
        if(project_point[1] < 465):
            close_hit_out =close_hit_out+1
        else:
            far_hit_out = far_hit_out+1

        cv2.circle(img_output, project_point, 3, (0, 0, 255), -1)
    print(tuple(map(int, project_point)))

print("+++++++++++++++++++++++++++++++++++++++++++")


cv2.imwrite('land_result.jpg', img_output)

print("close player——————————————————————")
print('close_LF:{},close_LM:{},close_LB:{},close_RF:{},close_RM:{},close_RB:{}'.format(close_LF,close_LM,close_LB,close_RF,close_RM,close_RB))
print("close_hit_out:",close_hit_out)
print("far player——————————————————————")
print('far_LF:{},far_LM:{},far_LB:{},far_RF:{},far_RM:{},far_RB:{}'.format(far_LF,far_LM,far_LB,far_RF,far_RM,far_RB))
print("far_hit_out:",far_hit_out)