import CourtDetect
import numpy as np
import cv2

# homography=CourtDetect.calculatehomographycandidate.best_homography
land_Video_Xy=[
[[639,700]],
[[231,586]],
[[825,304]],
[[353,283]],
[[765,269]],
[[692,518]],
[[105,675]],

[[60,549]],
[[875,289]],
[[739,279]]
]
realBd=[[[ 0. ,   0.  ]],
  [[ 0. ,  76]],
  [[ 0. ,   468]],
  [[ 0. ,   868]],
  [[ 0.  , 1264]],
  [[ 0.  , 134 ]],
  [[50,  0.  ]],
  [[ 50 ,  76]],
  [[ 50 ,  468]],
  [[ 50 ,  868]],
  [[ 50.  ,1264]],
  [[ 50 , 134 ]],
  [[ 305 , 0.  ]],
  [[ 305 , 76]],
  [ [305 , 468]],
  [[ 305 , 868]],
  [[ 305 ,1264]],
  [[ 305 ,1340 ]],
  [[ 560 ,  0.  ]],
  [[ 560  , 76]],
  [[ 560  , 468]],
  [[ 560  , 868]],
  [[ 560  ,1264]],
  [ [560 , 1340 ]],
  [[ 610  , 0.  ]],
  [ [610  , 76]],
  [ [610   ,468]],
  [[ 610 ,  868]],
  [[ 610  ,1264]],
  [[ 610 , 1340 ]]]

best_homography=np.array([[-117,  21.16771, 997.06421],
 [  -0.0059,   -4.8127,  643.54987],
 [  -0.,       0.03318,   1.     ]])
H = np.linalg.inv(best_homography)

real_xy=[]
for i in range(0,len(realBd)):

    a = np.array(realBd[i], dtype='float32')
    a = np.array([a])

    # finally, get the mapping
    pointsOut = cv2.perspectiveTransform(a, H)
    pointsOut = pointsOut.astype(int)
    print(pointsOut)
"""
best_homography = [[-116.75904   21.16771  997.06421]
 [  -0.0059    -4.8127   643.54987]
 [  -0.         0.03318    1.     ]]
 
 best homography = [[-126.82451   40.56482  818.48962]
 [  -8.97896  -20.93396  714.40594]
 [  -0.00018    0.03726    1.     ]]

"""
