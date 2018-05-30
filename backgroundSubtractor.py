# backgroundSubtractor.py
"""
backgroundSubtractor
1. MOG -- Gaussian Mixture-based Background/Foreground Segmentation Algorithm. “An improved adaptive background mixture model for real-time tracking with shadow detection” in 2001. by P. KadewTraKuPong and R. Bowden
2. MOG2 -- Gaussian Mixture-based Background/Foreground Segmentation Algorithm. “Improved adaptive Gausian mixture model for background subtraction” in 2004 and “Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction” in 2006. by Z.Zivkovic
3. GMG -- Algorithm combines statistical background image estimation and per-pixel Bayesian segmentation. “Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation” in 2012. by Andrew B. Godbehere et al.

@author: Yen
"""

import cv2


cap = cv2.VideoCapture('sf_Lombard_street_3.mp4')
fgbg_MOG = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg_MOG2 = cv2.createBackgroundSubtractorMOG2()
fgbg_GMG = cv2.bgsegm.createBackgroundSubtractorGMG() # SLOW


while True:
    ret, frame = cap.read()
    
    if ret == True:
        cv2.imshow('origin' ,frame)
        
        fgmask_MOG  = fgbg_MOG.apply(frame)
        fgmask_MOG2 = fgbg_MOG2.apply(frame)
        fgmask_GMG  = fgbg_GMG.apply(frame)
        cv2.imshow('fgbg_MOG' ,fgmask_MOG)
        cv2.imshow('fgbg_MOG2',fgmask_MOG2)
        cv2.imshow('fgbg_GMG' ,fgmask_GMG)
        
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
    else:
        break


cap.release()
cv2.destroyAllWindows()
    
    
    
