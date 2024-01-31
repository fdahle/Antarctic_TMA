
id1 = "CA216632V0282"
id2 = "CA216632V0283"

import base.load_image_from_file as liff
img1 = liff.load_image_from_file(id1)
img2 = liff.load_image_from_file(id2)

import cv2 as cv2

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

print(len(matches))
