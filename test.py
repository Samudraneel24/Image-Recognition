import numpy as np
import cv2
import sys
import glob


def sift_sim(path_a, path_b):
    '''
    Use SIFT features to measure image similarity
    @args:
      {str} path_a: the path to an image file
      {str} path_b: the path to an image file
    @returns:
      TODO
    '''
    # initialize the sift feature detector
    orb = cv2.ORB_create()

    # get the images
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    # find the keypoints and descriptors with SIFT
    kp_a, desc_a = orb.detectAndCompute(img_a, None)
    kp_b, desc_b = orb.detectAndCompute(img_b, None)

    # initialize the bruteforce matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match.distance is a float between {0:100} - lower means more similar
    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 70]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


if __name__ == '__main__':

	slide_arr = []
	ppt_arr = []
	for filename in glob.iglob('./Dataset/*/ppt.jpg', recursive=True):
		img = cv2.imread(filename)
		ppt_arr.append((filename, img))
		print(ppt_arr[1])
	
	# for filename in glob.iglob('./Dataset/*/*.jpg', recursive=True):
	# 	img = cv2.imread(filename)
	# 	# print("l")
	# 	if [filename, img] not in ppt_arr:
	# 		slide_arr.append([filename, img])

	# print(slide_arr)
    # img_a = sys.argv[1]
    # img_b = sys.argv[2]
    # sift_sim = sift_sim(img_a, img_b)
    # print(sift_sim)
