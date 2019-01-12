import cv2
import numpy as np
import glob
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import sys

def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('-m', '--models', help='models directory', required=True)
    parser.add_argument('-s', '--scenes', help='scenes directory', required=True)

    results = parser.parse_args(args)
    return results.models, results.scenes

def load_images(dir):
    dir += '/*'
    models = glob.glob(dir)
    images = []
    for filename in models:
        images.append(np.array(Image.open(filename)))
    return images

def findMatchesAndHomographies(model_number, scene_image, model_image):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(model_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(scene_image, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.80 * n.distance:
            good.append(m)

    # filter images with at least 250 good matches
    if len(good)>250:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = model_image.shape[:2]
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.imshow(str(model_number), cv2.resize(model_image, (300, 400)))
        print('         - model {} found with {} good matches'.format(model_number, len(good)))
        # draw found regions
        cv2.polylines(scene_image, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)

    return scene_image


def main():
    models_dir, scenes_dir = check_arg(sys.argv[1:])
    models = load_images(models_dir)
    scenes = load_images(scenes_dir)

    for i, scene_image in enumerate(scenes):
        print('--- scene image {}---'.format(i))
        for j, model_image in enumerate(models):
            scene_image = findMatchesAndHomographies(j, scene_image, model_image)

        cv2.imshow('scene', scene_image)
        plt.imsave('scene'+ str(i)+ '.jpg', scene_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()