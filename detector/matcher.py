from pydarknet import Detector
from pydarknet import Image as Img
import numpy as np
import cv2
from detector.rootsift import RootSIFT

class Matcher:

    def __init__(self, images, config=None, weights=None, data=None):
        self.d = {}
        if config != None:
            self.net = Detector(bytes(config, encoding="utf-8"), bytes(weights, encoding="utf-8"), 0, bytes(data,encoding="utf-8"))

        self.imagesPath = images

        self.images = []
        for filename in self.imagesPath:
            img = cv2.imread(filename)
            img = cv2.resize(img, (300,400))
            self.images.append(np.array(img))

        self.sift = cv2.xfeatures2d.SIFT_create()
        self.rsift = RootSIFT()
        # kps = detector.detect(gray)

        for index, img in enumerate(self.images):
            keypoints = self.sift.detect(img)
            keypoints, descriptors = self.rsift.compute(img, keypoints)
            self.d[index] = (keypoints, descriptors)

    def predict(self, image):
        # keypoints2, descriptors2 = self.sift.detectAndCompute(scene_image, None)
        image = cv2.resize(image, (300, 400))

        keypoints2 = self.sift.detect(image)
        keypoints2, descriptors2 = self.rsift.compute(image, keypoints2)
        bf = cv2.BFMatcher()
        bestImageIndex = 0
        max_matches = 0

        for index in self.d:
            if isinstance(descriptors2, np.ndarray):
                if len(self.d[index][1])>0 and len(descriptors2>0):
                    matches = bf.knnMatch(self.d[index][1], descriptors2, k=2)
                    goodMatches = 0
                    # Apply ratio test
                    for match in matches:
                        if len(match)==2:
                            m = match[0]
                            n = match[1]
                            if m.distance < 0.60 * n.distance:
                                goodMatches +=1

                    if goodMatches > max_matches:
                        max_matches = goodMatches
                        bestImageIndex = index
        # best_image = cv2.resize(self.images[bestImageIndex], (300, 400))
        return self.images[bestImageIndex], bestImageIndex

    def extract_BB_with_YOLO(self, scene_image):
        img = cv2.imread(scene_image)
        img2 = Img(img)
        # r = net.classify(img2)
        results = self.net.detect(img2)

        return [x[2] for x in results]


def cropped_from_coordinates(img, cropped_coordinates):
    cropped_images = []
    for x, y, w, h in cropped_coordinates:
        # print(bounds)
        # x, y, w, h = bounds
        x1 = int(x - w / 2) if int(x - w / 2) > 0 else 0
        y1 = int(y - h / 2) if int(y - h / 2) > 0 else 0
        x2 = int(x + w / 2) if int(x + w / 2) < img.shape[1] else img.shape[1]
        y2 = int(y + h / 2) if int(y + h / 2) < img.shape[0] else img.shape[0]

        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        cropped = img[y1:y2, x1:x2]
        cropped_images.append(cropped)

    return cropped_images