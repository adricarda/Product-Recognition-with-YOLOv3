import cv2
import glob
import numpy as np
import argparse
import sys

sys.path.insert(0,'..')
from detector.matcher import Matcher, cropped_from_coordinates

def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('-m', '--models', help='models directory', required=True)
    parser.add_argument('-s', '--scenes', help='scenes directory', required=True)

    results = parser.parse_args(args)
    return results.models, results.scenes

def main():
    models_dir, scenes_dir = check_arg(sys.argv[1:])
    models = glob.glob(models_dir+'/*')
    scenes = glob.glob(scenes_dir+'/*')
    M = Matcher(models, config='YOLO/yolov3.cfg', weights='YOLO/yolov3.weights', data='YOLO/obj.data')
    for i, scene_image in enumerate(scenes):
        print('reading new image {}'.format(scene_image))
        img = cv2.imread(scene_image)

        d = {}
        for i in range(len(models)):
            d[i] = []

        cropped_coordinates = M.extract_BB_with_YOLO(scene_image)
        cropped_images = cropped_from_coordinates(img, cropped_coordinates)
        for k, cropped in enumerate(cropped_images):
            prediction, index = M.predict(cropped)
            data = list(map(int, cropped_coordinates[k]))
            d[index].append(data)

            x, y, w, h = data
            x1 = int(x - w / 2) if int(x - w / 2) > 0 else 0
            y1 = int(y - h / 2) if int(y - h / 2) > 0 else 0
            x2 = int(x + w / 2) if int(x + w / 2) < img.shape[1] else img.shape[1]
            y2 = int(y + h / 2) if int(y + h / 2) < img.shape[0] else img.shape[0]

            copy = cv2.rectangle(img.copy(), (x1,y1), (x2,y2), (0, 255, 0), 4, cv2.LINE_AA)
            cv2.namedWindow("scene")
            cv2.moveWindow("scene", 100, 200)
            cv2.imshow('scene', cv2.resize(copy, (0, 0), fx=0.6, fy=0.6))

            # prediction = M.match_reference(cropped)
            h = cropped.shape[0]
            w = cropped.shape[1]
            prediction = cv2.resize(prediction, (w, h))
            concatenated = np.concatenate((cropped, prediction), axis=1)
            cv2.namedWindow("Cropped and prediction")
            cv2.moveWindow("Cropped and prediction", 1200, 200)
            cv2.imshow('Cropped and prediction', concatenated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        for i in d:
            print('Product {} - {} instances found:'.format(i, len(d[i])))
            for instance_number, instance_data in enumerate(d[i]):
                print('         Instance {} [position: ({}, {}) width: {}, height: {} ]'.format(instance_number,
                                                                                                instance_data[0],
                                                                                                instance_data[1],
                                                                                                instance_data[2],
                                                                                                instance_data[3]))

        print('---------------------')

if __name__ == "__main__":
    main()