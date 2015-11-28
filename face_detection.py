import cv2
import numpy as np
import load_images as li
import HOG
from sklearn import svm
import image_process as ip
import time
from nms import non_max_suppression_slow as nms
from sklearn.ensemble import AdaBoostClassifier


def extract_hog(images):
        res_im = HOG.resizeImages(images)
        hog_descriptors = map(HOG.HOG, res_im)
        return hog_descriptors

def gather_data(px, py, nx, ny):
        x = np.append(px, nx, axis=0)
        y = np.append(py, ny)
        return (x, y)



def train_svm(feature, label, n_train_set):
        clf = AdaBoostClassifier(svm.SVC(probability=True,kernel='linear'),n_estimators=50)
        clf.fit(feature, label)

        #false_positive = []

        #for image in n_train_set:
        #        for ri in ip.pyramid(image, pyramid_scale = 1.25):
        #                for (x, y, window) in ip.sliding_window(ri, stepSize=16, windowSize=(64,64)):
        #                        if window.shape[0] != minWinSize[1] or window.shape[1] != minWinSize[0]:
        #                                continue
        #                        test_x = np.array(HOG.HOG(window)).reshape(1,-1)
        #                        predict_y = clf.predict(test_x)

        #                        if predict_y == 1:
        #                                false_positive.append(test_x[0])

        #false_positive_label = np.zeros(len(false_positive))

        #new_feature = np.append(feature, np.array(false_positive), axis=0)
        #new_label = np.append(label, false_positive_label)

        #new_clf = svm.SVC(probability=True)
        #new_clf.fit(new_feature, new_label)
        return clf

if __name__ == "__main__":

        minWinSize = [64, 64]

        p_images = li.load("orl_faces")
        n_images = li.load("background")
        #p_test_images = li.load("positive_test_images")
        p_test_images = li.load("test")
        n_test_images = li.load("negative_test_images")

        test_images = np.append(p_test_images, n_test_images, axis=0)

        px = np.array(extract_hog(p_images))
        nx = np.array(extract_hog(n_images))
        py = np.ones(len(p_images))
        ny = np.zeros(len(n_images))

        #test_px = np.array(extract_hog(p_test_images))
        #test_nx = np.array(extract_hog(n_test_images))
        test_py = np.ones(len(p_test_images))
        test_ny = np.zeros(len(n_test_images))


        x, y = gather_data(px, py, nx, ny)

        model = train_svm(x, y, n_images)

        predictions = []

        count = 0.

        for idx, image in enumerate(n_test_images):

                scale = 0
                detections = []
                single_pred = 0
                for ri in ip.pyramid(image, pyramid_scale=1.25):
                        current_d = []
                        for (x, y, window) in ip.sliding_window(ri, stepSize=16, windowSize=(64,64)):
                                if window.shape[0] != minWinSize[1] or window.shape[1] != minWinSize[0]:
                                        continue
                                test_x = np.array(HOG.HOG(window)).reshape(1, -1)

                                predict_y = model.predict(test_x)

                                if predict_y == 1:
                     #                   print "Detection: location -> {} {}".format(x, y)
                     #                   print "Scale = {} **** Confidence Score {}\n".format(scale, model.decision_function(test_x))
                                        detections.append((x, y, x + 64, y + 64))
                                        current_d.append(detections[-1])
                                        single_pred = 1



                      #          clone = ri.copy()
                      #          for (x1, y1, x2, y2) in current_d:
                      #                  cv2.rectangle(clone, (x1, y1), (x2, y2), (255,0,0), thickness=2)
                      #          #cv2.rectangle(clone, (x, y), (x + minWinSize[1], y + minWinSize[0]),(0,255,0), thickness=2)
                      #  print "finish a scale"
                      #  cv2.imshow("Sliding Window in Progress", clone)
                      #  cv2.waitKey(0)

                      #  nms_clone = ri.copy()
                      #  nms_detections = nms(np.array(current_d), 0.3)
                      #  for startX, startY, endX, endY in nms_detections:
                      #          cv2.rectangle(nms_clone, (startX, startY), (endX, endY), (0,0,255),thickness=2)
                      #  cv2.imshow("apply non maximum suppression", nms_clone)
                      #  cv2.waitKey(0)
                        scale += 1
                count += 1.
                print "progression: {}%".format((count / len(n_test_images))*100)
                predictions.append(single_pred)

        print predictions, np.array(predictions).shape, float(predictions.count(0))/len(predictions)


                                #print model.decision_function(test_x)

                                # if predict_y:
                                #         clone = ri.copy()
                                #         cv2.rectangle(clone,(x, y), (x + 64, y + 64), (100,100,100) ,2)
                                #         cv2.imshow("Window", clone)
                                #         cv2.waitKey(0)
                                #         time.sleep(0.1)


                                #predict testdata







