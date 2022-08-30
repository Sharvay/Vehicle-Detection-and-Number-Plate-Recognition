import re
import os
import cv2
import time
import numpy as np
import tensorflow as tf
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow.keras.preprocessing.image import img_to_array,load_img

from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


def load_model():

    PATH_TO_SAVED_MODEL = "exported-models/my_ssd_resnet50_v1_fpn_640x6400/saved_model"
    print('Loading Model...')
    start_time = time.time()
    loaded_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    print('Done! Took {} seconds'.format(time.time() - start_time))
    
    return loaded_model

def show_inference(model,test_image_rgb):
    PATH_TO_LABELS = "annotations/label_map.pbtxt"
    # Loading the pbtxt
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)

    # test_image_o = cv2.imread(img_path)
    # test_image_rgb = cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB)
    test_image_array_ex = np.expand_dims(test_image_rgb, axis=0)

    test_image_tensor = tf.convert_to_tensor(test_image_array_ex)
    test_image_tensor = tf.cast(test_image_tensor, tf.uint8) #changinig the dtype according to serving_default.

    # Predictions
    prediction = model(test_image_tensor)

    num_bboxes = int(prediction.pop('num_detections'))
    prediction = {key: value[0, :num_bboxes].numpy() for key, value in prediction.items()}
    prediction['num_detections'] = num_bboxes

    # detection_classes should be ints.
    prediction['detection_classes'] = prediction['detection_classes'].astype(np.int64)

    # image_with_detections = test_image_rgb.copy()
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #       image_with_detections,
    #       prediction['detection_boxes'],
    #       prediction['detection_classes'],
    #       prediction['detection_scores'],
    #       category_index,
    #       use_normalized_coordinates=True,
    #       max_boxes_to_draw=5,
    #       min_score_thresh=0.4,
    #       agnostic_mode=False)

    # Drwaing Bbox and OCR
    output_image = test_image_rgb.copy()
    threshold = 0.55
    valid_predictions = sum(prediction['detection_scores']>=threshold)
    txt = list()
    scr = list() 
    for i in range(valid_predictions):
        ymin, xmin, ymax, xmax = prediction['detection_boxes'][i]
        score = prediction['detection_scores'][i]
        im_height, im_width = test_image_rgb.shape[:2]
        (x1, x2, y1, y2) = (int(xmin * im_width), int(xmax * im_width),int(ymin * im_height), int(ymax * im_height))
        cv2.rectangle(output_image, (x1,y1), (x2,y2),(0, 255, 0), 2)
        crop_img = test_image_rgb[y1:y2, x1:x2]
        # To grayscale
        # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # Binarization
        # crop_img = cv2.adaptiveThreshold(crop_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        text = str(pytesseract.image_to_string(crop_img,config='--psm 10'))
        text = re.sub(r"^[A-Za-z0-9 ]*$",'',text).strip()
        # cv2.putText(output_image,text,(x1-10,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        # cv2.putText(output_image,str(round(score*100,1)),(x1+10,y2+10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        txt.append(text)
        scr.append(str(round(score*100,1)))

    return output_image,txt,scr
