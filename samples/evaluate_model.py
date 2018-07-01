
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/carla/"))  # To find local version
import carla
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# COCO_MODEL_PATH = "/Users/zhou/Desktop/Sem.Pro-RemoteFile/mask_rcnn_carla_0050.h5"

COCO_MODEL_PATH = "/Users/zhou/Desktop/Sem.Pro-RemoteFile/mask_rcnn_carla_zurich_0100_mrcnn13.h5"
COCO_MODEL_PATH = "/cluster/work/riner/users/zgxsin/mask_rcnn_archive/Mask_RCNN13/logs/carla_zurich20180627T0940/mask_rcnn_carla_zurich_0100.h5"

# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(carla.CarlaConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5

config = InferenceConfig()
config.display()



# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['static', 'dynamic']



from mrcnn import visualize
from PIL import Image
from os import listdir
from os.path import isfile, join, isdir


extend_save_directory = "/Users/zhou/Desktop/evaluate_dataset/"
extend_save_directory = "/cluster/work/riner/users/zgxsin/semester_project/evaluate_dataset/"
dir_temp = extend_save_directory +'RGB/'
dirname_list = []
image_counter_list = []
f1_score_list = []

dir_list = [f for f in listdir(dir_temp) if isdir(join(dir_temp, f))]
for j, dirname in enumerate(dir_list):

    image_dir = os.path.join( extend_save_directory, "RGB",dirname)
    mask_dir = os.path.join( extend_save_directory, "Mask",dirname)
    image_list = [f for f in listdir( image_dir ) if isfile( join( image_dir, f ) )]
    f1_rate = 0
    precision_rate = 0
    recall_rate = 0
    image_counter = 0
    dirname_list.append( dirname )
    for i, filename in enumerate(image_list):
        image_path = os.path.join( image_dir, filename )
        try:
            image = skimage.io.imread( image_path )
        except:
            print( image_path, " Can't read this image!" )
            continue
        results = model.detect([image], verbose=0)
        r = results[0]
        # tem ,ax= visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                     class_names, r['scores'], show_bbox=True)
        # ax.imshow(tem)

        # mask path "/Users/zhou/Desktop/evaluate_dataset/Mask/zurich/zurich_000000_000019_gtFine_labelTrainIds.png"
        # image path '/Users/zhou/Desktop/evaluate_dataset/RGB/zurich/zurich_000000_000019_leftImg8bit.png'

        mask_filename = '_'.join(filename.split("_")[:3])+"_gtFine_labelTrainIds.png"
    #     print(mask_filename)
        mask_path = os.path.join( mask_dir, mask_filename )
        try:
            mask_temp = skimage.io.imread( mask_path, as_grey=True )
        except:
            print("Read Mask error!")
            continue


        mask_temp = mask_temp > 0
        gt_mask = np.asarray( mask_temp, np.uint8 )


        masks = r['masks'].astype(np.uint8)
        pred_mask = (np.sum(masks, axis=2)>0).astype(np.int)

        # f1_score( y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight = None)
        rate_temp_f1 = f1_score(gt_mask.flatten(), pred_mask.flatten())
        # rate_temp_recall = recall_score(gt_mask.flatten(), pred_mask.flatten())
        # rate_temp_precision = precision_score( gt_mask.flatten(), pred_mask.flatten())

        # print("The f1 score of the {} image is {}".format(image_counter, rate_temp_))

        # print( "The recall of the {} image is {}".format( image_counter, rate_temp ) )

    #     print(rate_temp)
        f1_rate = f1_rate + rate_temp_f1
        # precision_rate = precision_rate + rate_temp_recall
        # recall_rate =recall_rate + rate_temp_recall
        image_counter = image_counter +1
    image_counter_list.append(image_counter)
    f1_result = f1_rate/image_counter
    f1_score_list.append(f1_result)
    # pre_result =  precision_rate/image_counter
    # recall_result = recall_rate /image_counter
    print("The f1 score for {} Dataset is {}".format(dirname, f1_result))



data = [dirname_list, image_counter_list, f1_score_list]
total_average = sum(f1_score_list)/len(f1_score_list)
with open( 'model.csv','w', newline='' ) as myfile:
        wr = csv.writer( myfile,  delimiter=',')
        wr.writerows(data)
        wr.writerow(total_average)
print('Evaluation OVER!')
