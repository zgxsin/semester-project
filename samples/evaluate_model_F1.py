import os
import sys
import numpy as np
import skimage.io
import csv
from os import listdir
from os.path import isfile, join, isdir

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
sys.path.append(os.path.join(ROOT_DIR, "samples/carla/"))  # To find local version
import carla
from sklearn.metrics import f1_score

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

CARLA_MODEL_PATH = "/Users/zhou/Desktop/Sem.Pro-RemoteFile/mask_rcnn_carla_zurich_0100_mrcnn13.h5"
CARLA_MODEL_PATH = "/cluster/work/riner/users/zgxsin/mask_rcnn_archive/Mask_RCNN13/logs/carla_zurich20180627T0940/mask_rcnn_carla_zurich_0100.h5"


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
# Load weights trained on carla_zurich
model.load_weights(CARLA_MODEL_PATH, by_name=True)


class_names = ['static', 'dynamic']



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
        f1_rate = f1_rate + rate_temp_f1

        image_counter = image_counter +1
    image_counter_list.append(image_counter)
    f1_result = f1_rate/image_counter
    f1_score_list.append(f1_result)
    print("The f1 score for {} Dataset is {}".format(dirname, f1_result))

data = [dirname_list, image_counter_list, f1_score_list]
total_average = sum(f1_score_list)/len(f1_score_list)
with open('model.csv','w', newline='' ) as myfile:
        wr = csv.writer( myfile,  delimiter=',')
        wr.writerows(data)
        wr.writerow(total_average)
print('Evaluation OVER!')
