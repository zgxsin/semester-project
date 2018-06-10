"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os import listdir
from os.path import isfile, join
import cv2
# import os
import tarfile
import shutil


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CarlaConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "carla_zurich"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 3
    GPU_COUNT = 4
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CarlaDataset(utils.Dataset):

    def load_carla(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("carla", 1, "Dynamic")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir_origin = dataset_dir
        dataset_dir = os.path.join(dataset_dir, subset)

        # mypath = "/Users/zhou/Downloads/MyDocuments/Imaga_DataSet_Training/0_100/Dynamic/dataset/train_mask/"
        mask_path = os.path.join(dataset_dir, "Mask")

        ##############
        # copy data in the original direcotry to /scratch/zgxsin/dataset/
        ## orignal training data: /cluster/work/riner/users/zgxsin/semester_project/dataset/train
        ## oringal val data: /cluster/work/riner/users/zgxsin/semester_project/dataset/val
        ## command line: python3 carla.py train --dataset="/scratch/zgxsin/dataset/" --weights=coco
        #############
        # delete the directory first
        # if os.path.exists("/scratch/zgxsin"):
        #     shutil.rmtree("/scratch/zgxsin")
        # directory = ["/scratch/zgxsin/dataset/train/",
        #             "/scratch/zgxsin/dataset/val/"]
        # for i in range(len( directory)):
        #     if not os.path.exists(directory[i]):
        #         os.makedirs(directory[i])
        #
        # with tarfile.open('/cluster/work/riner/users/zgxsin/semester_project/dataset/train/RGB.tar', 'r' ) as tar:
        #     tar.extractall(path=directory[0])
        #     tar.close()
        #
        # with tarfile.open('/cluster/work/riner/users/zgxsin/semester_project/dataset/train/Mask.tar', 'r' ) as tar:
        #     tar.extractall(path=directory[0])
        #     tar.close()
        #
        # with tarfile.open('/cluster/work/riner/users/zgxsin/semester_project/dataset/val/RGB.tar', 'r' ) as tar:
        #     tar.extractall(path=directory[1])
        #     tar.close()
        #
        # with tarfile.open('/cluster/work/riner/users/zgxsin/semester_project/dataset/val/Mask.tar', 'r' ) as tar:
        #     tar.extractall(path=directory[1])
        #     tar.close()

        #############
        mask_list = [f for f in listdir(mask_path) if isfile(join(mask_path,f))]

        for i, filename in enumerate(mask_list):
            image_path = os.path.join(dataset_dir,"RGB",filename)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            mask_temp = skimage.io.imread(os.path.join(mask_path, filename), as_grey=True)
            # mask has to be bool type
            mask_temp = mask_temp > 0
            mask_temp = np.asarray(mask_temp, np.uint8)
            masks = []
            # extract instances masks from one single mask of the image
            connectivity = 8
            # Perform the operation
            output = cv2.connectedComponents(mask_temp, connectivity, cv2.CV_32S)
            # Get the results
            # The first cell is the number of labels
            num_labels = output[0]
            labels = output[1]
            # number of mask instances: count
            count = 0
            # zero represent the background, strat from 1
            for i in range(1, num_labels):
                # robust to noise, the instance region mush has more than 10 pixels
                if np.sum(labels == i) >= 20:
                    masks.append((labels == i))
                    count = count + 1
            # todo: what is masks is empty?
            masks = np.asarray(masks)
            self.add_image(
                "carla",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=masks)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "carla":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = info["polygons"].reshape(info["height"], info["width"], -1)
        mask_accu = info["polygons"]
        mask = np.empty(shape=(info["height"], info["width"], mask_accu.shape[0]), dtype=np.bool)
        # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                 dtype=np.uint8)
        # for i, p in enumerate(info["polygons"]):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #     mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        for i in range(mask_accu.shape[0]):
            mask[:,:,i] = mask_accu[i,:,:]

        if mask is not None:
            return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        else:
            return mask.astype(np.bool), np.empty([0], np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "carla":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class ZurichDataset(utils.Dataset):

    def read_video(self, directory, sample_rate, preprosessing):
        '''

        :param directory: the video path
        :param sample_rate: sample one frame every "sample_rate" frame in the original video
        :param preprosessing: whether or not to apply histogram equalization and gaussian blur
        :return:
        '''

        cam = cv2.VideoCapture( directory )
        count_frame = 0
        clahe = cv2.createCLAHE( clipLimit=3, tileGridSize=(4, 4) )
        index = 0

        image_list = []
        image_origin_list = []
        while True:
            ret, prev = cam.read()
            if not ret:
                break
            count_frame = count_frame + 1
            if count_frame % sample_rate == 0:
                # the resize function can be ignored later
                # prev = cv.resize(prev, (800, 800))
                image_origin_list.append(prev)
                prevgray = cv2.cvtColor( prev, cv2.COLOR_BGR2GRAY )

                if preprosessing:
                    image = clahe.apply( prevgray )
                    # using Gaussian Blur is not good, tested
                    image = cv2.GaussianBlur(image, (5, 5), 1 )
                else:
                    image = prevgray
                image_list.append(image)

                # image_stat= clahe.apply(image_stat)
                index = index + 1
        # change to np.float32, so that i will not overflow when doing subtraction
        return np.asarray( image_list, np.float32 ), image_origin_list

    def calculate_image_mask(self,target_index, image_array, threshold_rate):
        while (True):
            image_diff = np.subtract( image_array[target_index, :, :], image_array )
            diff_image_array = np.abs( image_diff )
            diff_image_array = diff_image_array.tolist()
            del diff_image_array[target_index]
            diff_image_array = np.asarray( diff_image_array )
            # threshold the difference image.
            image_mean = np.mean( diff_image_array, axis=(1, 2), dtype=np.float32 )
            image_std = np.std( diff_image_array, axis=(1, 2), dtype=np.float32 )
            threshold_image_list = [diff_image_array[i, :, :] >= image_mean[i] + image_std[i] for i in
                                    range( diff_image_array.shape[0] )]
            threshold_image_array = np.asarray( threshold_image_list )
            sum_diff_image = np.sum( threshold_image_array, axis=0 )
            # max_value = sum_diff_image.max()
            threshold_sum_diff = (sum_diff_image >= threshold_rate * threshold_image_array.shape[0])
            threshold_sum_diff = threshold_sum_diff.astype( np.uint8 )
            return sum_diff_image, threshold_sum_diff



    def morphlogical_process(self, threshold_sum_diff):
        # apply morphlogical operation
        kernel = np.ones( (4, 4), np.uint8 )
        kernel1 = np.ones( (2, 2), np.uint8 )
        # suppress noise using opening (erosion + dilation)
        opening = cv2.morphologyEx( threshold_sum_diff, cv2.MORPH_OPEN, kernel1, iterations=1 )
        # fill the holes in forground (closing)
        closing = cv2.morphologyEx( opening, cv2.MORPH_CLOSE, kernel, iterations=3 )
        # draw connected component
        src = closing
        src = np.asarray( src, np.uint8 )
        connectivity = 8
        # Perform the operation
        output = cv2.connectedComponents( src, connectivity, cv2.CV_32S )
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        labels = output[1]
        return num_labels, labels, closing


    def show_final_mask(self, target_index, num_labels, labels, iter, kernel_size, show):
        kernel = np.ones( (kernel_size, kernel_size), np.uint8 )
        count = 0
        connect_components = []

        for i in range( 1, num_labels ):
            # robust to noise
            if np.sum( labels == i ) >= 300:

                temp = labels == i
                temp = np.asarray( temp, dtype=np.uint8 )

                # component_after_closing = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel, iterations=4)
                if not show:
                    component_after_closing = cv2.morphologyEx( temp, cv2.MORPH_DILATE, kernel, iterations=iter )
                else:
                    # plt.figure( "Sample Frame " + str( target_index ) + ": Connected Components " + str( count + 1 ) )
                    # plt.title( "Sample Frame " + str( target_index ) + ": Connected Components " + str( count + 1 ) )
                    component_after_closing = cv2.morphologyEx( temp, cv2.MORPH_ERODE, kernel, iterations=iter )
                    # plt.imshow( component_after_closing, 'gray' )
                    # print( "The Number of Pixels in Component {0} is {1}".format( count + 1, np.sum( temp ) ) )
                connect_components.append( component_after_closing )

                count = count + 1
        return count, np.asarray(connect_components, dtype=bool)


    def load_zurich(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("zurich", 1, "Dynamic")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        # dataset_dir_origin = dataset_dir
        dataset_dir = os.path.join(dataset_dir, subset)

        video_list = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f ))]

        for i, filename in enumerate( video_list ):
            video_path = os.path.join( dataset_dir, filename )
            image_array, image_origin_list = self.read_video( video_path, sample_rate=30, preprosessing=False )
            # print( "We sample {0} frames from the video".format( image_array.shape[0] ) )
            sample_frame_array = np.asarray( range( image_array.shape[0]) )

            # target_indexs = sample_frame_array[0:image_array.shape[0]:5]

            target_indexs = [20]
            for target_index in target_indexs:
                # image_origin_list is RGB image
                height, width = image_origin_list[target_index].shape[:2]
                sum_diff_image, threshold_sum_diff = self.calculate_image_mask( target_index, image_array, 0.7 )
                num_labels, labels, closing = self.morphlogical_process( threshold_sum_diff )

                # display_process( target_index, image_origin_list, sum_diff_image, threshold_sum_diff, closing )

                _, connect_components = self.show_final_mask( target_index, num_labels, labels, iter=5, kernel_size=6,
                                                         show=False )

                connect_components_array = np.asarray( connect_components, np.int32 )
                input1 = np.asarray( np.sum( connect_components_array, 0 ) > 0, np.uint8 )

                connectivity = 8
                output1 = cv2.connectedComponents( input1, connectivity, cv2.CV_32S )
                _, final_connected_components_bool_array= self.show_final_mask( target_index, output1[0], output1[1], iter=5, kernel_size=6, show=True )

                self.add_image(
                    "zurich",
                    image_id=os.path.join( filename, "Sample_Frame", str(target_index) ),  # use file name as a unique image id
                    path=image_origin_list[target_index],
                    width=width, height=height,
                    polygons=final_connected_components_bool_array)

        ##############
        # copy data in the original direcotry to /scratch/zgxsin/dataset/
        ## orignal training data: /cluster/work/riner/users/zgxsin/semester_project/dataset/train
        ## oringal val data: /cluster/work/riner/users/zgxsin/semester_project/dataset/val
        ## command line: python3 carla.py train --dataset="/scratch/zgxsin/dataset/" --weights=coco
        #############
        # delete the directory first
        # if os.path.exists("/scratch/zgxsin"):
        #     shutil.rmtree("/scratch/zgxsin")
        # directory = ["/scratch/zgxsin/dataset/train/",
        #             "/scratch/zgxsin/dataset/val/"]
        # for i in range(len( directory)):
        #     if not os.path.exists(directory[i]):
        #         os.makedirs(directory[i])
        #
        # with tarfile.open('/cluster/work/riner/users/zgxsin/semester_project/dataset/train/RGB.tar', 'r' ) as tar:
        #     tar.extractall(path=directory[0])
        #     tar.close()
        #
        # with tarfile.open('/cluster/work/riner/users/zgxsin/semester_project/dataset/train/Mask.tar', 'r' ) as tar:
        #     tar.extractall(path=directory[0])
        #     tar.close()
        #
        # with tarfile.open('/cluster/work/riner/users/zgxsin/semester_project/dataset/val/RGB.tar', 'r' ) as tar:
        #     tar.extractall(path=directory[1])
        #     tar.close()
        #
        # with tarfile.open('/cluster/work/riner/users/zgxsin/semester_project/dataset/val/Mask.tar', 'r' ) as tar:
        #     tar.extractall(path=directory[1])
        #     tar.close()

        #############

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "zurich":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = info["polygons"].reshape(info["height"], info["width"], -1)
        mask_accu = info["polygons"]
        mask = np.empty(shape=(info["height"], info["width"], mask_accu.shape[0]), dtype=np.bool)
        # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                 dtype=np.uint8)
        # for i, p in enumerate(info["polygons"]):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #     mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        for i in range(mask_accu.shape[0]):
            mask[:,:,i] = mask_accu[i,:,:]


        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "zurich":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)






def train(model):
    """Train the model."""
    # Training dataset.
    # to handle broken pipe error
    from signal import signal, SIGPIPE, SIG_DFL
    signal( SIGPIPE, SIG_DFL )
    dataset_train = CarlaDataset()
    dataset_train.load_carla(args.dataset, "train")
    dataset_train.prepare()

    dataset_train2 = ZurichDataset()
    dataset_train2.load_zurich("/Users/zhou/Desktop/video_clip", "train" )
    dataset_train2.prepare()

    dataset_train_list = [dataset_train,dataset_train2]

    # Validation dataset
    dataset_val = CarlaDataset()
    dataset_val.load_carla(args.dataset, "val")
    dataset_val.prepare()

    dataset_val2 = ZurichDataset()
    dataset_val2.load_zurich("/Users/zhou/Desktop/video_clip", "val" )
    dataset_val2.prepare()

    dataset_val_list = [dataset_val, dataset_val2]
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train_list, dataset_val_list,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')
    ##after training, delete the temp directory
    if os.path.exists("/scratch/zgxsin" ):
        shutil.rmtree( "/scratch/zgxsin" )
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/carla/dataset/",
                        help='Directory of the CARLA dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments

    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CarlaConfig()
    else:
        class InferenceConfig(CarlaConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
