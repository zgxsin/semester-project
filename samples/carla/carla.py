"""
Mask R-CNN
Train on the combined CARLA dataset and real-world dataset

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Guoxiang Zhou


------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 carla.py train --dataset=/path/to/original carla/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 carla.py train --dataset=/path/to/original carla/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 carla.py train --dataset=/path/to/original carla/dataset --weights=imagenet

"""
import os
import sys
import numpy as np
import skimage.draw
from os import listdir
from os.path import isfile, join
import cv2
import tarfile
import shutil
import scipy
import imgaug
import time

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
    """Configuration for training on the combined dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "carla_zurich"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    GPU_COUNT = 4
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 150

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.75
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    # set MINI mask shape, since tram is always rectangular.
    # MINI_MASK_SHAPE = (56, 90)


############################################################
#  Dataset
############################################################

class CarlaDataset(utils.Dataset):

    def load_carla(self, dataset_dir, subset, original_directory):
        '''
        Load a subset of the CARLA dataset.
        :param dataset_dir: where dataset images are
        :param subset: 'train' or 'val'
        :param original_directory: where original dataset images are, we unzip the original
         dataset images to dataset_dir
        :return: None
        '''

        # Add classes. We have only one class 'dynamic' to add.
        self.add_class("carla", 1, "Dynamic")

        # load train or validation dataset
        assert subset in ["train", "val"]
        # the structure of directory is: dataset_dir-->{train, val}, train-->{RGB, Mask}, val-->{RGB, Mask}
        dataset_dir = os.path.join(dataset_dir, subset)
        mask_path = os.path.join(dataset_dir, "Mask")

        ######################################################
        # unzip data in the original direcotry to dataset_dir
        ######################################################
        # delete the directory 'dataset_dir' first if it exists
        directory = dataset_dir
        if not os.path.exists(directory):
            os.makedirs(directory)


        with tarfile.open(os.path.join(original_directory, subset, "RGB.tar"), 'r' ) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=directory)
            tar.close()


        with tarfile.open(os.path.join(original_directory, subset, "Mask.tar"), 'r' ) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=directory)
            tar.close()
        ##########################################################


        mask_list = [f for f in listdir(mask_path) if isfile(join(mask_path,f))]
        image_counter = 0
        for i, filename in enumerate(mask_list):
            image_path = os.path.join(dataset_dir,"RGB",filename)
            try:
                image = skimage.io.imread(image_path)
            except:
                print("Not a valid image: ",image_path)
                continue
            height, width = image.shape[:2]
            mask_temp = skimage.io.imread(os.path.join(mask_path, filename), as_grey=True)
            # mask has to be bool type
            mask_temp = mask_temp > 0
            mask_temp = np.asarray(mask_temp, np.uint8)
            masks = []
            # extract instances masks from one single mask of the image
            connectivity = 8
            output = cv2.connectedComponents(mask_temp, connectivity, cv2.CV_32S)
            # Get the results. The first element is the number of labels.
            num_labels = output[0]
            labels = output[1]
            # number of mask instances: count
            count = 0
            # zero represents the background, for loop starts from 1
            for i in range(1, num_labels):
                # robust to noise, the instance region must has more than 20 pixels
                temp = labels == i
                temp = scipy.ndimage.morphology.binary_fill_holes( temp ).astype( np.bool )
                if np.sum( temp ) >= 20:
                    masks.append( temp )
                    count = count + 1
            masks = np.asarray(masks)
            # if an image doesn't have instance masks, skip it.
            if not masks.size>0:
                continue

            self.add_image(
                "carla",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=masks)
            image_counter = image_counter+1

        string = "trainging" if subset=="train" else "validation"
        print("The number of {0} samples is {1} at CARLA Dataset".format(string, image_counter))

    def load_test(self, dataset_dir, subset):
        """Load a subset of the CARLA dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class( "carla", 1, "Dynamic" )

        # load train or validation dataset
        assert subset in ["train", "val"]
        # dataset_dir_origin = dataset_dir
        dataset_dir = os.path.join( dataset_dir, subset)
        mask_path = os.path.join(dataset_dir, "Mask")

        #############
        mask_list = [f for f in listdir( mask_path ) if isfile( join( mask_path, f ) )]

        for i, filename in enumerate( mask_list ):
            image_path = os.path.join( dataset_dir, "RGB", filename )
            image = skimage.io.imread( image_path )
            height, width = image.shape[:2]
            mask_temp = skimage.io.imread( os.path.join( mask_path, filename ), as_grey=True )
            # mask has to be bool type
            mask_temp = mask_temp > 0
            mask_temp = np.asarray( mask_temp, np.uint8 )
            masks = []
            # extract instances masks from one single mask of the image
            connectivity = 8
            output = cv2.connectedComponents( mask_temp, connectivity, cv2.CV_32S )
            # Get the results. The first element is the number of labels
            num_labels = output[0]
            labels = output[1]
            # number of mask instances: count
            count = 0
            # zero represent the background, for loop starts from 1
            for i in range( 1, num_labels ):
                # robust to noise, the instance region mush has more than 20 pixels
                if np.sum( labels == i ) >= 20:
                    masks.append( (labels == i) )
                    count = count + 1
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
        # If not a carla dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "carla":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = info["polygons"].reshape(info["height"], info["width"], -1)
        mask_accu = info["polygons"]
        mask = np.empty(shape=(info["height"], info["width"], mask_accu.shape[0]), dtype=np.bool)
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

    def load_zurich(self, subset, video_image_directory=None):
        '''
        Load a subset of the Zurich dataset.
        :param subset: 'train' or 'val'
        :param video_image_directory: where video images are
        :return: None
        '''

        self.add_class( "zurich", 1, "Dynamic" )
        assert subset in ["train", "val"]
        extend_save_directory = os.path.join(video_image_directory, subset)
        image_dir = os.path.join( extend_save_directory, "RGB" )
        mask_dir = os.path.join( extend_save_directory, "Mask" )
        image_list = [f for f in listdir( image_dir ) if isfile( join( image_dir, f ) )]
        image_counter = 0
        for i, filename in enumerate(image_list):
            image_path = os.path.join( image_dir, filename )
            try:
                image = skimage.io.imread( image_path )
            except:
                print( image_path, " Can't read this image!" )
                continue
            height, width = image.shape[:2]
            masks = []
            for i in range(100):
                mask_filename = filename.split( "." )[0] + "__CC%.1d" % i + ".png"
                mask_path = os.path.join( mask_dir, mask_filename )
                try:
                    mask_temp = skimage.io.imread( mask_path, as_grey=True )
                except:
                    # print("Read Mask Over!")
                    break
                    # mask has to be bool type
                mask_temp = mask_temp > 0
                # mask_temp = np.asarray( mask_temp, np.uint8 )
                masks.append( mask_temp )
            masks = np.asarray( masks, np.bool )
            if not masks.size>0:
                continue
            self.add_image(
                "zurich",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=masks )
            image_counter = image_counter + 1

        string = "training" if subset == "train" else "validation"
        print( "The number of {0} samples is {1} at Zurich Dataset".format( string, image_counter ) )


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a zurich dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "zurich":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = info["polygons"].reshape(info["height"], info["width"], -1)
        mask_accu = info["polygons"]
        mask = np.empty(shape=(info["height"], info["width"], mask_accu.shape[0]), dtype=np.bool)
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
    # Todo: change to your directory
    original_carla_directory = "/Users/zhou/Desktop/carla_2"
    unzip_directory = args.dataset
    if os.path.exists( unzip_directory ):
        shutil.rmtree( unzip_directory )
    # Todo: change to your directory
    video_image_directory = "/Users/zhou/Desktop/hh"  # local


    dataset_train = CarlaDataset()
    dataset_train.load_carla(args.dataset, "train", original_directory= original_carla_directory)
    dataset_train.prepare()

    dataset_train2 = ZurichDataset()
    dataset_train2.load_zurich("train", video_image_directory=video_image_directory)
    dataset_train2.prepare()

    dataset_train_list = [dataset_train,dataset_train2]

    # Validation dataset
    dataset_val = CarlaDataset()
    dataset_val.load_carla(args.dataset, "val", original_directory=  original_carla_directory)
    dataset_val.prepare()

    dataset_val2 = ZurichDataset()
    dataset_val2.load_zurich("val", video_image_directory=video_image_directory)
    dataset_val2.prepare()

    dataset_val_list = [dataset_val, dataset_val2]

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers.
    augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.Flipud(0.2),

                    imgaug.augmenters.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                             rotate=(-15, 15),
                        shear=(-16, 16),
                    ),
                    imgaug.augmenters.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))

                ])
    print("Training all network layers")

    # Notice the training and val samples for each data set, set the steps rationally
    # default carla rate = 0.5

    # Training - Stage 1
    # Finetune layers from ResNet stage 5 and up
    print( "Fine tune Resnet stage 5 and up" )
    model.train(dataset_train_list, dataset_val_list,
                 learning_rate=config.LEARNING_RATE,
                 epochs=50,
                 layers='5+',
                 augmentation=augmentation, carla_rate=1)

    # Training - Stage 2
    # Fine tune heads layers
    print( "Fine tune heads layers" )
    model.train( dataset_train_list, dataset_val_list,
                 learning_rate=config.LEARNING_RATE / 10,
                 epochs=100,
                 layers='heads',
                 augmentation=augmentation, carla_rate=0)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect dynamic regions.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/original carla/dataset/",
                        help='Directory of the CARLA dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

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
            # TODO: if you want to do inference, move to the jupyter notebook
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.5
            LOSS_WEIGHTS = {
                "rpn_class_loss": 0.,
                "rpn_bbox_loss": 0.,
                "mrcnn_class_loss": 0.,
                "mrcnn_bbox_loss": 0.,
                "mrcnn_mask_loss": 1.
            }

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
        weights_path = model.find_last()
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
