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
from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


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
    IMAGES_PER_GPU = 2
    GPU_COUNT = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # set MINI mask shape, since tram is always rectangular.
    # MINI_MASK_SHAPE = (56, 90)


############################################################
#  Dataset
############################################################

class CarlaDataset(utils.Dataset):

    def load_carla(self, dataset_dir, subset, original_directory):
        '''

        :param dataset_dir: working directory: /scratch/zgxsin/dataset/; unzip orignal data to "unzip_directory" where we will be working
        :param subset: train/val
        :param unzip_directory:
        :return:
        '''
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("carla", 1, "Dynamic")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        mask_path = os.path.join(dataset_dir, "Mask")

        ##############
        # copy data in the original direcotry to /scratch/zgxsin/dataset/
        ## orignal training data: /cluster/work/riner/users/zgxsin/semester_project/dataset/train
        ## oringal val data: /cluster/work/riner/users/zgxsin/semester_project/dataset/val
        ## command line: python3 carla.py train --dataset="/scratch/zgxsin/dataset/" --weights=coco
        #############
        # delete the directory first

        directory = dataset_dir

        if not os.path.exists(directory):
            os.makedirs(directory)


        with tarfile.open(os.path.join(original_directory, subset, "RGB.tar"), 'r' ) as tar:
            tar.extractall(path=directory)
            tar.close()


        with tarfile.open(os.path.join(original_directory, subset, "Mask.tar"), 'r' ) as tar:
            tar.extractall(path=directory)
            tar.close()

        #############
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
            masks = np.asarray(masks)
            self.add_image(
                "carla",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=masks)
            image_counter = image_counter+1

        string = "trainging" if subset=="train" else "validation"
        print("The number of {0} samples is {1} at CARLA Dataset".format(string, image_counter))

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

        cam = cv2.VideoCapture(directory)
        # fgbg = cv2.createBackgroundSubtractorMOG2( varThreshold=16, detectShadows=False )
        # kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (4, 4) )
        count_frame = 0
        clahe = cv2.createCLAHE( clipLimit=3, tileGridSize=(4, 4) )
        index = 0
        # fgmask_list = []
        image_list = []
        image_origin_list = []
        while True:
            ret, prev = cam.read()
            if not ret:
                break
            # fgmask = fgbg.apply( prev )
            count_frame = count_frame + 1
            if count_frame % sample_rate == 0:
                # the resize function can be ignored later
                # prev = cv.resize(prev, (800, 800))
                # add as RGB format
                image_origin_list.append(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB))
                # fgmask = cv2.morphologyEx( fgmask, cv2.MORPH_OPEN, kernel, iterations=2 )
                # fgmask = np.asarray( fgmask == 255, np.uint8 )
                # fgmask_list.append(fgmask)
                prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY )

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
        return np.asarray(image_list, np.float32), image_origin_list

    def calculate_image_mask(self,target_index, image_array, threshold_rate):
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



    def morphlogical_process(self, threshold_sum_diff, fgmask):
        # apply morphlogical operation
        kernel = np.ones( (4, 4), np.uint8 )
        kernel1 = np.ones( (2, 2), np.uint8 )
        # suppress noise using opening (erosion + dilation)

        threshold_sum_diff = np.asarray((threshold_sum_diff + fgmask) > 0, np.uint8 )
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
            if np.sum( labels == i ) >= 1000:

                temp = labels == i
                temp = np.asarray(temp, dtype=np.uint8 )

                # component_after_closing = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel, iterations=4)
                # component_after_closing is ndarray, np.uint8
                if not show:
                    component_after_closing = cv2.morphologyEx( temp, cv2.MORPH_DILATE, kernel, iterations=iter)

                else:
                    # plt.figure( "Sample Frame " + str( target_index ) + ": Connected Components " + str( count + 1 ) )
                    # plt.title( "Sample Frame " + str( target_index ) + ": Connected Components " + str( count + 1 ) )
                    component_after_closing = cv2.morphologyEx( temp, cv2.MORPH_ERODE, kernel, iterations=iter )
                    # plt.imshow( component_after_closing, 'gray' )
                    # print( "The Number of Pixels in Component {0} is {1}".format( count + 1, np.sum( temp ) ) )
                # when show = True, we do erosion, so the forground pixels may be below 300, which should be excluded.
                if np.count_nonzero(component_after_closing) >= 1000:
                    connect_components.append(component_after_closing )

                    count = count + 1
        return count, np.asarray(connect_components, dtype=bool)

    def superpixel_based_processing(self,mask_image, orignal_image):
        numSegments = 3000
        segments = slic(orignal_image, n_segments=numSegments, sigma=5, compactness=10 )

        for (i, segVal) in enumerate( np.unique( segments ) ):
            temp = mask_image[segments == segVal]
            frac = np.count_nonzero( temp ) / len( temp )
            if frac >= 0.3:
                mask_image[segments == segVal] = 1
            else:
                mask_image[segments == segVal] = 0
        return mask_image

    def save_image(self, filename, target_index, image, mask, save_directory):
        '''

        :param self:
        :param image:
        :param mask:
        :param directory: directory to save video images
        :return:
        '''

        if not os.path.exists(save_directory):
            os.makedirs(save_directory )
        image = Image.fromarray(image)
        if not os.path.exists(os.path.join(save_directory, "RGB")):
            os.makedirs(os.path.join(save_directory, "RGB"))
        if not os.path.exists(os.path.join(save_directory, "Mask")):
            os.makedirs(os.path.join(save_directory, "Mask"))
        image.save(os.path.join(save_directory, "RGB", filename.split('.')[0] + "__Frame" + str(target_index) + '.png'))

        # np.uint8 is important. otherwise may cause error
        mask = mask.astype(np.uint8)
        for n in range(mask.shape[0]):
            binary_image = cv2.cvtColor(mask[n], cv2.COLOR_GRAY2BGR)*255
            mask_image = Image.fromarray(binary_image)
            mask_image.save(os.path.join(save_directory,"Mask",filename.split('.')[0] + "__Frame" + str(target_index) + "__CC" + str(n) +'.png'))



    def load_zurich(self, dataset_dir, subset,  save_directory= None, mode = 1):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("zurich", 1, "Dynamic")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        assert mode in [1,2,3]
        ##########
        # mode=1: extract images and masks directly from videos and don't save them into directory
        ##########
        if mode == 1:
            save_bool = False
            save_directory = None
            load_from_image = False

        ##########
        # mode=2: extract images and masks directly from videos and save them into directory
        ##########
        elif mode == 2:

            save_bool = True
            load_from_image = False
        ##########
        # mode=3: extract images and masks directly from saved directory based on mode 2
        ##########
        else:
            save_bool = False
            load_from_image = True



        if not load_from_image:
            # dataset_dir_origin = dataset_dir
            dataset_dir = os.path.join(dataset_dir, subset)
            video_list = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f ))]

            # make directories to save video images
            if save_bool:
                extend_save_directory = os.path.join(save_directory, subset)
                os.makedirs(extend_save_directory)

            image_counter = 0
            for i, filename in enumerate( video_list ):
                video_path = os.path.join(dataset_dir, filename)
                image_array, image_origin_list = self.read_video( video_path, sample_rate=30, preprosessing=False )
                # if we do not read images from the video, skip it and continue
                if image_origin_list is None:
                    continue
                # print( "We sample {0} frames from the video".format( image_array.shape[0] ) )
                sample_frame_array = np.asarray( range( image_array.shape[0]) )
                # remove first 5 frames and last 5 frames to be robust to noise

                ## todo: this can be modified
                # target_indexs = sample_frame_array[2:image_array.shape[0]- 2:10]

                target_indexs = [20]

                for target_index in target_indexs:
                    # image_origin_list is RGB image
                    height, width = image_origin_list[target_index].shape[:2]
                    sum_diff_image, threshold_sum_diff = self.calculate_image_mask( target_index, image_array, 0.7)
                    num_labels, labels, closing = self.morphlogical_process( threshold_sum_diff,np.zeros(shape=threshold_sum_diff.shape[:2] )  )

                    # display_process( target_index, image_origin_list, sum_diff_image, threshold_sum_diff, closing )
                    closing_1 = closing.copy()
                    mask_image = self.superpixel_based_processing(mask_image=closing_1,
                                                              orignal_image=image_origin_list[target_index])

                    num_labels, labels, closing = self.morphlogical_process(mask_image,
                                                                        fgmask=np.zeros(shape=mask_image.shape[:2] ) )

                    _, connect_components = self.show_final_mask(target_index, num_labels, labels, iter=5, kernel_size=6,
                                                                 show=False)

                    connect_components_array = np.asarray( connect_components, np.int32 )
                    input1 = np.asarray( np.sum( connect_components_array, 0 ) > 0, np.uint8 )

                    connectivity = 8
                    output1 = cv2.connectedComponents( input1, connectivity, cv2.CV_32S )
                    _, final_connected_components_bool_array= self.show_final_mask( target_index, output1[0], output1[1], iter=5, kernel_size=6, show=True )
                    if save_bool:
                        self.save_image(filename, target_index, image_origin_list[target_index], final_connected_components_bool_array, save_directory=extend_save_directory)
                    self.add_image(
                        "zurich",
                        image_id=os.path.join(filename, "Sample_Frame", str(target_index) ),  # use file name as a unique image id
                        path=image_origin_list[target_index],
                        width=width, height=height,
                        polygons=final_connected_components_bool_array)
                    image_counter = image_counter + 1

            string = "training" if subset == "train" else "validation"
            print( "The number of {0} samples is {1} at Zurich Dataset".format( string, image_counter ))

        else:
            extend_save_directory = os.path.join( save_directory, subset )
            image_dir = os.path.join(extend_save_directory, "RGB")
            mask_dir = os.path.join( extend_save_directory, "Mask" )
            image_list = [f for f in listdir(image_dir ) if isfile(join(image_dir, f ) )]
            image_counter = 0
            for i, filename in enumerate( image_list ):
                image_path = os.path.join(image_dir, filename )
                try:
                    image = skimage.io.imread(image_path)
                except:
                    print(image_path, " Can't read this image!")
                    continue
                height, width = image.shape[:2]
                # image_path.split(".")[0]
                masks = []
                ## at most 100 instances for each image
                for i in range(100):
                    mask_filename = filename.split(".")[0]+"__CC%.1d" % i +".png"
                    mask_path = os.path.join(mask_dir,mask_filename)
                    try:
                        mask_temp = skimage.io.imread(mask_path, as_grey=True )
                    except:
                        # print("Read Mask Over!")
                        break
                # mask has to be bool type
                    mask_temp = mask_temp > 0
                    # mask_temp = np.asarray( mask_temp, np.uint8 )
                    masks.append(mask_temp)
                masks = np.asarray( masks, np.bool )
                self.add_image(
                    "zurich",
                    image_id=filename,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=masks)
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
    #####################
    # handle directories begin
    #####################
    # whether or not to save video images

    # save_video_image_directory = "/cluster/work/riner/users/zgxsin/semester_project/video_images/" # leonhard
    save_video_image_directory = "/Users/zhou/Desktop/hh" # local
    load_zurich_mode = 3
    #####-------------------------------------
    # mode=1: extract images and masks directly from videos and don't save them into directory
    # mode=2: extract images and masks directly from videos and save them into directory
    # mode=3: extract images and masks directly from saved directory based on mode 2
    #####-------------------------------------
    if load_zurich_mode ==2:
        if os.path.exists(save_video_image_directory ):
            shutil.rmtree(save_video_image_directory)

    # codes for handling carla images because of Leonhard limitation
    # Todo: attention
    # original_carla_directory = "/cluster/work/riner/users/zgxsin/semester_project/dataset" # leonhard
    original_carla_directory = "/Users/zhou/Desktop/data/carla" # local
    unzip_directory = args.dataset
    if os.path.exists( unzip_directory ):
        shutil.rmtree( unzip_directory )

    # Zurich video clip directory
    # /cluster/work/riner/users/zgxsin/semester_project/dataset/train/RGB.tar
    # Todo: attention
    # video_clip_directory = "/cluster/work/riner/users/zgxsin/semester_project/video_clip" ## leonhard
    video_clip_directory = "/Users/zhou/Desktop/video_clip"   ## local
    #####################
    # handle directories over
    #####################


    dataset_train = CarlaDataset()
    dataset_train.load_carla(args.dataset, "train", original_directory= original_carla_directory)
    dataset_train.prepare()

    dataset_train2 = ZurichDataset()
    dataset_train2.load_zurich(video_clip_directory, "train", save_directory= save_video_image_directory, mode= load_zurich_mode)
    dataset_train2.prepare()

    dataset_train_list = [dataset_train,dataset_train2]

    # Validation dataset
    dataset_val = CarlaDataset()
    dataset_val.load_carla(args.dataset, "val", original_directory=  original_carla_directory)
    dataset_val.prepare()

    dataset_val2 = ZurichDataset()
    dataset_val2.load_zurich(video_clip_directory, "val", save_directory= save_video_image_directory, mode = load_zurich_mode )
    dataset_val2.prepare()

    dataset_val_list = [dataset_val, dataset_val2]
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")

    # default carla rate = 0.5
    model.train(dataset_train_list, dataset_val_list,
                learning_rate=config.LEARNING_RATE,
                epochs=70,
                layers='heads', carla_rate= 0.5)

    ##after training, delete the temp directory
    if os.path.exists("/scratch/zgxsin" ):
        shutil.rmtree( "/scratch/zgxsin" )
        print('Delete /scratch/zgxsin! ')
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
