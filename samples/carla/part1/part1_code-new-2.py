import numpy as np
import cv2
import os,sys
from os import listdir
from os.path import isfile, join
from skimage.segmentation import slic
import scipy
from PIL import Image



# # Functions of the process
class ZurichDataset():
    def read_video(self,directory, sample_rate, preprosessing):
        cam = cv2.VideoCapture(directory)
        count_frame = 0
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))
        index = 0
        image_list = []
        image_origin_list = []
        while True:
            ret, prev = cam.read()
            if not ret:
                break
            count_frame = count_frame + 1
            if count_frame % sample_rate == 0:
                image_origin_list.append( cv2.cvtColor( prev, cv2.COLOR_BGR2RGB ) )
                prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY )
                if preprosessing:
                    image = clahe.apply(prevgray)
                # using Gaussian Blur is not good, tested
                    image = cv2.GaussianBlur(image, (5, 5), 1)
                else:
                    image = prevgray
                image_list.append(image)

                # image_stat= clahe.apply(image_stat)
                index = index + 1
        # change to np.float32, so that i will not overflow when doing subtraction
        return np.asarray(image_list, np.float32), image_origin_list


    def calculate_image_mask(self,target_index, image_array, threshold_rate):

        image_diff = np.subtract(image_array[target_index,:,:], image_array)
        diff_image_array = np.abs(image_diff)
        diff_image_array = diff_image_array.tolist()
        del diff_image_array[target_index]
        diff_image_array = np.asarray(diff_image_array)
        # threshold the difference image.
        image_mean = np.mean(diff_image_array, axis=(1,2),dtype= np.float32)
        image_std = np.std(diff_image_array, axis=(1,2), dtype=np.float32)
        threshold_image_list = [diff_image_array[i,:,:] >= image_mean[i]+ image_std[i] for i in range(diff_image_array.shape[0])]
        threshold_image_array = np.asarray(threshold_image_list)
        sum_diff_image = np.sum(threshold_image_array, axis = 0)
        # max_value = sum_diff_image.max()
        threshold_sum_diff = (sum_diff_image >= threshold_rate*threshold_image_array.shape[0])
        threshold_sum_diff = threshold_sum_diff.astype(np.uint8)
        return sum_diff_image, threshold_sum_diff

    def morphlogical_process(self,threshold_sum_diff, open_iter=2, close_iter=3):
        # apply morphlogical operation
        kernel= cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5, 5) )
        opening = cv2.morphologyEx(threshold_sum_diff, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
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

    def extract_bboxes(self, mask):
        """Compute bounding boxes from masks.
        mask: [height, width]. Mask pixels are either 1 or 0.

        Returns: bbox array [(y1, x1, y2, x2)].
        """
        boxes = np.zeros([1, 4], dtype=np.int32 )

        m = mask
            # Bounding box.
        horizontal_indicies = np.where( np.any( m, axis=0 ) )[0]
        vertical_indicies = np.where( np.any( m, axis=1 ) )[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # else:
            #     # No mask for this instance. Might happen due to
            #     # resizing or cropping. Set bbox to zeros
            #     x1, x2, y1, y2 = 0, 0, 0, 0
            # boxes[i] = np.array( [y1, x1, y2, x2] )

        return np.abs(x2-x1), np.abs(y1-y2)


    def show_final_mask(self, num_labels, labels, iter, kernel_size, show):
        kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (kernel_size, kernel_size) )
        count = 0
        connect_components = []

        for i in range(1, num_labels):
            # robust to noise
            if not show:
                temp = labels == i
                temp = np.asarray(temp, dtype=np.uint8)
                component_after_closing = cv2.morphologyEx(temp, cv2.MORPH_DILATE, kernel, iterations=iter)
                connect_components.append( component_after_closing )
                count = count + 1

            else:
                if np.sum( labels == i ) >= 3000:
                    temp = labels == i
                    temp = np.asarray( temp, dtype=np.uint8 )
                    component_after_closing = cv2.morphologyEx(temp, cv2.MORPH_ERODE, kernel, iterations=iter)

                    if np.count_nonzero( component_after_closing ) >= 3000:
                        width, height = self.extract_bboxes(component_after_closing)
                        if np.int32(width*height) >= 4000:
                            connect_components.append( component_after_closing )
                            count = count + 1

        return count, connect_components




    def save_image(self,filename, target_index, image, mask, save_directory):
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
        mask = np.asarray(mask, np.uint8)
        for n in range(mask.shape[0]):
            binary_image = cv2.cvtColor(mask[n], cv2.COLOR_GRAY2BGR)*255
            mask_image = Image.fromarray(binary_image)
            mask_image.save(os.path.join(save_directory,"Mask",filename.split('.')[0] + "__Frame" + str(target_index) + "__CC" + str(n) +'.png'))



    def superpixel_based_processing(self,mask_image, orignal_image, rate, numSegments=1000, numCompactness=15):
        ## 1080*1920/900 = 2304
        # numSegments = 1000
        segments = slic(orignal_image, n_segments=numSegments, sigma=5, compactness=numCompactness)
        for (i, segVal) in enumerate( np.unique(segments ) ):
            temp=mask_image[segments==segVal]
            frac = np.count_nonzero(temp)/len(temp)
            if frac >= rate:
                mask_image[segments == segVal]=1
            else:
                mask_image[segments == segVal]=0
        mask_image = cv2.morphologyEx( mask_image, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (3,3)), iterations=2 )
        return mask_image

    def process_video_image(self, video_dir, subset, save_directory):
        assert subset in ["train", "val"]
        dataset_dir = os.path.join( video_dir, subset )
        video_list = [f for f in listdir( dataset_dir ) if isfile( join( dataset_dir, f ) )]
        # make directories to save video images
        extend_save_directory = os.path.join( save_directory, subset )
        image_counter = 0
        for i, filename in enumerate( video_list ):
            video_path = os.path.join( dataset_dir, filename )
            image_array, image_origin_list = self.read_video( video_path, sample_rate=30, preprosessing=False )
            # if we do not read images from the video, skip it and continue
            if image_origin_list is None:
                continue
            # print( "We sample {0} frames from the video".format( image_array.shape[0] ) )
            sample_frame_array = np.asarray(range( image_array.shape[0] ) )
            # remove first 5 frames and last 5 frames to be robust to noise
            ## todo: this can be modified
            target_indexs = sample_frame_array[2:image_array.shape[0]- 2:10]
            # target_indexs = [20]
            for target_index in target_indexs:
                # image_origin_list is RGB image
                height, width = image_origin_list[target_index].shape[:2]
                sum_diff_image, threshold_sum_diff = self.calculate_image_mask(target_index, image_array, 0.6557)

                threshold_sum_diff = scipy.ndimage.morphology.binary_fill_holes( threshold_sum_diff ).astype( np.uint8 )


                closing_1 =  threshold_sum_diff.copy()
                mask_image = self.superpixel_based_processing( mask_image=closing_1,
                                                               orignal_image=image_origin_list[target_index], rate = 0.05, numSegments=1000,numCompactness=25)

                mask_image = scipy.ndimage.morphology.binary_fill_holes(mask_image).astype( np.uint8 )
                num_labels1, labels1, closing1 = self.morphlogical_process(mask_image, open_iter=0,close_iter=0 )


                _, connect_components = self.show_final_mask(num_labels1, labels1, iter=5, kernel_size=7,
                                                              show=False )

                connect_components_array = np.asarray(connect_components, np.uint8 )
                input1 = np.asarray(np.sum( connect_components_array, 0 ) > 0, np.uint8 )

                input1 = scipy.ndimage.morphology.binary_fill_holes( input1 ).astype( np.uint8 )
                connectivity = 8
                output1 = cv2.connectedComponents( input1, connectivity, cv2.CV_32S )
                _, final_connected_components_bool_list = self.show_final_mask(output1[0], output1[1],
                                                                                 iter=5, kernel_size=7, show=True)

                self.save_image(filename, target_index, image_origin_list[target_index],
                                 final_connected_components_bool_list, save_directory=extend_save_directory )

                image_counter = image_counter + 1
        string = "training" if subset == "train" else "validation"
        print( "The number of {0} samples is {1} at Zurich Dataset".format( string, image_counter ) )


def main():
    # video_clip_directory = "/cluster/work/riner/users/zgxsin/semester_project/video_clip"
    save_video_image_directory = "/Users/zhou/Desktop/new_hh"  # local
    video_clip_directory = "/Users/zhou/Desktop/video_clip"  ## local

    # save_video_image_directory = '/cluster/work/riner/users/zgxsin/semester_project/video_clip_images'
    # video_clip_directory = "/cluster/work/riner/users/zgxsin/semester_project/video_clip"
    dataset = ZurichDataset()
    dataset.process_video_image(video_clip_directory, "train", save_directory=save_video_image_directory)
    dataset.process_video_image(video_clip_directory, "val", save_directory=save_video_image_directory )


if __name__ == '__main__':
    main()

        


