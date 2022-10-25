from lib2to3.pytree import convert
import os
from PIL import Image

def convert_png_to_jpg(scan_directory, output_directory):
    for filename in os.listdir(scan_directory):
        current_file_path = os.path.join(scan_directory, filename)
        with open(current_file_path, 'r') as f: # open in readonly mode
        # do your stuff
            im1 = Image.open(current_file_path)
            im1.save(output_directory + filename.split(".")[0] + ".jpg")

def create_image_indexes(training_image_directory, training_relative_path, annotation_relative_path, filename):
    with open(filename, 'a') as my_file:
        for filename in os.listdir(training_image_directory):
            #current_file_path = os.path.join(train_directory, filename)
            current_relative_path_train = training_relative_path + "/" + filename
            current_relative_path_annotation = annotation_relative_path + "/" + filename.split(".")[0] + ".png"

            doc = '{"fpath_img": ' + '"' + current_relative_path_train + '", "fpath_segm": ' + '"' + current_relative_path_annotation + '", "width": 512, "height": 512} \n'
            my_file.write(doc)

if __name__ == "__main__":


    # For converting from .png to .jpg
    #scan_directory = "/home/edu/university_coding_projects/NYCU_Perception/projection_launcher/images/test"
    #output_directory = "/home/edu/Desktop/"
    #convert_png_to_jpg(scan_directory, output_directory)

    # For creating image indexes for machine learning
    training_image_directory = "/home/edu/university_coding_projects/NYCU_Perception/semantic-segmentation-pytorch-master/data/HabitatScenesV0/images/train"
    training_relative_path = "HabitatScenesV0/images/train"  # so it starts at the root directory "data"
    annotation_relative_path = "HabitatScenesV0/annotations/train"
    create_image_indexes(training_image_directory, training_relative_path, annotation_relative_path, "habitattraining.odgt")

    training_image_directory = "/home/edu/university_coding_projects/NYCU_Perception/semantic-segmentation-pytorch-master/data/HabitatScenesV0/images/val"
    training_relative_path = "HabitatScenesV0/images/val"
    annotation_relative_path = "HabitatScenesV0/annotations/val"
    create_image_indexes(training_image_directory, training_relative_path, annotation_relative_path, "habitatvalidation.odgt")
