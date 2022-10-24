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

if __name__ == "__main__":

    scan_directory = "/home/edu/university_coding_projects/NYCU_Perception/projection_launcher/images/test"
    output_directory = "/home/edu/Desktop/"

    convert_png_to_jpg(scan_directory, output_directory)