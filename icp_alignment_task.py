import cv2
from matplotlib.pyplot import axis
import numpy as np
import open3d as o3d
from yaml import load
import bev_copyv2

def transform_2d_to_3d(value, focal_length, height_z, axis_displacement):
    return (value-axis_displacement)*height_z/focal_length 

def load_img_to_pcd(path, f, axis_displacement):
    img = cv2.imread(path)

    rows, cols,_ = img.shape

    coord_3d = list()
    for i in range(rows):
        for j in range(cols):
            z = img[i,j][0]/255 * 20  # z value normalized to 20m (only estimate) TODO check the estimate
            coord_3d.append([(j-axis_displacement)*z/f, (i-axis_displacement)*z/f, z])  # we do the unprojection from 2d to 3d in this step
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord_3d)

    return pcd



if __name__ == "__main__":

    f = 256  # focal length
    axis_displacement = 256

    '''
    depth_img1 = cv2.imread("screenshots/depview1/front_depth_view.png")
    depth_img2 = cv2.imread("screenshots/depview2/front_depth_view.png")

    rows,cols,_ = depth_img1.shape

    img1_coordinates_3d = list()
    for i in range(rows):
        for j in range(cols):
            z = depth_img1[i,j][0]/255 * 20  # z value normalized to 20m (only estimate) TODO check the estimate
            img1_coordinates_3d.append([(j-axis_displacement)*z/f, (i-axis_displacement)*z/f, z])  # we do the unprojection from 2d to 3d in this step

    
    pcd = o3d.geometry.PointCloud() #o3d.io.read_point_cloud(img1_coordinates_3d, format="xyz", print_progress=False)
    pcd.points = o3d.utility.Vector3dVector(img1_coordinates_3d)
    o3d.visualization.draw_geometries([pcd])
    '''

    pcd1 = load_img_to_pcd("screenshots/depview1/front_depth_view.png", f, axis_displacement)
    o3d.visualization.draw_geometries([pcd1])

    pcd2 = load_img_to_pcd("screenshots/depview2/front_depth_view.png", f, axis_displacement)
    o3d.visualization.draw_geometries([pcd1, pcd2])

            
