from threading import local
from unittest import result
import cv2
from cv2 import merge
from matplotlib.pyplot import axis
import numpy as np
import open3d as o3d
from yaml import load
import copy

def depth_image_to_point_cloud(path, f, axis_displacement):
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

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def local_icp_algorithm(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

if __name__ == "__main__":

    f = 256  # focal length
    axis_displacement = 256

    pcd0 = o3d.geometry.PointCloud()  # initialize base point cloud for global mapping
    views = list()
    transformation_matrices = list()
    
    paths_to_images = [
        "/dep2view1/",
        "/dep2view2/",
        "/dep2view3/",
        "/dep2view4/",
        "/dep2view5/",
        "/dep2view6/"
    ]  # working!
    '''
    paths_to_images = [
        "/dep3view1/",
        "/dep3view2/",
        "/dep3view3/",
        "/dep3view4/",
        "/dep3view5/",
        "/dep3view6/",
        "/dep3view7/",
        "/dep3view8/",
        "/dep3view9/",
        "/dep3view10/",
        "/dep3view11/",
        "/dep3view12/",
        "/dep3view13/",
        "/dep3view14/",
        "/dep3view15/",
        "/dep3view16/",
        "/dep3view17/",
        "/dep3view18/",
        "/dep3view19/",
        "/dep3view20/",
    ]  # working!
    '''
    
    for i in range(len(paths_to_images)):

        if i+1 == len(paths_to_images):
            break

        pcd1 = depth_image_to_point_cloud("screenshots" + paths_to_images[i] + "front_depth_view.png", f, axis_displacement)
        pcd2 = depth_image_to_point_cloud("screenshots" + paths_to_images[i+1] + "front_depth_view.png", f, axis_displacement)
        

        voxel_size = 0.05  # 5cm
        source_down, source_fpfh = preprocess_point_cloud(pcd1, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size)

        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        print(result_ransac)

        #draw_registration_result(source_down, target_down, result_ransac.transformation)

        result_icp = local_icp_algorithm(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        print(result_icp)
        #draw_registration_result(source_down, target_down, result_icp.transformation)
        print(result_icp.transformation)

        if transformation_matrices:  # if i >= 1
            for n in range(len(views)):
                views[n] = views[n].transform(transformation_matrices[i-1])

        views.append(copy.deepcopy(source_down.transform(result_icp.transformation)))
        views.append(copy.deepcopy(target_down))
        transformation_matrices.append(result_icp.transformation)
        
        print("Finished iteration " + str(i))

        
        #o3d.visualization.draw_geometries(views)

    o3d.visualization.draw_geometries(views)




    pcd3 = depth_image_to_point_cloud("screenshots/dep2view6/front_depth_view.png", f, axis_displacement)