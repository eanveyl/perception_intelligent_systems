from gettext import translation
from unittest import result
import cv2
import numpy as np
import open3d as o3d
import copy
import time
import scipy.optimize
from collections import Counter
import matplotlib.colors

def clean_coordinates(points, threshold_length=0.46):  # 0.46 since its the absolute distance between steps (translation)
    points = np.array(points)  # convert to a numpy array
    remove_indexes = list()

    for i in range(len(points)):
        if i+1 == len(points):
            break
        
        if np.linalg.norm(np.subtract(points[i], points[i+1])) < threshold_length:  # the length of the vector
            remove_indexes.append(i+1)
    
    points = np.delete(points, np.r_[remove_indexes], axis=0)  # remove the list of indexes along the rows
    
    return points  

def transformation_matrix_from_angles(t_x=0, t_y=0, t_z=0, yaw=0, pitch=0, roll=0):
    T = np.vstack((np.hstack((np.identity(3),[[t_x], [t_y], [t_z]])), [0, 0, 0, 1]))  # translation matrix
    #print("T=" + str(T))

    alpha = yaw
    beta = pitch
    gamma = roll
    r_yaw = np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    r_pitch = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    r_roll = np.matrix([[1, 0 , 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
    rotation_matrix = r_yaw @ r_pitch @ r_roll
    R = np.vstack((np.hstack((rotation_matrix, np.zeros((3,1)))),[0,0,0,1]))  # rotation matrix
    return T@R

def pcd_rescale(pcd, scale):
    return o3d.utility.Vector3dVector(np.asarray(pcd.points)*scale)

def alignment_loss_function(translation_and_angles, source_points, target_points):
    source_point0 = np.hstack((source_points[0], [1]))
    target_point0 = np.hstack((target_points[0], [1]))
    source_point1 = np.hstack((source_points[1], [1]))  # append a 1 at the end of the 3D vector since it is a homogeneous coordinate system
    target_point1 = np.hstack((target_points[1], [1]))
    source_point2 = np.hstack((source_points[2], [1]))
    target_point2 = np.hstack((source_points[2], [1]))
    t_x, t_y, t_z, yaw, pitch, roll = translation_and_angles  # unpack the values
    transformation_matrix = transformation_matrix_from_angles(t_x=t_x, t_y=t_y, t_z=t_z, yaw=yaw, pitch=pitch, roll=roll)
    v1 = np.linalg.norm(np.subtract(transformation_matrix@source_point1, target_point1))
    v2 = np.linalg.norm(np.subtract(transformation_matrix@source_point0, target_point0))  
    v3 = np.linalg.norm(np.subtract(transformation_matrix@source_point2, target_point2)) # distance from one vector to the other one MINIMIZE THIS

    loss = v1 + v2 + v3

    #print("Loss=" + str(loss) + " | transformation_matrix=" + str(transformation_matrix))
    #print("Loss=" + str(loss) + " | v1=" + str(v1) + " v2=" + str(v2) + " v3=" + str(v3))
    return loss

def custom_voxel_down(pcd, voxel_size):
    pcd_down, _, merged_points_list = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), approximate_class=True)
    pool_colors = list()
    for merged_points in merged_points_list:  # this is a workaround since the approximate_class option is not really working
        colors = [matplotlib.colors.to_hex(np.asarray(pcd.colors)[point]) for point in merged_points]  # gather the colors and convert them to hex for easy counting
        #print(colors)
        occurences = Counter(colors)
        #print(occurences.most_common())
        pool_colors.append(np.array(matplotlib.colors.to_rgb(occurences.most_common(1)[0][0])))  # only append the most common color inside each voxel
    pcd_down.colors = o3d.utility.Vector3dVector(pool_colors)

    return pcd_down


def match_orientation(source_vector, target_vector, method=None):
    sol = scipy.optimize.minimize(alignment_loss_function, x0=np.random.rand(6,1), args=(source_vector, target_vector), method=method)
    print(sol)
    return sol

def depth_image_to_point_cloud(path_depth, path_rgb, f, axis_displacement):
    img_depth = cv2.imread(path_depth)
    img_rgb = cv2.imread(path_rgb)

    rows, cols,_ = img_depth.shape

    coord_3d = list()
    rgb_3d = list()
    for i in range(rows):
        for j in range(cols):
            z = img_depth[i,j][0]/255 * 10  # z value normalized to 20m (only estimate) TODO check the estimate
            coord_3d.append([(j-axis_displacement)*z/f, (i-axis_displacement)*z/f, z])  # we do the unprojection from 2d to 3d in this step
            rgb_3d.append([img_rgb[i,j][2]/255, img_rgb[i,j][1]/255, img_rgb[i,j][0]/255])  # CV2 uses BGR and O3D uses RGB
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord_3d)
    pcd.colors = o3d.utility.Vector3dVector(rgb_3d)

    return pcd

def preprocess_point_cloud(pcd, voxel_size, majority_voting_downsampling=False):
    print(":: Downsample with a voxel size %.3f." % voxel_size)

    if not majority_voting_downsampling:
        pcd_down = pcd.voxel_down_sample(voxel_size)  # original
    else:
        pcd_down = custom_voxel_down(pcd, voxel_size)  # uses majority class voting to decide on the color for the resulting point after the downsampling.     


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

    views = list()
    transformation_matrices = list()
    
    # This part is used to import views generated autonomously within the global view
    highest_image_number = 30
    paths_to_depth_images = list()
    paths_to_rgb_images = list()
    for i in range(highest_image_number+1):
        paths_to_depth_images.append("/automated_views/automated_front_depth_view" + str(i) + ".png")
        paths_to_rgb_images.append("/automated_views/automated_front_rgb_view" + str(i) + ".png")
    print("Loaded following images:" + str(paths_to_depth_images))
    paths_to_depth_images.reverse()

    with open('screenshots/automated_views/ground_truth.txt') as f_gt:
        lines = f_gt.readlines()
    ground_truth_coordinates = list()
    for l in lines: 
        l.strip()  # remove the newline
        x_gt, y_gt, z_gt = l.split()
        ground_truth_coordinates.append([float(x_gt), float(y_gt), float(z_gt)])

    for i in range(len(paths_to_depth_images)):
        tic = time.time()  # used to measure loop execution time

        if i+1 == len(paths_to_depth_images):
            for n in range(len(views)):
                views[n] = views[n].transform(transformation_matrices[i-1])  # TODO not sure if I should have this for-loop here
            break
        
        pcd1 = depth_image_to_point_cloud("screenshots" + paths_to_depth_images[i], "screenshots" + paths_to_rgb_images[i], f, axis_displacement)
        pcd2 = depth_image_to_point_cloud("screenshots" + paths_to_depth_images[i+1], "screenshots" + paths_to_rgb_images[i+1], f, axis_displacement)
        print("Processing images: " + str(paths_to_depth_images[i]) + " , " + str(paths_to_depth_images[i+1]))

        voxel_size = 0.1   # 10cm #0.05  # 5cm
        source_down, source_fpfh = preprocess_point_cloud(pcd1, voxel_size, majority_voting_downsampling = True)
        target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size, majority_voting_downsampling = True)

        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        print("Global registration=" + str(result_ransac))
        #draw_registration_result(source_down, target_down, result_ransac.transformation)

        result_icp = local_icp_algorithm(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        print("ICP result=" + str(result_icp))
        #draw_registration_result(source_down, target_down, result_icp.transformation)
        print(result_icp.transformation)

        if transformation_matrices:  # if i >= 1
            for n in range(len(views)):
                views[n] = views[n].transform(transformation_matrices[i-1])  # transform all previous views to the new coordinate system to daisychain all point clouds together

        source_down.points.append([0, 0, 0])
        source_down.colors.append([1, 0, 0])

        views.append(copy.deepcopy(source_down))
        transformation_matrices.append(result_icp.transformation)  # save the latest transformation matrix, which will be used in the next iteration
        
        print("Finished iteration #" + str(i) + "/" + str(len(paths_to_depth_images)-1) + " - Total progress=" + str(int(i/(len(paths_to_depth_images)-2)*100)) + "%")
        print("Iteration time [s]=" + str(time.time() - tic))


    # Connect the estimated position with lines
    '''
    path_lines = list()
    for k in range(len(views)-1):
        l_temp = o3d.geometry.LineSet().create_from_point_cloud_correspondences(views[k], views[k+1], [(len(views[k].points)-1, len(views[k+1].points)-1)])
        l_temp.paint_uniform_color([1, 0, 0])
        path_lines.append(l_temp)
    '''  # WORK IN PROGRESS

    # Add the ground truth data
    pcd_gt = o3d.geometry.PointCloud()
    ground_truth_coordinates.reverse()
    ground_truth_coordinates = np.multiply(np.array([1,1,-1]), ground_truth_coordinates)
    pcd_gt.points = o3d.utility.Vector3dVector(ground_truth_coordinates)
    '''
    estimation_points = [view.points[-1] for view in views]
    estimation_pcd = o3d.geometry.PointCloud()
    estimation_points = clean_coordinates(estimation_points)  # remove duplicate coordinates to allow for alignment afterwards
    estimation_pcd.points = o3d.utility.Vector3dVector(estimation_points)
    estimation_pcd.paint_uniform_color([1,0,0])
    pcd_gt.points = pcd_rescale(pcd_gt, -1.75*1.0883786784134555)  
    pcd_gt.points = o3d.utility.Vector3dVector(clean_coordinates(np.asarray(pcd_gt.points)))
    pcd_gt.transform(transformation_matrix_from_angles(t_x=-0.5, t_z=0, pitch=np.pi/4-0.1, yaw=np.pi))  # good approximation
    pcd_gt.transform(transformation_matrix_from_angles(t_x=1.4096242, t_y=-0.44186634, t_z=-6.09409756, yaw=-0.08277106, pitch=-0.46497909, roll=0.07145173))
    pcd_gt.transform(transformation_matrix_from_angles(t_x=0.005692279091429608, t_y=0.101841584301287, t_z=-0.004671523940031141, yaw=0.08850188056792659, pitch=0.006484899384427202, roll=-0.023994325861561187))
    sol = match_orientation(np.asarray(pcd_gt.points)[0:3], np.asarray(estimation_pcd.points)[0:3])
    pcd_gt.transform(transformation_matrix_from_angles(t_x=sol.x[0], t_y=sol.x[1], t_z=sol.x[2], yaw=sol.x[3], pitch=sol.x[4], roll=sol.x[5]))
    '''
    # Visualize the whole thing
    pcd_gt.paint_uniform_color([0,1,0])  # paint it green
    o3d.visualization.draw_geometries(views + [pcd_gt])

    print("DONE")

