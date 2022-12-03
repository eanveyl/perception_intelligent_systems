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
import matplotlib.pyplot as plt
from rrt import RRT, dijkstra, plot
from matplotlib import collections  as mc
from load import follow_path
from tqdm import tqdm


target_colors = {
    "refrigerator": [255, 0, 0], 
    "rack": [0, 255, 133], 
    "cushion": [255, 9, 92], 
    "lamp": [160, 150, 20], 
    "cooktop": [7, 255, 224]
}

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

def distance(x, y):
	return np.linalg.norm(np.array(x) - np.array(y))

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
        print("Using custom voxel downsampling!")
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

def navigate_to(points_2d, ziel: str, n_iter: int, startpos: tuple=(0,0), obstacle_radius: float=0.5, stepSize: float=0.5, discretize_map: bool=False):
    global target_colors
    target_color = np.divide(target_colors[ziel], 255)
    x, y, colors = points_2d

    #startpos = (0.5, -5.75) #works(-2,-2)#works:(0,0)

    obstacles = list()
    for i, c in enumerate(colors): 
        if target_color[0] == c[0] and target_color[1] == c[1] and target_color[2] == c[2]:
            t_x = x[i]
            t_y = y[i]
            continue
        else:
            if not (np.isclose(np.abs(x[i]), startpos[0]) or np.isclose(np.abs(x[i]), startpos[1])):  # this is to remove an obstacle near the start point
                obstacles.append((x[i], y[i]))
    
    print("Found target color, corresponding coordinates: {}, {}".format(t_x, t_y))
    endpos = (t_x, t_y)
    
    G = RRT(startpos, endpos, obstacles, n_iter, obstacle_radius, stepSize)
    
    if G.success:
        path = dijkstra(G)
        print("Sucessfully found a path!")
        print(path)
        return G, path
    else:
        print("No path was found!")
        return G, None



if __name__ == "__main__":
    views = list()
    remove_floor_and_ceiling = True

    # This part is for loading a *given* point cloud and using it for navigation.
    coord = np.load("/home/edu/university_coding_projects/NYCU_Perception/projection_launcher/screenshots/semantic_3d_pointcloud/point.npy") * 10000/255
    views = [o3d.geometry.PointCloud()]
    views[0].points = o3d.utility.Vector3dVector(coord)  # load the coordinates
    semantics = np.load("/home/edu/university_coding_projects/NYCU_Perception/projection_launcher/screenshots/semantic_3d_pointcloud/color01.npy")
    views[0].colors = o3d.utility.Vector3dVector(semantics)  # load the colors for semantics
    views[0] = custom_voxel_down(views[0], 0.1)  # do a small downsampling to increase performance

    # Task 1: Remove floor and ceiling
    if remove_floor_and_ceiling:
        for v,_ in enumerate(views):
            point_cloud = np.asarray(views[v].points)
            colors = np.asarray(views[v].colors)
            marked_for_removal = list()
            for p in range(point_cloud.shape[0]):
                if point_cloud[p][1] > 0.5 or point_cloud[p][1] < -1.2:  # if y-axis value > 1.3 (floor) or < -0.7 (ceiling)
                    marked_for_removal.append(p)
            point_cloud = np.delete(point_cloud, marked_for_removal, axis=0)
            colors = np.delete(colors, marked_for_removal, axis=0)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            views[v] = pcd
    
    # Task 1: create 2D view of the apartment, top-down
    x = np.asarray(views[0].points).take(indices=0, axis=1)
    y = -1*np.asarray(views[0].points).take(indices=2, axis=1)
    plt.scatter(x, y, c=np.asarray(views[0].colors), s=1)  # take x (as x) and z (as y) values to show them as 2d
    plt.show()

    # Show the 3D model just for fun too
    o3d.visualization.draw_geometries(views)

    # Task 2: RRT Algorithm in the 2D view
    rrt_obstacle_radius = 0.075
    rrt_step_size = 0.5
    startpos = (0.5, -5.75) #notworking(2,-8)#works(-2,-2)#works:(0,0)#not working (0.5, -5.75)
    target = "refrigerator"

    # Execute the RRT algorithm here
    G, shortest_path = navigate_to([x, y, np.asarray(views[0].colors)], target, 2500, startpos, obstacle_radius=rrt_obstacle_radius, stepSize=rrt_step_size)

    # Retrieve created graph nodes
    px = [x for x, y in G.vertices]
    py = [y for x, y in G.vertices]

    # And plot them along the rest of the 2D world
    fig, ax = plt.subplots()
    ax.scatter(px, py, c='cyan')  # add normal graph nodes in cyan color
    ax.scatter(G.startpos[0], G.startpos[1], c='black')  # starting and ending points as black dots
    ax.scatter(G.endpos[0], G.endpos[1], c='black')

    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)  # connecting normal graph nodes in green
    ax.add_collection(lc)

    if shortest_path is not None:  # draw the shortest path if possible
        paths = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)  # draw lines along optimal path in blue
        ax.add_collection(lc2)

    for i, obs in enumerate(zip(x, y)):
        circle = plt.Circle(obs, rrt_obstacle_radius, color=np.asarray(views[0].colors)[i]) # and add the obstacles from 2D view
        ax.add_artist(circle)
    
    plt.show()

    # Homework 3 - Task 3 
    if shortest_path is not None: 
        follow_path(shortest_path, startpos)
    else: 
        print("Oops... No path was found! Can't follow path.")
    
    print("DONE")

