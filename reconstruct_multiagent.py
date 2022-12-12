import cv2
import numpy as np
import open3d as o3d
import copy
import time
from collections import Counter
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import collections  as mc


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


def plot_paths(coords, nxgraph, filename):
    # Retrieve created graph nodes
    px = [x for x, y in coords]
    py = [y for x, y in coords]

    # And plot them along the rest of the 2D world
    fig, ax = plt.subplots()
    ax.scatter(px, py, c='red')  # add normal graph nodes in cyan color

    lines = [(nx.get_node_attributes(nxgraph, "pos")[e_idxs[0]], nx.get_node_attributes(nxgraph, "pos")[e_idxs[1]]) for e_idxs in nxgraph.edges]
    #lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)  # connecting normal graph nodes in green
    ax.add_collection(lc)

    #plt.show()  # buggy
    plt.savefig(filename)


def apply_transform_to_pcd_list(pcd_in: o3d.geometry.PointCloud, matrix) -> None:  # this happens implace
    pcd = copy.deepcopy(pcd_in)
    for i,_ in enumerate(pcd):
        pcd[i].transform(matrix)

    return pcd

def local_icp_algorithm(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def process_views(highest_image_number: int, path_to_folder: str, use_point_cloud_matching: bool, f: int, remove_floor_and_ceiling: bool):
    views = list()
    transformation_matrices = list()
    paths_to_depth_images = list()
    paths_to_rgb_images = list()
    for i in range(highest_image_number+1):
        paths_to_depth_images.append(path_to_folder + "/automated_front_depth_view" + str(i) + ".png")
        paths_to_rgb_images.append(path_to_folder + "/automated_front_rgb_view" + str(i) + ".png")
    print("Loaded following images:" + str(paths_to_depth_images))
    paths_to_depth_images.reverse()

    with open("screenshots" + path_to_folder + "/ground_truth_coord.txt") as f_gt:
        lines = f_gt.readlines()
    ground_truth_coordinates = list()
    for l in lines: 
        l.strip()  # remove the newline
        x_gt, y_gt, z_gt = l.split()
        ground_truth_coordinates.append([float(x_gt), float(y_gt), float(z_gt)])

    with open("screenshots" + path_to_folder + "/ground_truth_rot.txt") as f_gt:
        lines = f_gt.readlines()
    ground_truth_rotation = list()
    heading = list()
    for l in lines: 
        l.strip()  # remove the newline
        w_gt, x_gt, y_gt, z_gt = l.split()
        ground_truth_rotation.append([float(w_gt), float(x_gt), float(y_gt), float(z_gt)])
        if (float(w_gt) >= 0 and float(y_gt) <= 0) or (float(w_gt) <= 0 and float(y_gt) >= 0):
            cur_heading = np.abs(np.arcsin(float(y_gt))*2)  # if we are between 0° and 180°
        else:
            cur_heading = (np.pi - np.abs(np.arcsin(float(y_gt))*2)) + np.pi # if we are between 180° and 360° in the heading (counting from north, right turn is positive)
        print("current heading: {}°".format((cur_heading/np.pi) * 180))
        heading.append(cur_heading)  # TODO this means that our heading will be between 0 and 2*pi...
        #... but if we want to calculate any differences of angles, we must make sure that we don't falsely calculate
        #... a huge angle. e.g. 300°-15° -> 285° but it should be actually 75°

    
    coordinate_system_origins = list()
    for i in range(len(paths_to_depth_images)):
        tic = time.time()  # used to measure loop execution time

        if i+1 == len(paths_to_depth_images):
            for n in range(len(views)):
                views[n] = views[n].transform(transformation_matrices[i-1])  # TODO not sure if I should have this for-loop here
            break
        
        pcd1 = depth_image_to_point_cloud("screenshots" + paths_to_depth_images[i], "screenshots" + paths_to_rgb_images[i], f, axis_displacement)
        pcd2 = depth_image_to_point_cloud("screenshots" + paths_to_depth_images[i+1], "screenshots" + paths_to_rgb_images[i+1], f, axis_displacement)
        print("Processing images: " + str(paths_to_depth_images[i]) + " , " + str(paths_to_depth_images[i+1]))

        voxel_size = 0.1   # 10cm #0.05 # 5cm

        if use_point_cloud_matching:
            source_down, source_fpfh = preprocess_point_cloud(pcd1, voxel_size, majority_voting_downsampling=False)
            target_down, target_fpfh = preprocess_point_cloud(pcd2, voxel_size, majority_voting_downsampling=False)
            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            print("Global registration=" + str(result_ransac))
            #draw_registration_result(source_down, target_down, result_ransac.transformation)

            result_icp = local_icp_algorithm(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            print("ICP result=" + str(result_icp))
            #draw_registration_result(source_down, target_down, result_icp.transformation)
            print(result_icp.transformation)
            cur_transform = result_icp.transformation
        else:
            source_down = pcd1.voxel_down_sample(voxel_size)  # original
            g = copy.deepcopy(ground_truth_coordinates)
            g.reverse()
            g = np.array(g)
            delta_pos = g[i+1] - g[i]
            np.linalg.norm(delta_pos)

            h = copy.deepcopy(heading)
            h.reverse()
            delta_rot = h[i] - h[i+1]  # consider the case that you have a huge delta rot from the 360°->0° transition
            if np.abs(delta_rot / np.pi * 180) > 12:  # if we have a delta_rot bigger than the (known) rotation step of around 10°
                print("Catching special case for heading...")
                if np.pi < delta_rot < 2*np.pi:
                    delta_rot = 2*np.pi - delta_rot
                elif -2*np.pi < delta_rot < -np.pi:
                    delta_rot = (delta_rot + 2*np.pi)*-1
                delta_rot = delta_rot * -1

            cur_transform = transformation_matrix_from_angles(t_z=np.sqrt(delta_pos[0]**2 + delta_pos[2]**2), pitch=delta_rot)


        if transformation_matrices:  # if i >= 1
            for n in range(len(views)):
                views[n] = views[n].transform(transformation_matrices[i-1])  # transform all previous views to the new coordinate system to daisychain all point clouds together
                try:
                    coordinate_system_origins[n] = transformation_matrices[i-1]@coordinate_system_origins[n]
                except:
                    coordinate_system_origins[n] = transformation_matrices[i-1]@coordinate_system_origins[n].T

        source_down.points.append([0, 0, 0])
        source_down.colors.append([1, 0, 0])

        coordinate_system_origins.append(np.matrix([0,0,0,1]))
        views.append(copy.deepcopy(source_down))
        transformation_matrices.append(cur_transform)  # save the latest transformation matrix, which will be used in the next iteration
        
        print("Finished iteration #" + str(i) + "/" + str(len(paths_to_depth_images)-1) + " - Total progress=" + str(int(i/(len(paths_to_depth_images)-2)*100)) + "%")
        print("Iteration time [s]=" + str(time.time() - tic))


    # Add the ground truth data
    pcd_gt = o3d.geometry.PointCloud()
    #ground_truth_coordinates.reverse()  # because we now read from last to first image
    ground_truth_coordinates = np.multiply(np.array([1,1,-1]), ground_truth_coordinates)  # invert the z axis

    # Add the estimated coordinate origins
    coordinate_system_origins[-1] = coordinate_system_origins[-1].T  # flip the last one because it is horizontally aligned and not vertical
    calibration = np.subtract(ground_truth_coordinates[0], coordinate_system_origins[-1][0:3])[0]  # calculate the difference between the first two coordinates
    ground_truth_coordinates_calibrated = np.subtract(ground_truth_coordinates, calibration)  # and use that to move the ground truth ever so slightly to align the first points
    pcd_gt.points = o3d.utility.Vector3dVector(ground_truth_coordinates_calibrated)

    # Visualize the whole thing
    pcd_gt.paint_uniform_color([0,1,0])  # paint it green

    if remove_floor_and_ceiling:
        for v,_ in enumerate(views):
            point_cloud = np.asarray(views[v].points)
            colors = np.asarray(views[v].colors)
            marked_for_removal = list()
            for p in range(point_cloud.shape[0]):
                if point_cloud[p][1] > 1.3 or point_cloud[p][1] < -0.7:  # if y-axis value > 1.3 (floor) or < -0.7 (ceiling)
                    marked_for_removal.append(p)
            point_cloud = np.delete(point_cloud, marked_for_removal, axis=0)
            colors = np.delete(colors, marked_for_removal, axis=0)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            views[v] = pcd

    return ground_truth_coordinates, ground_truth_rotation, heading, views, coordinate_system_origins, transformation_matrices, pcd_gt


if __name__ == "__main__":
    f = 256  # focal length
    axis_displacement = 256
    remove_floor_and_ceiling = True
    use_point_cloud_matching = False  # when true, use the ICP algorithm. Otherwise estimate the transformation matrix from the robot's movement

    # This part is used to import views generated autonomously within the global view
    # Process Agent A
    highest_image_number = 40
    a1_ground_truth_coordinates, a1_ground_truth_rotation, a1_heading, a1_views, a1_coordinate_system_origins, a1_transformation_matrices, a1_pcd_gt = process_views(highest_image_number, "/automated_views_multirobot1", use_point_cloud_matching, f, remove_floor_and_ceiling)
    o3d.visualization.draw_geometries(a1_views + [a1_pcd_gt])
    
    a1_walkable = list(zip([cs[0].item() for cs in a1_coordinate_system_origins], [cs[2].item() for cs in a1_coordinate_system_origins]))
    a1_walkable.reverse()
    a1_graph = nx.Graph()
    for i, e in enumerate(a1_walkable):
        a1_graph.add_node(i, pos=e)
        if i == 0:
            continue
        else: 
            a1_graph.add_edge(i-1, i)

    plot_paths(a1_walkable, a1_graph, "a1_walkable.png")
    # TRY THIS: http://www.open3d.org/docs/0.7.0/tutorial/Basic/visualization.html

    # Process Agent B
    highest_image_number = 19
    a2_ground_truth_coordinates, a2_ground_truth_rotation, a2_heading, a2_views, a2_coordinate_system_origins, a2_transformation_matrices, a2_pcd_gt = process_views(highest_image_number, "/automated_views_multirobot2", use_point_cloud_matching, f, remove_floor_and_ceiling)
    o3d.visualization.draw_geometries(a1_views + a2_views)
    
    a2_walkable = list(zip([cs[0].item() for cs in a2_coordinate_system_origins], [cs[2].item() for cs in a2_coordinate_system_origins]))
    a2_walkable.reverse()
    a2_graph = nx.Graph()
    for i, e in enumerate(a2_walkable):
        a2_graph.add_node(i, pos=e)
        if i == 0:
            continue
        else: 
            a2_graph.add_edge(i-1, i)

    plot_paths(a2_walkable, a2_graph, "a2_walkable.png")

    # Now, let Agent A and Agent B meet
    # This works, but it's kinda cheating. 
    v = np.subtract(a2_ground_truth_coordinates[0], a1_ground_truth_coordinates[0])
    delta_heading = a1_heading[0] - a2_heading[0]  # heading difference at their start positions. Typically 0. TODO for delta_heading different than 0 we should check this code if it's no buggy
    bridge_transform = transformation_matrix_from_angles(t_x=v[0], t_y=v[1], t_z=v[2], pitch=delta_heading)
    a2_views = apply_transform_to_pcd_list(a2_views, bridge_transform)  # this happens inplace
    a2_coordinate_system_origins = [bridge_transform@cs for cs in a2_coordinate_system_origins]  # transform the coordinate system origins

    o3d.visualization.draw_geometries(a1_views + a2_views)

    # careful here, we reverse() the second list because we want to append the nodes from last to first (seen from traveled order)
    a2_walkable_transformed = list(zip([cs[0].item() for cs in a2_coordinate_system_origins], [cs[2].item() for cs in a2_coordinate_system_origins]))
#    a2_walkable_transformed.reverse()
    a12_walkable = a1_walkable + a2_walkable_transformed
    a12_graph = nx.Graph()
    for i, e in enumerate(a12_walkable):
        a12_graph.add_node(i, pos=e)
        if i == 0:
            continue
        else: 
            a12_graph.add_edge(i-1, i)
    plot_paths(a12_walkable, a12_graph, "a12_walkable.png")

    print("DONE")
