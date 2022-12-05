import numpy as np
import open3d as o3d
from collections import Counter
import matplotlib.colors
import matplotlib.pyplot as plt
from rrt import RRT, dijkstra, RRT_star
from matplotlib import collections  as mc
from load import follow_path


target_colors = {
    "refrigerator": [255, 0, 0], 
    "rack": [0, 255, 133], 
    "cushion": [255, 9, 92], 
    "lamp": [160, 150, 20], 
    "cooktop": [7, 255, 224]
}

target_colors_coords = {
    "refrigerator": (1.55, -0.18),
    "rack": (3.57, -3.8), 
    "cushion": (0.87, -7.8),
    "lamp" :  (-1.71, -6.32), 
    "cooktop": (-0.37, 1.01)
}


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


def navigate_to(points_2d, ziel: str, n_iter: int, startpos: tuple=(0,0), obstacle_radius: float=0.5, stepSize: float=0.5, discretize_map: bool=False):
    global target_colors
    target_color = np.divide(target_colors[ziel], 255)
    x, y, colors = points_2d

    obstacles = list()
    for i, c in enumerate(colors): 
        if target_color[0] == c[0] and target_color[1] == c[1] and target_color[2] == c[2]:
            #t_x = x[i]
            #t_y = y[i]
            continue
        else:
            if not (np.isclose(np.abs(x[i]), startpos[0]) or np.isclose(np.abs(x[i]), startpos[1])):  # this is to remove an obstacle near the start point
                obstacles.append((x[i], y[i]))
    
    endpos = target_colors_coords[ziel]
    print("Found target object " + str(ziel) + ", corresponding coordinates: " + str(endpos))
    
    #G = RRT_star(startpos, endpos, obstacles, n_iter, obstacle_radius, stepSize)
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
                if point_cloud[p][1] > -0.1 or point_cloud[p][1] < -1.1: #if point_cloud[p][1] > 0.5 or point_cloud[p][1] < -1.2:  # if y-axis value > 1.3 (floor) or < -0.7 (ceiling)
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
    rrt_obstacle_radius = 0.1#0.075
    rrt_step_size = 0.5
    startpos = (0,0)#(0.5, -5.75) #notworking(2,-8)#works(-2,-2)#works:(0,0)#not working (0.5, -5.75)
    target = "cushion" #works"rack" #works"cushion" #works"cooktop" #"#works"refrigerator"

    # Execute the RRT algorithm here
    G, shortest_path = navigate_to([x, y, np.asarray(views[0].colors)], target, 5500, startpos, obstacle_radius=rrt_obstacle_radius, stepSize=rrt_step_size)

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

