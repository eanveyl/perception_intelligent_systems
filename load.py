#export MESA_LOADER_DRIVER_OVERRIDE=i965  USE THIS
import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os


# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "/home/edu/university_coding_projects/NYCU_Perception/scenes/apartment_0/habitat/mesh_semantic.ply"
action_names = []
sim = None
agent = None
n_view = 0  # initialize the step count at 0
sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def navigateAndSee(action="", save_semantic_and_depth=True):
    if action in action_names:
        global observations  # define observations as a variable outside the navigateAndSee function, to facilitate saving an image
        global sim
        global agent
        observations = sim.step(action)
        #print("action: ", action)

        cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        cv2.imshow("semantic", transform_semantic(observations["semantic_sensor"]))
        cv2.imshow("RGB front view", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("RGB top down", transform_rgb_bgr(observations["color_sensor2"]))

        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

        global n_view
        cv2.imwrite("automated_front_rgb_view" + str(n_view) + ".png", transform_rgb_bgr(observations["color_sensor"]))
        if save_semantic_and_depth:
            cv2.imwrite("automated_front_depth_view" + str(n_view) + ".png", transform_depth(observations["depth_sensor"]))
            cv2.imwrite("automated_front_semantic_view" + str(n_view) + ".png", transform_semantic(observations["semantic_sensor"]))

        text_file = open("ground_truth.txt", "a")
        n = text_file.write(str(sensor_state.position[0]) + " " + str(sensor_state.position[1]) + " " + str(sensor_state.position[2]) + "\n")
        text_file.close()
        n_view += 1  # increases the step number to ensure we don't create file name collisions
        # return camera pose as camera pose: x y z rw rx ry rz
        return sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z

def save_img():
    cv2.imwrite("front_view.png", transform_rgb_bgr(observations["color_sensor"]))  # save as image
    cv2.imwrite("top_view.png", transform_rgb_bgr(observations["color_sensor2"]))  # save as image
    cv2.imwrite("front_depth_view.png", transform_depth(observations["depth_sensor"]))  # save as image
    print("image saved at " + str(os.getcwd()))

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # define a second camera for top-down views
    rgb_sensor_spec2 = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec2.uuid = "color_sensor2"
    rgb_sensor_spec2.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec2.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec2.position = [0.0, settings["sensor_height"]+0.9, 0]
    rgb_sensor_spec2.orientation = [
        -np.pi/2,
        0.0,
        0.0,
    ]
    rgb_sensor_spec2.sensor_subtype = habitat_sim.SensorSubType.PINHOLE


    #depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic sensor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec, rgb_sensor_spec2]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def follow_path(path: list, start_pos: tuple):
    global action_names
    global sim 
    global agent
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    
    # inverted_path = list()
    # for i,p in enumerate(path): 
    #     inverted_path.append((p[0], -1*p[1]))  # flip the y values around since the simulator uses the other direction in coordinate system
    # path = inverted_path

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([float(start_pos[0]), 0, float(start_pos[1])])  # agent in world space
    agent.set_state(agent_state)
    
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    rot_step_size = 0.08715573698282242*2
    fwd_step_size = 0.25
    fine_tune_iterations = 5

    x0, y0 = path[0]
    heading = 0
    for i, p in enumerate(path):
        if i==0:
            continue  # skip the first iteration, since it has the starting position
        for _ in range(fine_tune_iterations):
            x1, y1 = p
            r = np.sqrt((x1-x0)**2 + (y1-y0)**2)

            # translate the coordinate system to origin
            x1 = x1 - x0
            y1 = y1 - y0
            x0 = 0
            y0 = 0

            # create a unit vector with the current heading
            heading_ref_x = heading + np.pi/2
            heading_vector = (np.cos(heading_ref_x), np.sin(heading_ref_x))

            # calculate the angle difference between the two vectors 
            theta = angle_between(heading_vector, (x1, y1))
            order = "turn_left" if np.cross(heading_vector, (x1, y1)) > 0 else "turn_right"
            n_turns = int(np.floor(theta/rot_step_size))

            for _ in range(n_turns):
                x_cur, _, y_cur, _, _, ry, _ = navigateAndSee(order, save_semantic_and_depth=False)

            n_forward_steps = int(np.floor(r/fwd_step_size))  # check how many steps forward we need to do
            order = "move_forward"
            for _ in range(n_forward_steps):
                x_cur, _, y_cur, _, _, ry, _ = navigateAndSee(order, save_semantic_and_depth=False)  # move forward steps

            heading = 2*ry
            x0 = x_cur  # retrieve ground truth information
            y0 = -y_cur  # invert y because we don't like weird axes

            if np.sqrt((p[0]-x0)**2 + (p[1]-y0)**2) < fwd_step_size:  # if we are close enough (less than one step away) to our checkpoint
                print("Checkpoint {},{} reached with enough accuracy.".format(p[0],p[1]))
                break  # go to the next checkpoint 
            else:
                print("Checkpoint {},{} not reached with enough accuracy. Attempting a correction.".format(p[0],p[1]))
                # ... otherwise we will do another "fine-tuning" round

            # x1, y1 = p
            # r = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            # theta = np.arcsin((x1-x0)/r)
            # phi = np.arcsin(-(y1-y0)/r)

            # theta -= heading
            # phi += heading

            # if (-np.pi/2 < np.arcsin((x1-x0)/r) < 0) and (-np.pi/2 < np.arcsin(-(y1-y0)/r) < 0):  # next point is
            #     #... in the first quadrant (with view pointing up ) and both arcsin are actually pointing in exactly opposite
            #     #... directions to what they actually should.
            #     theta_corr = - (np.pi + np.arcsin((x1-x0)/r)) - heading  # bend it in the right direction
            #     if (-2*np.pi < theta_corr < -(3/2)*np.pi):
            #         theta_corr = 2*np.pi + theta_corr
            #     phi_corr = - (np.pi + np.arcsin(-(y1-y0)/r)) + heading  # bend it in the right direction

            #     theta = theta_corr
            #     phi = phi_corr 

            # #if 0 < theta < np.pi/2 and  0 < phi < np.pi/2:
            # if (0 < theta < np.pi or  -2*np.pi < theta < -np.pi) and (0 < phi < np.pi or -2*np.pi < phi < -np.pi):
            #     # turn right, target ahead
            #     steer = theta
            #     #steer = steer - heading  # subtract the heading we had from previous steps (if any)
            #     order = "turn_right"
            #     n_turns = int(np.floor(steer/rot_step_size))
            # elif -np.pi/2 < theta < 0 and np.pi/2 < phi < np.pi: 
            #     # turn left, target ahead
            #     steer = theta
            #     #steer = steer - heading  # subtract the heading we had from previous steps (if any)
            #     order = "turn_left"
            #     n_turns = int(np.abs(np.floor(steer/rot_step_size)))  # floor for negative numbers will round to the smaller integer!
            # #elif np.pi/2 < theta < np.pi and -np.pi/2 < phi < 0: 
            # elif 0 < theta < np.pi and -np.pi/2 < phi < 0:
            #     # turn right, target behind
            #     steer = np.pi/2 - phi
            #     #steer = steer - heading  # subtract the heading we had from previous steps (if any)
            #     order = "turn_right"
            #     n_turns = int(np.floor(steer/rot_step_size))
            # #elif -np.pi < theta < -np.pi/2 and (3/4)*np.pi < phi < 2*np.pi:
            # elif -np.pi < theta < 0 and (3/2)*np.pi < phi < 2*np.pi:
            #     # turn left, target behind
            #     steer = -np.pi/2 + phi
            #     #steer = steer - heading  # subtract the heading we had from previous steps (if any)
            #     order = "turn_left"
            #     n_turns = int(np.abs(np.floor(steer/rot_step_size)))  # floor for negative numbers will round to the smaller integer!
            # else:
            #     assert(False, "Ähm...")  # don't know yet what should happen here. TODO Probably just move one step forward

            
            # for _ in range(n_turns):
            #     x_cur, _, y_cur, _, _, ry, _ = navigateAndSee(order, save_semantic_and_depth=False)

            # n_forward_steps = int(np.floor(r/fwd_step_size))  # check how many steps forward we need to do
            # order = "move_forward"
            # for _ in range(n_forward_steps):
            #     x_cur, _, y_cur, _, _, ry, _ = navigateAndSee(order, save_semantic_and_depth=False)  # move forward steps

            # heading = -2*ry #% 2*np.pi # invert because we count right as positive heading
            # x0 = x_cur  # retrieve ground truth information
            # y0 = y_cur

            # if np.sqrt((x1-x0)**2 + (y1-y0)**2) < fwd_step_size:  # if we are close enough (less than one step away) to our checkpoint
            #     print("Checkpoint {},{} reached with enough accuracy.".format(x1,y1))
            #     break  # go to the next checkpoint 
            # else:
            #     print("Checkpoint {},{} not reached with enough accuracy. Attempting a correction.".format(x1,y1))
            #     # ... otherwise we will do another "fine-tuning" round
    

if __name__ == "__main__":  # this runs if the file is executed directly
    follow_path([(0,0), (1,1), (2,1.2), (1.95, 2.75), (3.175, 1.88), (2.175, 1.55), (1.236, 0.90), (-0.23, -0.55), (-1.657, -2.66),
    (-1.498, -5.22), (0,389, -5.26)], (0,0))

    # cfg = make_simple_cfg(sim_settings)
    # sim = habitat_sim.Simulator(cfg)

    # # initialize an agent
    # agent = sim.initialize_agent(sim_settings["default_agent"])

    # # Set agent state
    # agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array([0.0, 0, 0.0])  # agent in world space
    # agent.set_state(agent_state)

    # # obtain the default, discrete actions that an agent can perform
    # # default action space contains 3 actions: move_forward, turn_left, and turn_right
    
    # action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    # print("Discrete action space: ", action_names)

    # FORWARD_KEY="w"
    # LEFT_KEY="a"
    # RIGHT_KEY="d"
    # FINISH="f"
    # SAVE_IMG="v"
    # print("#############################")
    # print("use keyboard to control the agent")
    # print(" w for go forward  ")
    # print(" a for turn left  ")
    # print(" d for trun right  ")
    # print(" f for finish and quit the program")
    # print(" v for saving the current RGB view as an image")
    # print("#############################")

    # observations = None  # define observations as a variable outside the navigateAndSee function, to facilitate saving an image
    
    # action = "move_forward"
    # navigateAndSee(action)

    # while True:
    #     keystroke = cv2.waitKey(0)
    #     if keystroke == ord(FORWARD_KEY):
    #         action = "move_forward"
    #         navigateAndSee(action)
    #         print("action: FORWARD")
    #     elif keystroke == ord(LEFT_KEY):
    #         action = "turn_left"
    #         navigateAndSee(action)
    #         print("action: LEFT")
    #     elif keystroke == ord(RIGHT_KEY):
    #         action = "turn_right"
    #         navigateAndSee(action)
    #         print("action: RIGHT")
    #     elif keystroke == ord(FINISH):
    #         print("action: FINISH")
    #         break
    #     elif keystroke == ord(SAVE_IMG):
    #         print("action: SAVE IMAGE")
    #         save_img()
    #     else:
    #         print("INVALID KEY")
    #         continue
