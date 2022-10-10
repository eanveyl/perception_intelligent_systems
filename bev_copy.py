import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.points = points
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=np.pi/2):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        # build the intrinsics projection
        f = 1#np.sqrt(self.height**2+self.width**2)/(2*np.tan(fov*0.5))
        u_o = 0#int(self.width/2)
        v_o = 0#0int(self.height/2)  # TODO this is just an assumption! Check this part! 
        K = np.matrix([[f, 0, u_o], [0, f, v_o], [0, 0, 1]])  # intrinsics projection
        print("K=" + str(K))

        # build the I_c matrix
        I_c = np.hstack((np.identity(3), np.zeros((3,1))))

        # build the camera extrinsic matrix
        # start with the translation
        t_x = 0 # translation in x
        t_y = 0  # translation in y
        t_z = 0  # translation in z
        T = np.vstack((np.hstack((np.identity(3),[[t_x], [t_y], [t_z]])), [0, 0, 0, 1]))  # translation matrix
        print("T=" + str(T))

        # rotation 
        alpha = 0  # yaw
        beta = 0#-np.pi/2  # pitch
        gamma = 0  # roll
        r_yaw = np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        r_pitch = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        r_roll = np.matrix([[1, 0 , 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
        rotation_matrix = r_yaw @ r_pitch @ r_roll
        R = np.vstack((np.hstack((rotation_matrix, np.zeros((3,1)))),[0,0,0,1]))  # rotation matrix
        print("R=" + str(R))

        # perform the transformation in total
        new_pixels = list()
        for x,y in self.points:
            u, v, *leftover = K@I_c@T@R@np.array([[x], [y], [0], [1]])  # X, Y, Z, 1  # TODO assuming z is the height of the camera above the ground (?)
            new_pixels.append([u.item()/leftover.item(), v.item()/leftover.item()])#, 1]))  # use .item() function since the u, v are unpacked as 1x1 matrices and we only need the float inside # TODO: divide u and v by the leftover to transform from homogeneous coorinates back??
            print(new_pixels)
            if leftover and np.sum(leftover)>1:
                print("warning, the computation delivered a leftover of != 1")

        

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """
        new_pixels = np.array(np.rint(new_pixels).astype(np.int32))  # round to the nearest integer to avoid problems
        new_image = cv2.fillPoly(
            self.image.copy(), [new_pixels], color)
        #new_image = cv2.fillPoly(self.image.copy(), [np.array([[710, 641], [206, 632], [696, 488], [458, 485]])], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = 0  # in degrees

    front_rgb = "screenshots/view1/front_view.png"
    top_rgb = "screenshots/view1/front_view.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
