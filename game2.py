import pygame
import pygame.gfxdraw
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.ndimage import rotate

import cv2

import matplotlib.pyplot as plt

class box:

    """Implements the car of the game. Accelerating, turning are done by this classes methods.
    The car is moved every frame according to the speed and direction of the car.
    """

    def __init__(self, x, y, h, w, color, vh, vw):

        self.init_x = x 
        self.init_y = y
        
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0

        self.v = 0
        self.a = 0

        self.dir = 0

        self.color = color

        self.vh = vh
        self.vw = vw

        self.turning_limit = 0.1

        self.corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        self.vision_limits = np.array([[-vw/2, -vh/2], [vw/2, -vh/2], [vw/2, vh/2], [-vw/2, vh/2]])
        
        vc = self.vision_limits

        self.vision_corners = [[vc[i,0] + self.x, vc[i,1] + self.y] for i in range(4)]

    def draw(self, surface):
        
        draw_dir = -self.dir

        c = self.corners @ -np.array([[np.cos(draw_dir), -np.sin(draw_dir)], [np.sin(draw_dir), np.cos(draw_dir)]])
        
        corners = [[c[i,0] + self.x, c[i,1] + self.y] for i in range(4)]
         
        pygame.gfxdraw.filled_polygon(surface, corners, (255, 255, 0))
        pygame.draw.lines(surface, (0,0,0), True, corners, 3)
        #pygame.draw.lines(surface, (0,0,0), True, self.vision_corners , 3)
        

        
    def move(self):
        self.v += self.a

        self.x += np.sin(self.dir) * self.v 
        self.y -= np.cos(self.dir) * self.v 

        self.v *= 0.8

        vc = self.vision_limits

        self.vision_corners = [[vc[i,0] + self.x, vc[i,1] + self.y] for i in range(4)]
        
    def turn(self, direction):

        """
        This method is called to turn the car when action is given to the car.
        """

        if abs(direction) > self.turning_limit:
            self.dir += np.sign(direction) * self.turning_limit
        else:
            self.dir += direction
    
    def accelerate(self, amount):

        """
        This method is called to change the acceleration of the car when action is given.
        """

        self.a = amount

    def reset(self):

        """
        Reset the car to initial parameters.
        """

        self.x = self.init_x
        self.y = self.init_y
        self.v = 0
        self.dir = 0
        self.a = 0

        vc = self.vision_limits

        self.vision_corners = [[vc[i,0] + self.x, vc[i,1] + self.y] for i in range(4)]


def get_track(w, h):

    """
    Loads the track points from a text file.
    """

    track_points = pd.read_csv("rata.csv", header = None)

    outer_track = track_points.iloc[:,0:2]
    inner_track = track_points.iloc[:24,2:4]

    outer_track_points = []
    inner_track_points = []


    for i in range(len(outer_track)):
        outer_track_points.append((outer_track.iloc[i, 0] / 1600 * w, outer_track.iloc[i, 1] / 1000 * h))

    for i in range(len(inner_track)):
        inner_track_points.append((inner_track.iloc[i, 0] / 1600 * w, inner_track.iloc[i, 1] / 1000 * h))

    return outer_track_points, inner_track_points

def draw_track(surface, inner, outer):

    pygame.gfxdraw.filled_polygon(surface, outer, (60, 60, 60))
    pygame.gfxdraw.filled_polygon(surface, inner, (13, 156, 0))

    pygame.draw.lines(surface, (255, 255, 255), True, outer, 5)
    pygame.draw.lines(surface, (255, 255, 255), True, inner, 5)

def intersection(l1, l2):
    
    """
    Returns the intersection of two lines l1 and l2 defined as starting and end points of those lines.
    """

    k1 = (l1[0][1] - l1[1][1]) / (l1[0][0] - l1[1][0])
    k2 = (l2[0][1] - l2[1][1]) / (l2[0][0] - l2[1][0])
    
    b1 = l1[0][1] - k1 * l1[0][0]
    b2 = l2[0][1] - k2 * l2[0][0]
    
    int_x = (b2 - b1) / (k1 - k2)
    int_y = k1 * int_x + b1

    x1_ok = min(l1[0][0], l1[1][0]) < int_x and max(l1[1][0], l1[0][0]) > int_x
    y1_ok = min(l1[0][1], l1[1][1]) < int_y and max(l1[0][1], l1[1][1]) > int_y

    x2_ok = min(l2[0][0], l2[1][0]) < int_x and max(l2[1][0], l2[0][0]) > int_x
    y2_ok = min(l2[0][1], l2[1][1]) < int_y and max(l2[0][1], l2[1][1]) > int_y

    #return (int_x, int_y)

    if y1_ok and x1_ok and x2_ok and y2_ok:
        return (int_x, int_y)
    else:
        return (-1,-1) 


def dist(l1, i):

    return (l1[0][0] - i[0])**2 + (l1[0][1] - i[1])**2


def min_track_intersection(l1, track):
    
    ints = []
    
    for i in range(len(track)):
        if (i != len(track) - 1):
            l2 = (track[i], track[i + 1])
        else:
            l2 = (track[i], track[0])

        intr = intersection(l1, l2)

        if intr[0] != -1:
            ints.append(intr)
    
    #print(ints)

    min_d = 10000000
    min_point = (0,0)

    for i in ints:
        d = dist(l1, i)
        
        if d < min_d:
            min_d = d
            min_point = i

    return min_point

def min_intersection(l1, inner, outer):

    inner_int = min_track_intersection(l1, inner)
    outer_int = min_track_intersection(l1, outer)

    if inner_int[0] == 0:
        return outer_int
    elif dist(l1, inner_int) < dist(l1, outer_int):
        return inner_int
    else:
        return outer_int



def is_out(x, y, points):

    """Determines (x,y) is outside the track."""

    counter = 0
    for i in range(len(points)):
        if (i != len(points) - 1): 
            start = points[i]
            end = points[i + 1]
        else:
            start = points[i]
            end = points[0]

        if (start[0] < x and end[0] < x):
            counter = counter
        else:

            if (end[0] == start[0]):
                k = 1000000000
            elif(end[1] == start[1]):
                k = 0.000000001
            else:
                k = (end[1] - start[1]) * 1.0 / (end[0] - start[0])
            
            b = end[1] - k * end[0]

            xa = (b - y) / (-k)

            if((xa < max(x, min(start[0], end[0]))) or (xa > max(end[0], start[0]))):
                counter = counter
            else:
                counter += 1

    if counter % 2 == 0:
        return True
    else:
        return False


def control(box, style):

    global running 

    acc = 0.2
    turning = 0.2

    if style == "manual":
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    box.accelerate(acc)
                if event.key == pygame.K_DOWN:
                    box.accelerate(-acc)
                if event.key == pygame.K_RIGHT:
                    box.turn(turning)
                if event.key == pygame.K_LEFT:
                    box.turn(-turning)


def distance(x3, y3, track):
    distances = []

    for i in range(len(track)):
        if i == len(track) - 1:
            start = track[-1]
            end = track[0]
        else:
            start = track[i]
            end = track[i + 1]

        x1 = start[0]
        y1 = start[1]
        x2 = end[0]
        y2 = end[1]

        px = x2-x1
        py = y2-y1

        norm = px*px + py*py

        u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        dist = (dx*dx + dy*dy)**.5

        distances.append(dist)

    return np.min(distances)

def process_image(obs):

        result = cv2.resize(obs, (40, 40))

        return result

class action_space:

    """Made this to implement the sample and shape functions, 
    since I didn't know what data structure had those functions."""
    
    def __init__(self):
        self.shape = (2, 1)
    
    def sample(self):
        return np.random.rand(self.shape[0]) - 0.5

class game:

    def __init__(self, draw=True, manual_control=False, n_directions=24, 
                 reward_type="speed", throttle_min=1, throttle_max=2, 
                 vision="simple", vision_size=300):

        pygame.init()

        self.display_width = 1600
        self.display_height = 1000
        
        try:
            self.surface = pygame.display.set_mode((self.display_width, self.display_height))
        except pygame.error:
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.surface = pygame.display.set_mode((self.display_width, self.display_height)) 

        #self.surface = pygame.display.set_mode((self.display_width, self.display_height)) 

        car_x = 230 / 1600 * self.display_width
        car_y = 800 / 1000 * self.display_height

        # car_x = 1420 / 1600 * self.display_width
        # car_y = 800 / 1000 * self.display_height

        self.vision_size = vision_size
        self.vision = vision

        self.car = box(car_x, car_y, 20, 10, (0,0,0), vision_size * 2, vision_size)
        self.clock = pygame.time.Clock()
        self.running = True
        self.graphics = draw
        self.manual_control = manual_control

        self.outer, self.inner = get_track(self.display_width, self.display_height)
        self.line_points = []

        self.n_directions = n_directions

        if vision == "simple":
            self.observation_space = np.zeros(self.n_directions)
        else:
            self.observation_space = np.zeros((self.vision_size * self.vision_size))
        
        self.action_space = action_space()

        self.throttle_min = throttle_min
        self.throttle_max = throttle_max
        
        self.reward_type = reward_type

        self._max_episode_steps = 1000

        dirs = np.linspace(-np.pi/2, np.pi/2, 8) + self.car.dir
        
        for d in dirs:
            self.line_points.append((self.car.x + np.cos(d) * 10, self.car.y + np.sin(d) * 10))

    def get_reduced_vision(self):

        """"
        This implements the vision that is just the distances to the wall in 12 directions.
        """

        dirs = np.linspace(0.0001, 2 * np.pi - 0.0001, self.n_directions + 1)[:-1] - self.car.dir

        x = self.car.x
        y = self.car.y

        line_points = []
        dists = []

        length = 100000

        for d in dirs:
            line_points.append((x + np.sin(d) * length, y + np.cos(d) * length))

        for l in line_points:
            intr = min_intersection(((self.car.x, self.car.y), l), self.outer, self.inner)
            dist = np.sqrt((self.car.x - intr[0])**2 + (self.car.y - intr[1])**2)
            dists.append(dist)

        self.line_points = line_points

        dists = np.array(dists)

        norm = np.sqrt(np.sum(dists * dists))

        return dists / norm

    def get_state(self):
        return self.get_vision()


    def get_vision(self):

        if self.vision == "simple":
            return self.get_reduced_vision()

        pa = np.array(pygame.PixelArray(self.surface))

        l = self.car.vision_corners

        result = np.zeros((self.car.vw, self.car.vh))

        left = max(0, int(l[0][0]))
        right = min(self.surface.get_width(), int(l[1][0]))
        top = max(0, int(l[0][1]))
        bottom = min(self.surface.get_height(), int(l[3][1]))   

        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

        if left > l[0][0]:
            pad_left = int(abs(l[0][0]))
        elif right < l[1][0]:
            pad_right = int(l[1][0] - right)

        if top > l[0][1]:
            pad_top = int(abs(l[0][1]))
        elif bottom < l[3][1]:
            pad_bottom = int(l[3][1] - bottom)

        # print(pad_left)
        # print(pad_right)
        # print(pad_top) 
        # print(pad_bottom)

        vision = pa[left:right, top:bottom].T
        vision = np.pad(vision, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        
        vision = rotate(vision, self.car.dir / np.pi * 180, reshape = False, mode = "nearest")

        red = vision >> 16
        green = (vision >> 8) - (red << 8)
        blue = vision - (red << 16) - (green << 8)

        im = np.stack((red, green, blue), axis = 2)
        im = np.dot(im, [0.299, 0.587, 0.114])

        height = im.shape[0]

        res = im[:height//2, :]
        
        return process_image(res)
        #return im
    
    def reset(self):
        self.car.reset()

        return self.get_state()

    def step(self, action, given_obs=np.zeros((80,80))):
        
        start_reward = self.reward()

        for i in range(1):

            #Here I've bound throttle to one to see if even that works
            
            #obs, reward, done = self.frame(action[0] / 10, max(0, action[1]) * 3, given_obs) 

            throttle = self.throttle_min + max(0, action[1]) * (self.throttle_max - self.throttle_min)

            obs, reward, done = self.frame(action[0] / 10, throttle, given_obs) 
            
            if not done:
                continue
            else:
                break
        
        #print(reward)
        self.string_image()

        return obs, reward, done, self.string_description(action)

    def string_image(self):

        pixels =  np.array(pygame.PixelArray(self.surface))

        pixels = pixels[::5, ::5]

        for i in range(pixels.shape[1]):
            print("".join(list(map(lambda x: chr(100 + x%50), pixels[:, i]))))


    def string_description(self, action):

        outer = distance(self.car.x, self.car.y, self.outer)
        inner = distance(self.car.x, self.car.y, self.inner)
        
        max_road_width = self.display_width / 1600 * 250
        width = 40

        left_lane = int(outer / max_road_width * width)
        right_lane = int(inner / max_road_width * width)
    
        left_wall = int(width / 2 - left_lane)
        right_wall = int(width / 2 - right_lane)

        car = "| |"

        if action[0] < -0.05:
            car = "\\ \\"
        elif action[0] > 0.05:
            car = "/ /"

        center_x = self.display_width / 2 
        center_y = self.display_height / 2

        rel_x = self.car.x - center_x
        rel_y = center_y - self.car.y

        progress = np.tan(rel_y / rel_x) / 2 / np.pi

        string = "#" * left_wall + "." * left_lane + car + "." * right_lane + "#" * right_wall + " Pos: {:.2f}".format(progress, 2)
        

        return string


    def reward(self):

        outer = distance(self.car.x, self.car.y, self.outer)
        inner = distance(self.car.x, self.car.y, self.inner)
        
        # if self.car.v < 0:
        #     reward = self.car.v
        # else:



        if self.reward_type == "speed":
            reward = self.car.v
        elif self.reward_type == "cte":
            reward = self.car.v * (1 - abs(outer - inner) / abs(outer + inner))

        #reward = self.car.v

        #print("Distance from outer: {}, Distance from inner: {}, Reward: {}".format(outer, inner, reward))
        
        return reward

    def frame(self, steering, acc, obs):

        # if self.manual_control:
        #     control(self.car, "manual")
        # else:
        self.car.accelerate(acc)
        self.car.turn(steering)
        self.car.move()

        done = False
        rwd = self.reward()


        if is_out(self.car.x, self.car.y, self.outer) or not is_out(self.car.x, self.car.y, self.inner):
            self.car.reset()
            rwd = 0
            done = True

        if self.graphics:
            self.draw(obs)
        
        
        obs = self.get_state()
        
        self.clock.tick(30)

        return obs, rwd, done, 


    def draw(self, obs):
        self.surface.fill((13, 156, 0))
        draw_track(self.surface, self.inner, self.outer)
        self.car.draw(self.surface)
        

        if self.vision == "simple":

            self.get_reduced_vision()

            for l in self.line_points:
                l1 = ((self.car.x, self.car.y), l)
                
                i = min_intersection(l1, self.outer, self.inner)
                pygame.draw.line(self.surface, (255,0,0), (self.car.x, self.car.y), i, 1)
                
                #print(i)
                pygame.draw.rect(self.surface, (255,0,0), pygame.Rect(i[0] - 2, i[1] - 2, 4, 4))

        surf = pygame.surfarray.make_surface(obs)
        self.surface.blit(surf, (self.display_width - 100,0))

        font = pygame.font.Font('freesansbold.ttf', 12) 
        text = font.render("Speed: {}, Acc: {}, Direction: {}, Reward: {}".format(round(self.car.v, 2), round(self.car.a, 2), round(self.car.dir, 2), round(self.reward(), 2)), True, (0,0,0))
        self.surface.blit(text, (10, 10))

        pygame.display.update()



def main():
    
    pygame.init()
    
    new_game = game()
    
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        new_game.step(np.random.rand() / 5 - 0.1, np.random.rand() / 5 - 0.1)

if __name__ == "__main__":
    
    main()

    # from PIL import Image

    # g = game()

    # acc = 0.2
    # direction = 0.2

    # running = True

    # obs = np.random.rand(80, 80)

    # counter = 0

    # while running:

    #     counter += 1

    #     cur_acc = 0
    #     cur_dir = 0
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #         elif event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_UP:
    #                 cur_acc += acc
    #             if event.key == pygame.K_DOWN:
    #                 cur_acc -= acc
    #             if event.key == pygame.K_RIGHT:
    #                 cur_dir += direction
    #             if event.key == pygame.K_LEFT:
    #                 cur_dir -= direction

            
    #     obs, _, _ = g.step(cur_dir, cur_acc, obs)

    #     im  = Image.fromarray(obs).convert("L")
    #     im.save('outfile_{}.png'.format(counter))
        

