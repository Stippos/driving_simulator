import pygame
import pygame.gfxdraw
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.ndimage import rotate

from scipy.misc import toimage


import matplotlib.pyplot as plt

class box:
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

        if abs(direction) > self.turning_limit:
            self.dir += np.sign(direction) * self.turning_limit
        else:
            self.dir += direction
    
    def accelerate(self, amount):
        self.a = amount

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.v = 0
        self.dir = 0
        self.a = 0

        vc = self.vision_limits

        self.vision_corners = [[vc[i,0] + self.x, vc[i,1] + self.y] for i in range(4)]


def get_track(w, h):
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
    
def is_out(x, y, points):
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

class game:

    def __init__(self, draw = True, manual_control = False):

        pygame.init()

        self.display_width = 1600
        self.display_height = 1000
        self.surface = pygame.display.set_mode((self.display_width, self.display_height))

        car_x = 230 / 1600 * self.display_width
        car_y = 400 / 1000 * self.display_height
        # 

        # car_x = 1420 / 1600 * self.display_width
        # car_y = 800 / 1000 * self.display_height

        self.car = box(car_x, car_y, 40, 20, (0,0,0), 600, 300)

        self.clock = pygame.time.Clock()

        self.running = True

        self.outer, self.inner = get_track(self.display_width, self.display_height)

        self.graphics = draw

        self.manual_control = manual_control

    def get_vision(self):

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

        return im[:height//2, :]
        #return im

    def reset(self):
        self.car.reset()

        return self.get_vision()

    def step(self, steering, acc, given_obs):
        
        start_reward = self.reward()

        for i in range(2):
            obs, reward, done = self.frame(steering, acc, given_obs) 
            if not done:
                continue
            else:
                break
        
        #print(reward)

        return obs, reward, done

    def reward(self):

        outer = distance(self.car.x, self.car.y, self.outer)
        inner = distance(self.car.x, self.car.y, self.inner)
        
        # if self.car.v < 0:
        #     reward = self.car.v
        # else:
        #     reward = (1 - abs(outer - inner) / abs(outer + inner)) * self.car.v
        
        reward = self.car.v

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
        
        
        obs = self.get_vision()
        
        self.clock.tick(120)

        return obs, rwd, done, 


    def draw(self, obs):
        self.surface.fill((13, 156, 0))
        draw_track(self.surface, self.inner, self.outer)
        self.car.draw(self.surface)

        surf = pygame.surfarray.make_surface(obs)
        self.surface.blit(surf, (1500,0))

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
        

