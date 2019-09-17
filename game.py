import pygame
import pygame.gfxdraw
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt

class box:
    def __init__(self, x, y, h, w, color, vh, vw):
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

        self.corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        self.vision_limits = np.array([[-vw/2, -vh/2], [vw/2, -vh/2], [vw/2, vh/2], [-vw/2, vh/2]])
        self.vision_corners = self.vision_limits

    def draw(self, surface):
        
        draw_dir = -self.dir

        c = self.corners @ -np.array([[np.cos(draw_dir), -np.sin(draw_dir)], [np.sin(draw_dir), np.cos(draw_dir)]])
        
        corners = [[c[i,0] + self.x, c[i,1] + self.y] for i in range(4)]
        
        vc = self.vision_limits

        self.vision_corners = [[vc[i,0] + self.x, vc[i,1] + self.y] for i in range(4)]
         
        pygame.gfxdraw.filled_polygon(surface, corners, (255, 255, 0))
        pygame.draw.lines(surface, (0,0,0), True, corners, 3)
        pygame.draw.lines(surface, (0,0,0), True, self.vision_corners , 3)
        
        font = pygame.font.Font('freesansbold.ttf', 12) 
        text = font.render("Speed: {}, Acc: {}, Direction: {}".format(round(self.v, 2), round(self.a, 2), round(self.dir, 2)), True, (0,0,0))
        surface.blit(text, (10, 10))

        
    def move(self):
        self.v += self.a

        self.x += np.sin(self.dir) * self.v 
        self.y -= np.cos(self.dir) * self.v 

        self.v *= 0.8
        
    def turn(self, direction):
        self.dir += direction
    
    def accelerate(self, amount):
        self.a += amount

    def reset(self):
        self.x = 230
        self.y = 400
        self.v = 0
        self.dir = 0
        self.a = 0


def get_track():
    track_points = pd.read_csv("rata.csv", header = None)

    outer_track = track_points.iloc[:,0:2]
    inner_track = track_points.iloc[:24,2:4]

    outer_track_points = []
    inner_track_points = []


    for i in range(len(outer_track)):
        outer_track_points.append((outer_track.iloc[i, 0], outer_track.iloc[i, 1]))

    for i in range(len(inner_track)):
        inner_track_points.append((inner_track.iloc[i, 0], inner_track.iloc[i, 1]))

    return outer_track_points, inner_track_points

def draw_track(surface, inner, outer):

    pygame.gfxdraw.filled_polygon(surface, outer, (120, 120, 120))
    pygame.gfxdraw.filled_polygon(surface, inner, (13, 156, 0))

    pygame.draw.lines(surface, (255, 255, 0), True, outer, 5)
    pygame.draw.lines(surface, (255, 255, 0), True, inner, 5)

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

def get_vision(surface, box):
    
    global image

    pa = np.array(pygame.PixelArray(surface))

    l = box.vision_corners

    left = int(l[0][0])
    right = int(l[1][0])
    top = int(l[0][1])
    bottom = int(l[3][1])  

    vision = pa[left:right, top:bottom].T

    if not image:
        image = plt.imshow(vision)
    else:
        image.set_data(vision)
    plt.show(block = False)

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


class game:

    def __init__(self, draw = True):
        self.display_width = 1600
        self.display_height = 1000
        self.surface = pygame.display.set_mode((self.display_width, self.display_height))

        self.car = box(230, 400, 40, 20, (0,0,0), 300, 300)

        self.clock = pygame.time.Clock()

        self.running = True

        self.outer, self.inner = get_track()

        self.draw = draw

    def get_vision(self):
        pa = np.array(pygame.PixelArray(self.surface))

        l = self.car.vision_corners

        left = int(l[0][0])
        right = int(l[1][0])
        top = int(l[0][1])
        bottom = int(l[3][1])  

        return pa[left:right, top:bottom].T

    def reset(self):
        self.car.reset()

        return self.get_vision()


    def step(self, steering, acc):

        self.car.accelerate(acc)
        self.car.turn(steering)
        self.car.move()

        reward = self.car.v
        obs = self.get_vision()
        done = False

        if is_out(self.car.x, self.car.y, self.outer) or not is_out(self.car.x, self.car.y, self.inner):
            self.car.reset()
            reward = -1
            done = True

        self.draw()
        
        self.clock.tick(30)

        return obs, reward, done, 


    def draw(self):
        self.surface.fill((13, 156, 0))
        draw_track(self.surface, self.inner, self.outer)
        self.car.draw(self.surface)

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
    plt.show()
