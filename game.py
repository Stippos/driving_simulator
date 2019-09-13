import pygame
import pygame.gfxdraw
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull

class box:
    def __init__(self, x, y, h, w, color, vx, vy):
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

        self.corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])

    def draw(self, surface):
        
        draw_dir = -self.dir

        c = self.corners @ -np.array([[np.cos(draw_dir), -np.sin(draw_dir)], [np.sin(draw_dir), np.cos(draw_dir)]])
        
        corners = [[c[i,0] + self.x, c[i,1] + self.y] for i in range(4)]

        pygame.gfxdraw.filled_polygon(surface, corners, (255, 255, 0))
        pygame.draw.lines(surface, (0,0,0), True, corners, 3)
        
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

def main():
    
    pygame.init()

    display_width = 1600
    display_height = 1000

    acc = 0.2
    turning = 0.2

    surface = pygame.display.set_mode((display_width, display_height))

    pygame.display.set_caption('Lörs')

    new_box = box(230, 400, 40, 20, (0,0,0), 1, 0)

    clock = pygame.time.Clock()

    running = True
    
    outer, inner = get_track()

    while running:    
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    new_box.accelerate(acc)
                if event.key == pygame.K_DOWN:
                    new_box.accelerate(-acc)
                if event.key == pygame.K_RIGHT:
                    new_box.turn(turning)
                if event.key == pygame.K_LEFT:
                    new_box.turn(-turning)
                
        surface.fill((13, 156, 0))
        
        draw_track(surface, inner, outer)

        new_box.draw(surface)

        new_box.move()
        
        if is_out(new_box.x, new_box.y, outer):
            new_box.reset()

        if not is_out(new_box.x, new_box.y, inner):
            new_box.reset()

        new_box.x = new_box.x % display_width
        new_box.y = new_box.y % display_height

        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()
