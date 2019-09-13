import pygame
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

        self.dir = np.pi

        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, [self.x, self.y, self.h, self.w])

        line_start = np.array([self.x + self.w / 2, self.y + self.h / 2])

        draw_dir = self.dir - np.pi / 2

        line_end = line_start + np.array([np.cos(draw_dir), np.sin(draw_dir)]) * 15
        pygame.draw.line(surface, self.color, line_start, line_end, 5)

    def move(self):
        self.v += self.a

        self.x += np.sin(self.dir) * self.v 
        self.y -= np.cos(self.dir) * self.v 

        self.v *= 0.8
        
    def turn(self, direction):
        self.dir += direction
    
    def accelerate(self, amount):
        self.a += amount

def get_track():
    track_points = pd.read_csv("rata.csv", header = None)

    outer_track = track_points.iloc[:,0:2]
    inner_track = track_points.iloc[:24,2:4]

    print(outer_track)
    
    outer_track_points = []
    inner_track_points = []


    for i in range(len(outer_track)):
        outer_track_points.append((outer_track.iloc[i, 0], outer_track.iloc[i, 1]))

    for i in range(len(inner_track)):
        inner_track_points.append((inner_track.iloc[i, 0], inner_track.iloc[i, 1]))


    return outer_track_points, inner_track_points

def draw_track(surface, inner, outer):

    pygame.draw.lines(surface, (255, 0, 0), True, outer)
    pygame.draw.lines(surface, (255, 0, 0), True, inner)
    

def generate_track(n_points, w, h):

    points = np.random.rand(n_points, 2)
    
    points[:,0] = (points[:,0] * (w - 100)) + 50
    points[:,1] = (points[:,1] * (h - 100)) + 50

    hull = ConvexHull(points)

    arcs = []
    points_list = []

    for s in hull.simplices:
        midpoint = [points[s, 0].mean() + (np.random.rand() * w) / 8, points[s, 1].mean() + (np.random.rand() * h) / 8]
        arc1_s = [points[s, 0][0], midpoint[0]]
        arc1_e = [points[s, 1][0], midpoint[1]]
        arc2_s = [points[s, 0][1], midpoint[0]]
        arc2_e = [points[s, 1][1], midpoint[1]]

        arc1_s = [points[s, 0][0], points[s, 1][0]]
        arc1_e = [midpoint[0], midpoint[1]]
        arc2_s = [midpoint[0], midpoint[1]]
        arc2_e = [points[s, 0][1], points[s, 1][1]]

        arcs.append([arc1_s, arc1_e])
        arcs.append([arc2_s, arc2_e])

        points_list.append(arc1_s)
        points_list.append(midpoint)
        points_list.append(arc2_e)

    return arcs

def draw_track_arcs(surface, arcs):

    for a in arcs:
        pygame.draw.line(surface, (255, 0, 0), a[0], a[1], 20)


def main():
    
    pygame.init()

    display_width = 1600
    display_height = 1000

    acc = 0.2
    turning = 0.2

    black = (0,0,0)
    white = (255,255,255)
    
    surface = pygame.display.set_mode((display_width, display_height))

    pygame.display.set_caption('Lörs')

    new_box = box(100, 100, 10, 10, black, 1, 0)

    clock = pygame.time.Clock()

    running = True
    
    outer, inner = get_track()

    track = generate_track(6, display_width, display_height)

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

        surface.fill(white)
        
        #draw_track_arcs(surface, track)
        #pygame.draw.lines(surface, (255, 0, 0), True, points)
        
        draw_track(surface, outer, inner)

        new_box.draw(surface)

        new_box.move()
        
        new_box.x = new_box.x % display_width
        new_box.y = new_box.y % display_height

        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()
