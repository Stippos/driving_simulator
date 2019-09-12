import pygame

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

        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, [self.x, self.y, self.h, self.w])

    def move(self):
        self.vx += self.ax
        self.vy += self.ay
        self.x += self.vx
        self.y += self.vy

        self.vx *= 0.8
        self.vy *= 0.8


def main():
    
    pygame.init()

    display_width = 800
    display_height = 600

    black = (0,0,0)
    white = (255,255,255)
    
    surface = pygame.display.set_mode((800, 600))

    pygame.display.set_caption('Lörs')

    new_box = box(100, 100, 10, 10, black, 1, 0)

    clock = pygame.time.Clock()

    running = True
    
    while running:    
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    new_box.ay -= 1
                if event.key == pygame.K_DOWN:
                    new_box.ay += 1
                if event.key == pygame.K_RIGHT:
                    new_box.ax += 1
                if event.key == pygame.K_LEFT:
                    new_box.ax -= 1

        surface.fill(white)
        
        new_box.draw(surface)

        new_box.move()
        
        new_box.x = new_box.x % display_width
        new_box.y = new_box.y % display_height

        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()
