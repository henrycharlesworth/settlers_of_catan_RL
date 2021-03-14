import os
import sys

import pygame
# from pygame.locals import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ui.sftext.sftext import SFText


text = """
Hello there I am awesome.

{style}{color (255, 0, 0)}kajfjsjkjgskjsjlk\ngfjklsjsdjgsjgkfkd'kkdf
fdsfdabad
fdsfabgdbgagb
dfbdbdabalkldfknlkbndbalkbkl
sbfdsmbladfkbndklabnbdknlbnlkdanb\n
sndfml lfdkabnfdalkbdklabnl kgan klnkldf nklnkbnkd fnkbnkfdn kbdfkbn kdfbnkdf nkbn dfknkb
 
skdnvklsdnvsklvfnkbkd lfbnkldnakbnknklbnf kdbnkdfnklbndk flbndn kbnd fkn bkdnknbk

sd lsld fk nf lbf nbk ldwiu wqyr738 893y9yh9 hwrh9f sjnfqj ndndin
"""

pygame.init()

SCREEN = pygame.display.set_mode((1024, 768))
BACKGROUND_COLOUR = (0, 255, 0)


min_x = 200
max_x = 800
min_y = 200
max_y = 500

target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
shape_surf = pygame.Surface(target_rect.size)

sftext = SFText(text=text, surface=shape_surf, font_path=os.path.join('.', 'resources'))

clock = pygame.time.Clock()

counter = 0

alive = True
while alive:  # main game loop
    SCREEN.fill(BACKGROUND_COLOUR)
    counter += 1
    if counter == 25:
        sftext.text = "new line hi\n" + sftext.text
        sftext.parse_text()
        counter = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            alive = False
        else:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    alive = False
                else:
                    sftext.on_key_press(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button <= 3:
                    # self.on_mouse_press(event)
                    pass
                else:
                    sftext.on_mouse_scroll(event)

    shape_surf.fill((255, 255, 255))
    sftext.on_update()
    SCREEN.blit(shape_surf, target_rect)
    pygame.display.update()
    sftext.post_update()

    clock.tick(60)