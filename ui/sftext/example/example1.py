# ---------------------------------------------------------
# sftext - Scrollable Formatted Text for pygame
# Copyright (c) 2016 Lucas de Morais Siqueira
# Distributed under the GNU Lesser General Public License version 3.
#
#     Support by using, forking, reporting issues and giving feedback:
#     https://https://github.com/LukeMS/sftext/
#
#     Lucas de Morais Siqueira (aka LukeMS)
#     lucas.morais.siqueira@gmail.com
#
#    This file is part of sftext.
#
#    sftext is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    sftext is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with sftext. If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------

import os
import sys

import pygame
from pygame.locals import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ui.sftext.sftext import SFText

import ui.sftext.example.lorem_ipsum as lorem_ipsum


if __name__ == '__main__':
    pygame.init()

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    DISPLAYSURF = pygame.display.set_mode((1024, 768), pygame.NOFRAME)
    pygame.display.set_caption('Scrollable Formatted Text')

    mytext = lorem_ipsum.text

    sftext = SFText(text=mytext, font_path=os.path.join('.', 'resources'))

    clock = pygame.time.Clock()

    alive = True
    while alive:  # main game loop
        for event in pygame.event.get():
            if event.type == QUIT:
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

        sftext.on_update()
        pygame.display.flip()
        sftext.post_update()

        clock.tick(60)
