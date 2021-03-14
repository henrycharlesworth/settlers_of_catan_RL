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
from sftext import SFText

text = """{align center}{size 40}Lorem ipsum{style}\n\n
        {size 10}Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce quis tempor enim, ac pretium diam. In urna lectus, condimentum eget convallis in, hendrerit ut nibh. Nullam tristique elementum sem. Suspendisse volutpat, lacus id eleifend pellentesque, quam risus scelerisque erat, non egestas massa nisi quis est. Morbi viverra elementum nunc, nec blandit leo pretium id. Duis bibendum posuere augue. Mauris risus ex, venenatis non sem euismod, auctor bibendum justo.{style}

        {style}{size 15}Morbi orci leo, scelerisque a arcu ac, viverra eleifend risus. Nulla ultrices lorem ac rutrum tristique. Etiam sed posuere enim. Nullam sed sollicitudin odio. Mauris a semper ante. Duis nec mauris ipsum. Pellentesque euismod iaculis felis a venenatis. Maecenas tincidunt, erat non pretium eleifend, massa eros aliquam felis, eu dapibus tortor mauris eget tellus. Mauris dapibus fermentum enim nec lacinia. Nam sed velit lacinia, interdum massa sit amet, congue nibh.{style}

        Cras in nisi facilisis, consequat sem sit amet, aliquet ligula. Fusce cursus ante pharetra, mollis nibh ut, mollis ipsum. Proin hendrerit ipsum sit amet purus molestie, eu semper orci euismod. Nulla in faucibus ex. Mauris egestas iaculis ullamcorper. Maecenas sed nisi vitae ipsum feugiat lobortis. Proin lobortis diam ac ligula aliquam elementum. Nullam est diam, gravida id neque id, faucibus eleifend nibh. Integer maximus euismod tellus, a lobortis tellus congue in. Vivamus maximus nulla a sem ornare, nec sollicitudin quam suscipit. Donec quis porta metus. In hac habitasse platea dictumst.

        {style}{size 25}Suspendisse lacus nisl, pretium vel leo non, aliquam aliquam elit. Aenean et facilisis sapien. Donec congue vel augue vel porta. Donec purus mauris, congue sit amet justo eu, fringilla scelerisque odio. Cras commodo ultricies est. Fusce in pulvinar elit. Aliquam vestibulum efficitur turpis, et tempus massa efficitur mollis. Nulla vel egestas ex. Phasellus placerat egestas imperdiet. Sed lectus massa, tempor at imperdiet ac, blandit at lorem. Pellentesque lacinia dui vel eleifend imperdiet. Morbi faucibus, odio non pulvinar cursus, enim felis rutrum urna, id tempor nibh felis ut nulla. Vestibulum dui dui, rhoncus eget gravida nec, rutrum quis magna. Suspendisse faucibus tortor vel congue vestibulum.
"""


if __name__ == '__main__':
    pygame.init()

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    DISPLAYSURF = pygame.display.set_mode((1024, 768), pygame.NOFRAME)
    pygame.display.set_caption('Scrollable Formatted Text')

    # You can pass in only what you want to change on the default style.
    mystyle = {
        'color': (255, 255, 255)
    }
    sftext = SFText(text=text, font_path='resources', style=mystyle)

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
