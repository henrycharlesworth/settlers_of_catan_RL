# ---------------------------------------------------------
# sftext - Scrollable Formatted Text for pygame
# Copyright (c) 2016 Lucas de Morais Siqueira
# Distributed under the GNU Lesser General Public License version 3.
#
#       \ vvvvvvvvvvvvvvvvvvvvvvvvv /
#     >>> SCROLLABLE FORMATTED TEXT <<<
#       / ^^^^^^^^^^^^^^^^^^^^^^^^^ \
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

import pygame

from pygame.locals import *

import ui.sftext.resources as resources
from ui.sftext.style.style import Style


class SFText():
    def __init__(self, text, surface=None, font_path='.', style=None):

        if isinstance(text, bytes):
            # print('text is', bytes)
            self.text = text.decode('utf-8')
        elif isinstance(text, str):
            # print('text is', str)
            self.text = text

        self.fonts = resources.Fonts(path=font_path)

        if style:
            Style.set_default(style)

        if surface is None:
            self.screen = pygame.display.get_surface()
        else:
            self.screen = surface
        self.screen_rect = self.screen.get_rect()
        self.bg = self.screen.copy()

        # print('parsing text')
        self.parse_text()
        # print('done parsing')

    def set_font(self, obj):
        if obj['bold'] and obj['italic'] and obj['separate_bolditalic']:
            obj['font_obj'] = self.fonts.load(
                obj['separate_bolditalic'], obj['size'])
        elif obj['separate_bold'] and obj['bold']:
            obj['font_obj'] = self.fonts.load(
                obj['separate_bold'], obj['size'])
        elif obj['separate_italic'] and obj['italic']:
            obj['font_obj'] = self.fonts.load(
                obj['separate_italic'], obj['size'])
        else:
            obj['font_obj'] = self.fonts.load(
                obj['font'], obj['size'])

    def parse_text(self):
        self.parsed = []
        scr_w = self.screen_rect.width

        self.default_style = Style.default_style
        self.default_style['font_obj'] = self.fonts.load(
            self.default_style['font'], self.default_style['size'])
        self.default_style['w'], self.default_style['h'] = (
            self.default_style['font_obj'].size(' '))

        y = 0
        for line in self.text.splitlines():
            x = 0
            for style in line.split("{style}"):

                text, styled_txt = Style.split(style)

                self.set_font(styled_txt)
                font = styled_txt['font_obj']

                w, h = styled_txt['w'], styled_txt['h'] = font.size(' ')
                # determine the amount of space needed to render text

                wraps = self.wrap_text(text, scr_w, x, styled_txt)

                for wrap in wraps:
                    rect = pygame.Rect((0, 0), font.size(wrap['text']))

                    if (x + wrap['w1'] + w * 3) > scr_w:
                        x = 0
                        y += wrap['h']

                    if len(wraps) == 1 and wrap['align'] == 'center':
                        rect.midtop = (
                            self.screen_rect.centerx,
                            self.screen_rect.bottom + y)
                    else:
                        rect.topleft = (
                            x + w * 3,
                            self.screen_rect.bottom + y)
                    wrap['rect'] = rect
                    wrap['x'] = x
                    wrap['y'] = y
                    if False:
                        print("\n{}: {},".format('x', wrap['x']), end='')
                        print("{}: {},".format('y', wrap['y']), end='')
                        print("{}: {},".format('w', wrap['w']), end='')
                        print("{}: {}".format('h', wrap['h']))
                        print(wrap['text'])
                    self.parsed.append(wrap)

                    x += wrap['w1']
            y += wrap['h']
        # exit()
        # print('done parsing')

        self.start_y = 0 - self.screen_rect.h + self.default_style['h']

        self.y = int(self.start_y)

        self.end_y = (
            -sum(p['h'] for p in self.parsed if p['x'] == 0)
            - self.default_style['h'] * 2)

    def wrap_text(self, text, width, _x, styled_txt):
        style = dict(styled_txt)
        x = int(_x)
        wrapped = []
        size = style['font_obj'].size
        c_width = style['w']

        # print(size(text))
        # print(width)
        if size(text)[0] <= (
            width - c_width * 6 - x
        ):
            # print('fits')
            style['text'] = text
            style['w1'] = size(text)[0]
            wrapped.append(style)

            return wrapped
        else:
            # print("doesn't fit")
            # print(text)
            wrapped = [text]
            guessed_length = ((width - c_width * 6 - x) // c_width)
            all_fit = False
            all_fit_iter = 1
            while not all_fit:
                #########
                # DEBUG #
                if False:
                    print("all_fit iteraions: {}".format(all_fit_iter))
                    if all_fit_iter >= guessed_length * 5:
                        exit()
                # DEBUG #
                #########
                for i in range(len(wrapped)):
                    # print('for i in range(len(wrapped))')
                    fit = size(wrapped[i])[0] < width - c_width * 6 - x
                    # print(width - c_width * 6 - x)
                    iter_length = int(guessed_length)
                    # print(iter_length)
                    fit_iter = 0
                    while not fit:
                        #########
                        # DEBUG #
                        if False:
                            fit_iter += 1
                            print("fit iteraions: {}, iter_length: {}".format(
                                fit_iter, iter_length))
                            if fit_iter >= guessed_length * 5:
                                print(wrapped[i])
                                exit()
                        # DEBUG #
                        #########
                        if guessed_length <= 2 or iter_length <= 2:
                            # print('if guessed_length <= 2')
                            x = 0
                            guessed_length = (
                                (width - c_width * 6 - x) // c_width)
                            iter_length = int(guessed_length)
                            continue
                        guess = wrapped[i][:iter_length]
                        # print('while not fit: "{}"'.format(guess))
                        if guess[-1:] not in [" ", ",", ".", "-", "\n"]:
                            # print('if guess[-1:] not in:')
                            iter_length -= 1
                        else:
                            if size(guess)[0] < width - c_width * 6 - x:
                                remains = wrapped[i][iter_length:]
                                wrapped[i] = guess
                                wrapped.append(remains)
                                fit = True
                            else:
                                iter_length -= 1
                    all_fit_iter += 1
                    # print("Cut point: {}".format(iter_length))
                    # print('Guess: ({})"{}"'.format(type(guess), guess))
                    # print('Remains: "{}"'.format(remains))
                    # print("[{}]fit? {}".format(i, fit))
                status = True
                for i in range(len(wrapped)):
                    if size(wrapped[i])[0] >= width:
                        status = False
                all_fit = status

            for i in range(len(wrapped)):
                # print('"{}"'.format(wrapped[i]))
                style['text'] = wrapped[i]
                style['w1'] = size(wrapped[i])[0]
                wrapped[i] = dict(style)

            return wrapped

    def on_update(self):
        for i, p in enumerate(self.parsed[:]):
            rect = p['rect'].move(0, self.y)

            if not isinstance(p['text'], pygame.Surface):
                p['font_obj'].set_bold(False)
                p['font_obj'].set_italic(False)

                if p['bold'] and p['italic'] and not p['separate_bolditalic']:
                    print('pygame-bold', p['text'])
                    p['font_obj'].set_bold(p['bold'])
                    print('pygame-italic', p['text'])
                    p['font_obj'].set_italic(p['italic'])
                elif not p['separate_bold'] and p['bold']:
                    print('pygame-bold', p['text'])
                    p['font_obj'].set_bold(p['bold'])
                elif not p['separate_italic'] and p['italic']:
                    print('pygame-italic', p['text'])
                    p['font_obj'].set_italic(p['italic'])

                p['font_obj'].set_underline(p['underline'])

                p['text'] = p['font_obj'].render(p['text'], 1, p['color'])
            self.screen.blit(p['text'], rect)

            if rect.top >= (
                self.screen_rect.bottom - self.default_style['h']
            ):
                break
        """
        if obj['bold'] and obj['italic'] and obj['separate_bolditalic']:
            obj['font_obj'] = self.fonts.load(
                obj['separate_bold_italic'], obj['size'])
        elif obj['separate_bold'] and obj['bold']:
            obj['font_obj'] = self.fonts.load(
                obj['separate_bold'], obj['size'])
        elif obj['separate_italic'] and obj['italic']:
            obj['font_obj'] = self.fonts.load(
                obj['separate_italic'], obj['size'])
        else:
            obj['font_obj'] = self.fonts.load(
                obj['font'], obj['size'])
        """

    def scroll(self, y=0):
        if isinstance(y, int):
            self.y += y
            if self.y < self.end_y:
                self.y = self.end_y
            elif self.y > self.start_y:
                self.y = self.start_y
        elif isinstance(y, str):
            if y == 'home':
                self.y = self.start_y
            elif y == 'end':
                self.y = self.end_y

    def post_update(self):
        self.screen.blit(self.bg, (0, 0))

    def on_key_press(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            self.scroll(50)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_PAGEUP:
            self.scroll(500)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_HOME:
            self.scroll('home')

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
            self.scroll(-50)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_PAGEDOWN:
            self.scroll(-500)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_END:
            self.scroll('end')

    def on_mouse_scroll(self, event):
        if event.button == 4:
            self.scroll(50)
        elif event.button == 5:
            self.scroll(-50)
