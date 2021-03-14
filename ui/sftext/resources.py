# ---------------------------------------------------------
# sftext - Scrollable Formatted Text for pygame
# Copyright (c) 2016 Lucas de Morais Siqueira
# Distributed under the GNU Lesser General Public License version 3.
#
#       \ vvvvvvvvvvvvvvvvv /
#     >>> RESOURCES MANAGER <<<
#       / ^^^^^^^^^^^^^^^^^ \
#               This module is based on LazyImageLoading
#               http://pygame.org/wiki/LazyImageLoading
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

import sys
import os

import pygame
import weakref


class Resources(object):

    _names = {}

    @classmethod
    def __init__(cls, loader, path, types, weak_ref=True):
        cls._index(path, types)
        if weakref:
            cls.cache = weakref.WeakValueDictionary()
        else:
            cls.cache = {}
        cls.loader = loader

    @classmethod
    def __getattr__(cls, name):
        try:
            img = cls.cache[name]
        except KeyError:
            img = cls.loader(cls._names[name])
            cls.cache[name] = img
        return img

    @classmethod
    def load(cls, name):
        return cls.__getattr__(name)

    @classmethod
    def _index(cls, path, types):
        if sys.version_info >= (3, 5):
            # Python version >=3.5 supports glob
            import glob
            for img_type in types:
                for filename in glob.iglob(
                    (path + '/**/' + img_type), recursive=True
                ):
                    f_base = os.path.basename(filename)
                    cls._names.update({f_base: filename})
        else:
            # Python version <=3.4
            import fnmatch

            for root, dirnames, filenames in os.walk(path):
                for img_type in types:
                    for f_base in fnmatch.filter(filenames, img_type):
                        filename = os.path.join(root, f_base)
                        cls._names.update({f_base: filename})


class Images(Resources):
    @classmethod
    def __init__(cls, path=".", types=['*.jpg', '*.png', '*.bmp']):
        super().__init__(
            loader=pygame.image.load,
            path=path,
            types=types)


class Fonts(Resources):
    @classmethod
    def __init__(cls, path=".", types=['*.ttf']):
        super().__init__(
            loader=pygame.font.Font,
            path=path,
            types=types,
            weak_ref=False)

    @classmethod
    def __getattr__(cls, name, size):
        try:
            font = cls.cache[name, size]
        except KeyError:
            font = cls.loader(cls._names[name], size)
            cls.cache[name, size] = font
        return font

    @classmethod
    def load(cls, name, size):
        return cls.__getattr__(name, size)
