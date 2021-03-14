# ---------------------------------------------------------
# sftext - Scrollable Formatted Text for pygame
# Copyright (c) 2016 Lucas de Morais Siqueira
# Distributed under the GNU Lesser General Public License version 3.
#
#       \ vvvvvvvvvvvvv /
#     >>> STYLE MANAGER <<<
#       / ^^^^^^^^^^^^^ \
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

import re

DEFAULT_STYLE = {
    # At the momment only font filenames are supported. That means the font
    # must be in the same directory as the main script.
    # Or you could (should?) use a resource manager such as
    'font': 'caladea-regular.ttf',
    'size': 20,
    'indent': 0,
    'bold': False,
    'italic': False,
    'underline': False,
    'color': (128, 144, 160),  # RGB values
    'align': 'left',
    # if a separate file should be used for italic/bold, speciy it;
    # if not, use None
    'separate_italic': 'caladea-italic.ttf',
    'separate_bold': 'caladea-bold.ttf',
    'separate_bolditalic': 'caladea-bolditalic.ttf',
}


class Style:
    default_style = DEFAULT_STYLE

    @classmethod
    def set_default(cls, source):
        if isinstance(source, str):
            cls.string = str(source)
            cls._get_style()
            cls.default_style = dict(cls.style)
        elif isinstance(source, dict):
            for key, value in source.items():
                cls.default_style[key] = value
        return cls.default_style

    @classmethod
    def get_default(cls, string):
        return cls.default_style

    @classmethod
    def stylize(cls, string, style=None):
        if style is None:
            style = cls.default_style
        stylized = ""

        for key, value in style.items():
            if isinstance(value, str):
                stylized += ("{" + "{} '{}".format(key, value) + "'}")
            else:
                stylized += ("{" + "{} {}".format(key, value) + "}")
        stylized += string
        return stylized

    @classmethod
    def split(cls, string):
        cls.string = str(string)
        cls._get_style()
        return cls.string, cls.style

    @classmethod
    def remove(cls, string):
        cls.string = str(string)
        cls._get_style()
        return cls.string

    @classmethod
    def get(cls, string):
        cls.string = str(string)
        cls._get_style()
        return cls.style

    @classmethod
    def _get_style(cls):
        cls.style = {}
        cls.style['font'] = cls._get_font()
        cls.style['size'] = cls._get_size()
        cls.style['bold'] = cls._get_font_bold()
        cls.style['italic'] = cls._get_font_italic()
        cls.style['underline'] = cls._get_font_underline()
        cls.style['color'] = cls._get_font_color()
        cls.style['align'] = cls._get_font_align()
        cls.style['indent'] = cls._get_font_indent()
        cls.style['separate_bold'] = cls._get_separate_bold()
        cls.style['separate_italic'] = cls._get_separate_italic()
        cls.style['separate_bolditalic'] = cls._get_separate_bolditalic()

    @classmethod
    def _get_font(cls):
        pattern = (
            "{font(_name)? ('|\")(?P<font>[A-Za-z0-9_ -]+"
            "(?P<ext>.ttf))('|\")}")
        searchgroup = re.search(pattern, cls.string)
        if searchgroup:
            if searchgroup.group('ext'):
                font = searchgroup.group('font')
            else:
                font = searchgroup.group('font') + ".ttf"
                print(font)
        else:
            font = cls.default_style['font']
        cls.string = re.sub(
            (
                "({font(_name)? ('|\")([A-Za-z0-9_ -]+"
                "(.ttf)?)('|\")})"),
            '',
            cls.string)
        return font

    @classmethod
    def _get_separate_italic(cls):
        pattern = (
            "{separate_italic ('|\")(?P<separate_italic>[A-Za-z0-9_ -]+"
            "(?P<ext>.ttf))('|\")}")
        searchgroup = re.search(pattern, cls.string)
        if searchgroup:
            if searchgroup.group('ext'):
                separate_italic = searchgroup.group('separate_italic')
            else:
                separate_italic = searchgroup.group('separate_italic') + ".ttf"
                print(separate_italic)
        else:
            if cls.style['font'] == cls.default_style['font']:
                separate_italic = cls.default_style['separate_italic']
            else:
                separate_italic = None
        cls.string = re.sub(
            (
                "({separate_italic ('|\")([A-Za-z0-9_ -]+"
                "(.ttf)?)('|\")})"),
            '',
            cls.string)
        return separate_italic

    @classmethod
    def _get_separate_bold(cls):
        pattern = (
            "{separate_bold ('|\")(?P<separate_bold>[A-Za-z0-9_ -]+"
            "(?P<ext>.ttf))('|\")}")
        searchgroup = re.search(pattern, cls.string)
        if searchgroup:
            if searchgroup.group('ext'):
                separate_bold = searchgroup.group('separate_bold')
            else:
                separate_bold = searchgroup.group('separate_bold') + ".ttf"
                print(separate_bold)
        else:
            if cls.style['font'] == cls.default_style['font']:
                separate_bold = cls.default_style['separate_bold']
            else:
                separate_bold = None
        cls.string = re.sub(
            (
                "({separate_bold ('|\")([A-Za-z0-9_ -]+"
                "(.ttf)?)('|\")})"),
            '',
            cls.string)
        return separate_bold

    @classmethod
    def _get_separate_bolditalic(cls):
        pattern = (
            "{separate_bolditalic ('|\")"
            "(?P<separate_bolditalic>[A-Za-z0-9_ -]+"
            "(?P<ext>.ttf))('|\")}")
        searchgroup = re.search(pattern, cls.string)
        if searchgroup:
            if searchgroup.group('ext'):
                separate_bolditalic = searchgroup.group('separate_bolditalic')
            else:
                separate_bolditalic = searchgroup.group(
                    'separate_bolditalic') + ".ttf"
                print(separate_bolditalic)
        else:
            if cls.style['font'] == cls.default_style['font']:
                separate_bolditalic = cls.default_style['separate_bolditalic']
            else:
                separate_bolditalic = None
        cls.string = re.sub(
            (
                "({separate_bold ('|\")([A-Za-z0-9_ -]+"
                "(.ttf)?)('|\")})"),
            '',
            cls.string)
        return separate_bolditalic

    @classmethod
    def _get_size(cls):
        pattern = "{(font_)?size (?P<size>\d+)}"
        searchgroup = re.search(pattern, cls.string)
        if searchgroup:
            size = searchgroup.group('size')
        else:
            size = cls.default_style['size']
        cls.string = re.sub(pattern, '', cls.string)
        return int(size)

    @classmethod
    def _get_font_bold(cls):
        pattern = "{bold ('|\"|)(?P<bold>True|False)('|\"|)}"
        searchgroup = re.search(pattern, cls.string, re.I)
        if searchgroup:
            bold = searchgroup.group('bold')
            cls.string = re.sub(pattern, '', cls.string)
            return bold.lower() == "true"
        else:
            bold = cls.default_style['bold']
            cls.string = re.sub(pattern, '', cls.string)
            return bold

    @classmethod
    def _get_font_italic(cls):
        pattern = "{italic ('|\"|)(?P<italic>True|False)('|\"|)}"
        searchgroup = re.search(pattern, cls.string, re.I)
        if searchgroup:
            italic = searchgroup.group('italic')
            cls.string = re.sub(pattern, '', cls.string)
            return italic.lower() == "true"
        else:
            italic = cls.default_style['italic']
            cls.string = re.sub(pattern, '', cls.string)
            return italic

    @classmethod
    def _get_font_underline(cls):
        pattern = "{underline ('|\"|)(?P<underline>True|False)('|\"|)}"
        searchgroup = re.search(pattern, cls.string, re.I)
        if searchgroup:
            underline = searchgroup.group('underline')
            cls.string = re.sub(pattern, '', cls.string)
            return underline.lower() == "true"
        else:
            underline = cls.default_style['underline']
            cls.string = re.sub(pattern, '', cls.string)
            return underline

    @classmethod
    def _get_font_color(cls):
        pattern = "{color \((?P<color>\d+\, *\d+\, *\d+)(?P<alpha>\, *\d+)?\)}"
        searchgroup = re.search(pattern, cls.string)
        if searchgroup:
            color = searchgroup.group('color')
            color = tuple(int(c) for c in color.split(","))
        else:
            color = cls.default_style['color']
        cls.string = re.sub(pattern, '', cls.string)
        return color

    @classmethod
    def _get_font_align(cls):
        pattern = "{(.)?align ('|\"|)(?P<align>(left|center|right))('|\"|)}"
        searchgroup = re.search(pattern, cls.string)
        if searchgroup:
            align = searchgroup.group('align')
        else:
            align = cls.default_style['align']
        cls.string = re.sub(pattern, '', cls.string)
        return align

    @classmethod
    def _get_font_indent(cls):
        pattern = "{indent (?P<indent>\d+)}"
        searchgroup = re.search(pattern, cls.string)
        if searchgroup:
            indent = searchgroup.group('indent')
        else:
            indent = cls.default_style['indent']
        cls.string = re.sub(pattern, '', cls.string)
        return int(indent)


if __name__ == '__main__':
    mystyle = {
        # At the momment only font filenames are supported. That means the font
        # must be in the same directory as the main script.
        # Or you could (should?) use a resource manager such as
        'font': 'Fontin.ttf',
        'size': 20,
        'indent': 0,
        'bold': False,
        'italic': False,
        'underline': False,
        'color': (128, 144, 160),  # RGB values
        'align': 'left',
        # if a separate file should be used for italic/bold, speciy it;
        # if not, use None
        'separate_italic': 'Fontin-Italic.ttf',
        'separate_bold': 'Fontin-Bold.ttf'
    }

    Style.set_default(mystyle)
    plain_text, new_style = Style.split("{bold 'True'}Boldy!")
    print('\n"{}"'.format(new_style))
