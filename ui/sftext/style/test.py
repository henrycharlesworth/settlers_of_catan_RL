from style import Style
from pprint import pprint

old_style = {
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

mystring = ("Hello World!")

# you can start with an already formatted string
print()
print("This is our test text:")
print('"{}"'.format(mystring))

# you can format plain text with a given style or the default one
print("\nThis is our text with style:")
newstring = Style.stylize(mystring, old_style)
print('"{}"'.format(newstring))

# Style.split returns a string without its style and a separate style dict
text, newstyle = Style.split(newstring)
print("\nThe style format is a dictionary like this one:")
pprint(newstyle)
print("\nThis is formatted text with its style removed.")
print("It shoul be equal to the text we started with.")
print('"{}"'.format(text))
assert(text == mystring)
assert(newstyle == old_style)

newstyle['color'] = (255, 255, 255)
newstyle['size'] = 18
# you can set a default style from a dict
Style.set_default(newstyle)

# or you can set a default style from a formatted string.
# both will do the same.
style2 = Style.set_default(mystring)

# you can format plain text with a given style
# in this example the styles is the same as the on we started with...
# but you could change the style of a text by using:
#   formatted1 = Style.stylize(plain_text, style1)
#   formatted2 = Style.stylize(
#       string=Style.remove(formatted1),
#       style = style2
#   )
