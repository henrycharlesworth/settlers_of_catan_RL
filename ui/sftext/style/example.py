from style import Style

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
bold_specific_font_text = Style.stylize("{bold True}Boldy!")
print('\n"{}"'.format(bold_specific_font_text))
