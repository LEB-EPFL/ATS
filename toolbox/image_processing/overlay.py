from typing import Tuple
import numpy as np
import cairo
import matplotlib.pyplot as plt
import math


def main():

    image = Overlay()
    # image.slow_sign(scale=3)
    # image.timestamp('04:32')
    # image.scale_bar()
    image.title_slide("Test Title", "This is a little longer description for what is \n"
                      "in the movie.\n"
                      "And a third line to test")
    plt.imshow(image.get_image())
    plt.gca().set_facecolor('k')
    plt.show()


class Overlay:

    def __init__(self, shape=(256, 256)):
        self.shape = shape
        width, height = shape
        self.data = np.zeros((width, height, 4), dtype=np.uint8)
        self.surface = cairo.ImageSurface.create_for_data(self.data, cairo.FORMAT_ARGB32, width, height)
        self.context = cairo.Context(self.surface)

    def title_slide(self, title, description):
        self.context.set_source_rgb(1, 1, 1)
        title_pos = (1/30, 1/30)
        self.text(title, title_pos, alignv='top')
        desc_pos = (1/30, 3/30)
        self.multiline_text(description, desc_pos, bold=False)

    def scale_bar(self, width=0.5, size_str='1 μm', font_size=13, pos=(1/15, 1-1/30)):
        if width > 1:
            width = width/self.shape[0]
        self.context.set_source_rgb(1, 1, 1)
        line_height = 1/30
        line_pos = (pos[0], pos[1] - line_height)
        self.line(line_pos, width, height=line_height)
        text_pos = (pos[0] + width/2, line_pos[1] - line_height)
        self.text(size_str, text_pos, alignh='center', alignv='bottom')

    def timestamp(self, time_str, font_size=13, pos=(1-1/30, 1/30)):
        self.context.set_source_rgb(1, 1, 1)
        self.text(time_str, pos, alignh='right', alignv='top')

    def slow_sign(self, scale=1, pos=(1/30, 1/30)):
        self.context.set_source_rgb(1, 1, 1)
        line_width = 1/100*scale
        height = 1/30*scale
        self.line(pos=pos, width=line_width, height=height)
        self.triangle(pos=(pos[0] + line_width*1.3, pos[1]), height=height)

    def triangle(self, pos=(1/30, 1/30), height=1/30, width=None):
        if width is None:
            width = height/2
        im_width, im_height = self.shape
        self.context.move_to(im_width*pos[0], im_height*pos[1])
        self.context.line_to(im_width*pos[0], im_height*pos[1] + im_height*height)
        self.context.line_to(im_width*pos[0] + im_width*width, im_height*pos[1] + im_height*height/2)
        self.context.fill()

    def line(self, pos=(1/30, 1/30), width=1/70, height=1/30):
        im_width, im_height = self.shape

        self.context.move_to(im_width*pos[0], im_height*pos[1])
        self.context.line_to(im_width*pos[0] + im_width*width, im_height*pos[1])
        self.context.line_to(im_width*pos[0] + im_width*width, im_height*pos[1] + im_height*height)
        self.context.line_to(im_width*pos[0], im_height*pos[1] + im_height*height)
        self.context.fill()

    def circle(self, pos, radius = 10, line_width = 2, color = [255, 255, 255, 0.8]):
        self.context.set_source_rgba(*color)
        self.context.set_line_width(2)
        self.context.arc(pos[0], pos[1], radius, 0, 2*math.pi)    
        self.context.stroke()

    def text(self, text_str: str, pos: Tuple, font_size: int = 1/20,
             alignh='left', alignv='bottom', bold: bool = True):
        font_size = font_size*self.shape[1]
        pos = np.multiply(self.shape, pos)
        if bold:
            self.context.select_font_face("Arial", cairo.FONT_SLANT_NORMAL,
                                          cairo.FONT_WEIGHT_BOLD)
        else:
            self.context.select_font_face("Arial", cairo.FONT_SLANT_NORMAL,
                                          cairo.FONT_WEIGHT_NORMAL)
        self.context.set_font_size(font_size)
        (x, y, width, height, dx, dy) = self.context.text_extents(text_str)
        if alignh == 'right':
            pos[0] = pos[0] - width
        elif alignh == 'center':
            pos[0] = pos[0] - width/2

        if alignv == 'top':
            pos[1] = pos[1] + height

        self.context.move_to(*pos)
        self.context.show_text(text_str)

    def multiline_text(self, text_str: str, pos: Tuple, font_size: int = 1/20, bold=False):
        text_list = text_str.split('\n')
        for line in text_list:
            self.text(line, pos, font_size, alignv='top', bold=bold)
            pos = (pos[0], pos[1] + font_size*1.1)


    def get_image(self):
        return self.data


if __name__ == "__main__":
    main()
