"""
This module contains random toolpath gen func:
    1) generate info for path
    2) Transformation from fig to box
    3) plotter to save image
"""


# LIBRARIES


# import cv2
import matplotlib.pyplot as plt
import compas.geometry as cg
import random as r
import math as m


# FUNCTIONS
'''
1) gen_xdir_line
2) rotate_line
3) centerize_line
4) get_path
5) save_line_image
6) random_line_gen
7) deploy_fig_to_box
'''


def gen_xdir_line(frame, x_len, min_fac=0.5, max_fac=1.0):
    start = frame.point
    factor = r.random()
    if factor < min_fac:
        factor = min_fac
    elif factor > max_fac:
        factor = max_fac
    length = factor * x_len
    end = cg.Point(frame.point.x+length, frame.point.y, frame.point.z)
    line = cg.Line(start, end)
    return line


def rotate_line(line):
    r_frame = cg.Frame(line.midpoint, cg.Vector.Xaxis(), cg.Vector.Yaxis())
    r_axis = r_frame.zaxis
    r_angle = m.radians(r.randrange(0, 359))
    R = cg.Rotation.from_axis_and_angle(r_axis, r_angle)
    line_rotate = line.transformed(R)
    return line_rotate


def centerize_line(line, fig_x, fig_y):
    framefrom = cg.Frame(line.midpoint, cg.Vector.Xaxis(), cg.Vector.Yaxis())
    fig_center = cg.Point(fig_x/2, fig_y/2, 0)
    frameto = cg.Frame(fig_center, cg.Vector.Xaxis(), cg.Vector.Yaxis())
    T = cg.Transformation.from_frame_to_frame(framefrom, frameto)
    line_trans = line.transformed(T)
    return line_trans


def get_path(iteration):
    DIR = 'G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/00_test/02_toolpath'
    num = "{0:05}".format(iteration)
    FILE = "/toolpath_{}.png".format(num)
    PATH = DIR + FILE
    return PATH


def save_line_image(line_to_draw, fig_x, fig_y, iteration):
    PATH = get_path(iteration)
    x1 = [line_to_draw.start[0], line_to_draw.end[0]]
    y1 = [line_to_draw.start[1], line_to_draw.end[1]]
    plt.figure(figsize=(10, 10))
    plt.xlim([0, fig_x])
    plt.ylim([0, fig_y])
    plt.axis('off')
    plt.plot(x1, y1)
    # plt.show()
    plt.savefig(PATH)


def random_line_gen(COMPAS_FRAME, fig_x, fig_y):
    line_x = gen_xdir_line(COMPAS_FRAME, x_len=fig_x)
    line_r = rotate_line(line_x)
    line_centerized = centerize_line(line_r, fig_x, fig_y)
    return line_centerized


def deploy_fig_to_box(line, fig_x, fig_y, fig_z, box_x, box_y, box_z):
    framefrom = cg.Frame(line.midpoint, cg.Vector.Xaxis(), cg.Vector.Yaxis())

    x_val = r.randrange(int(fig_x/2), int(box_x-fig_x/2))
    y_val = r.randrange(int(fig_y/2), int(box_y-fig_y/2))
    origin = cg.Point(x_val, y_val, 0)
    frameto = cg.Frame(origin, cg.Vector.Xaxis(), cg.Vector.Yaxis())

    T = cg.Transformation.from_frame_to_frame(framefrom, frameto)
    line_trans = line.transformed(T)
    return line_trans, x_val, y_val


if __name__ == "__main__":
    pass
