"""
description
"""
# LIBRARIES

# from compas_view2.app import App
from collections import deque
import compas.geometry as cg
import compas.utilities as cu
import random as r
import math as m
import numpy as np
import cv2
import json


def clean(type='along_x'):

    # set global facts
    with open('data/facts.json') as f:
        facts = json.load(f)

    __BLADE_W__ = 50
    __MERGIN__ = 100

    bounds = facts['feature_bounds']
    min_bound = bounds['min_bound']
    max_bound = bounds['max_bound']

    x_length = max_bound[1] - min_bound[1]
    y_length = max_bound[0] - min_bound[0]

    x_num = m.floor(y_length / __BLADE_W__)
    y_num = m.floor(x_length / __BLADE_W__)

    frames = []

    if type == 'along_y':
        __ANGLE__ = 15
        for i in range(y_num-1):
            y_start = cg.Point((i+1) * __BLADE_W__, 0 + __MERGIN__, 65)
            y_end = cg.Point((i+1) * __BLADE_W__, y_length - __MERGIN__, 65)
            f_start = cg.Frame(y_start, cg.Vector.Xaxis(), cg.Vector.Yaxis())
            f_end = cg.Frame(y_end, cg.Vector.Xaxis(), cg.Vector.Yaxis())

            if i % 2 == 0:
                xaxis = cg.Vector(m.cos(m.radians(-__ANGLE__)), m.sin(m.radians(-__ANGLE__)), 0)
                yaxis = cg.Vector(-m.sin(m.radians(-__ANGLE__)), m.cos(m.radians(-__ANGLE__)), 0)
                f_start = cg.Frame(y_start, xaxis, yaxis)
                f_end = cg.Frame(y_end, xaxis, yaxis)
                frames.append(f_start)
                frames.append(f_end)
            else:
                xaxis = cg.Vector(m.cos(m.radians(__ANGLE__)), m.sin(m.radians(__ANGLE__)), 0)
                yaxis = cg.Vector(-m.sin(m.radians(__ANGLE__)), m.cos(m.radians(__ANGLE__)), 0)
                f_start = cg.Frame(y_start, xaxis, yaxis)
                f_end = cg.Frame(y_end, xaxis, yaxis)
                frames.append(f_end)
                frames.append(f_start)

    elif type == 'along_x':
        __ANGLE__ = -15
        for i in range(x_num-1):
            x_start = cg.Point(0 + __MERGIN__/2, (i+1) * __BLADE_W__, 65)
            x_end = cg.Point(x_length - __MERGIN__/2, (i+1) * __BLADE_W__, 65)
            f_start = cg.Frame(x_start, cg.Vector.Xaxis(), cg.Vector.Yaxis())
            f_end = cg.Frame(x_end, cg.Vector.Xaxis(), cg.Vector.Yaxis())

            if i % 2 == 0:
                xaxis = cg.Vector(m.sin(m.radians(-__ANGLE__)), -m.cos(m.radians(-__ANGLE__)), 0)
                yaxis = cg.Vector(m.cos(m.radians(-__ANGLE__)), m.sin(m.radians(-__ANGLE__)), 0)
                f_start = cg.Frame(x_start, xaxis, yaxis)
                f_end = cg.Frame(x_end, xaxis, yaxis)
                frames.append(f_start)
                frames.append(f_end)
            else:
                xaxis = cg.Vector(m.sin(m.radians(__ANGLE__)), -m.cos(m.radians(__ANGLE__)), 0)
                yaxis = cg.Vector(m.cos(m.radians(__ANGLE__)), m.sin(m.radians(__ANGLE__)), 0)
                f_start = cg.Frame(x_start, xaxis, yaxis)
                f_end = cg.Frame(x_end, xaxis, yaxis)
                frames.append(f_end)
                frames.append(f_start)

    return frames
