"""
description
"""


# LIBRARIES


import random_toolpath_gen as rtg

import compas.geometry as cg


# HARD CODED VALUES


BASE = cg.Point(0, 0, 0)
COMPAS_FRAME = cg.Frame(BASE, cg.Vector.Xaxis(), cg.Vector.Yaxis())

fig_x = 100.
fig_y = 100.
fig_z = 0.


# RUN CODE


# draw a line
random_line = rtg.random_line_gen(COMPAS_FRAME, fig_x, fig_y)

# save path image
rtg.save_line_image(random_line, fig_x, fig_y, iteration=0)
