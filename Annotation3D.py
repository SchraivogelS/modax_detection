#!/usr/bin/env python3

"""
Class for 3D plot annotations
Project: ITIDE
Author: SCS
Date: 28.07.2021
Note: Inspired from https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot/42915422#42915422
"""

from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation


def annotate_3D_point_on_axis(ax, point, label, xtext=-1.5, ytext=1.5):
    annotate3D(ax, s=label, xyz=point, fontsize=11, xytext=(xtext, ytext),
               textcoords='offset points', ha='right', va='bottom')


def annotate3D(ax, s, *args, **kwargs):
    """add annotation text s to to Axes3d ax"""

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)


class Annotation3D(Annotation):
    """Annotate the point xyz with text s"""

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, s, xy=(0, 0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.xy = (xs, ys)
        Annotation.draw(self, renderer)
