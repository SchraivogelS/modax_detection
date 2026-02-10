#!/usr/bin/env python3

"""
Plotting utility functions
Author: SCS
Date: 26.07.2021
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from plotly.offline import iplot
from plotly import figure_factory as FF

import plotly.io as pio
pio.renderers.default = "browser"

def get_annotation_dict(point, label):
    ret = dict(
        showarrow=False,
        x=point[0],
        y=point[1],
        z=point[2],
        text=label,
        xanchor="left",
        xshift=10,
        opacity=0.7,
        font=dict(color="green", size=12)
    )
    return ret


def iplot_mesh(verts, faces, landmarks=None, title='', alpha_surf=1, orthographic=False,
               equal_axis_limit: int = None, show=True, save_dir=None):
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    # colormap = ['rgb(254, 205, 98)', 'rgb(106, 38, 6)']

    if equal_axis_limit is not None and equal_axis_limit <= 0:
        raise ValueError(f'Invalid axis limit {equal_axis_limit}')

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=px.colors.sequential.YlOrBr,  # algae, amp
                            simplices=faces,
                            edges_color=None,
                            backgroundcolor='rgb(255, 255, 255)',
                            show_colorbar=False,
                            title=title)
    fig['data'][0].update(opacity=alpha_surf)

    if orthographic:
        fig.update_geos(projection_type="orthographic")
        #  fig.layout.scene.camera.projection.type = "orthographic"

    if equal_axis_limit is not None:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-equal_axis_limit, equal_axis_limit]),
                yaxis=dict(range=[-equal_axis_limit, equal_axis_limit]),
                zaxis=dict(range=[-equal_axis_limit, equal_axis_limit])
            )
        )

    if landmarks:
        fig.add_trace(
            go.Scatter3d(
                x=[i[0] for i in landmarks.values()],
                y=[i[1] for i in landmarks.values()],
                z=[i[2] for i in landmarks.values()],
                mode="markers",
                marker=dict(size=5)
            )
        )
        fig.update_layout(
            scene=dict(
                annotations=[get_annotation_dict(val, key) for key, val in landmarks.items()]
            )
        )

    if show:
        iplot(fig)

    if save_dir is not None and os.path.exists(save_dir):
        fig_path = os.path.join(save_dir, title + '.html')
        fig.write_html(fig_path)
        print(f'Wrote figure to {fig_path}')
    return fig


def plot_landmarks_on_axes(landmarks, ax, keys=('RW', 'OW', 'C', 'A')):
    from Annotation3D import annotate_3D_point_on_axis as annotate

    for key in keys:
        ax.scatter3D(landmarks[key][0], landmarks[key][1], landmarks[key][2])
        annotate(ax, landmarks[key], key)


def pyplot_mesh(verts, faces, face_normals=None, landmarks=None, alpha_surf=1.0, block=False,
                equal_axis_limit: int = None) -> plt.axes:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    # todo: cannot remove edge lines
    mesh = Poly3DCollection(verts[faces], facecolors='g', edgecolor='none', linewidth=0,
                            alpha=alpha_surf)

    if face_normals is not None:
        from matplotlib.colors import LightSource
        light = LightSource(azdeg=-83, altdeg=61)  # 225.0, altdeg=45.0

        # Prevent shadows of the image being too dark (correct by linear interpolation)
        min = np.min(light.shade_normals(face_normals, fraction=1.0))  # min shade value
        max = np.max(light.shade_normals(face_normals, fraction=1.0))  # max shade value
        diff = max - min
        newMin = 0.3
        newMax = 0.95
        newdiff = newMax - newMin

        # Using a constant color, put in desired RGB values here.
        colourRGB = np.array((54.0 / 255.0, 1.0, 57 / 255.0, 0.8))

        # Use face normals and light orientation to generate shading value
        # apply to RGB colors for each face.
        rgbNew = np.array([colourRGB * (newMin + newdiff * ((shade - min) / diff)) for shade in
                           light.shade_normals(face_normals, fraction=1.0)])

        # Apply color to face
        mesh.set_facecolor(rgbNew)

    ax.add_collection3d(mesh)
    # alternatively
    # ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
    #                linewidth=0.2, antialiased=True)

    if landmarks is not None:
        plot_landmarks_on_axes(landmarks, ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if equal_axis_limit is not None:
        ax.set_xlim(-equal_axis_limit, equal_axis_limit)
        ax.set_ylim(-equal_axis_limit, equal_axis_limit)
        ax.set_zlim(-equal_axis_limit, equal_axis_limit)

    ax.view_init(azim=-83, elev=61)

    plt.tight_layout()
    plt.show(block=block)
    return ax


def iplot_rotax_on_fig(rotax, center_vzero, fig):
    # Rotation axis
    soa = np.array([[center_vzero[0], center_vzero[1], center_vzero[2], rotax[0], rotax[1],
                     rotax[2]]])
    x, y, z, u, v, w = soa.T

    fig.add_trace(
        go.Scatter3d(
            mode='lines',
            x=np.array([x, u]).flatten(),
            y=np.array([y, v]).flatten(),
            z=np.array([z, w]).flatten(),
            line=dict(
                color='black',
                width=6
            )
        )
    )

    fig.add_trace(
        go.Scatter3d(
            mode='markers',
            x=x,
            y=y,
            z=z,
            marker=dict(
                color='white',
                size=6,
                line=dict(
                    color='black',
                    width=3
                )
            )
        )
    )

    # set camera position
    camera = dict(
        eye=dict(x=0.5, y=-0.5, z=2)
    )

    # arrow
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    x=u,
                    y=v,
                    z=w,
                    ax=x[0],
                    ay=y[0],
                    arrowhead=3,
                    xanchor="left",
                    yanchor="bottom"
                )]
        ),
        scene_camera=camera
    )

    fig.update_traces(showlegend=False)
    fig.show()
    return fig


def plot_HU(hu, block=False):
    """
    Plot Hounsfield units (HU)
    @param hu: Hounsfield units to plot
    """
    plt.hist(hu.flatten(), bins=50, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.title('Air -1000, Lung -500, Fat [-100,-50], Water 0\n'
              'Blood [30,70], Muscle[10,40], Liver[40,60], Bone[700,3000]')
    plt.show(block=block)
