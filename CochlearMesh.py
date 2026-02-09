#!/usr/bin/env python3

"""
Class for cochlear mesh generation
Author: WW, NH, SCS
Date: 31.08.2021

###########################################################################
# Spiral Shape Recognition using Equiform Motions
# SNF Postdoc.Mobility Fellowship

# (C) 2018-2019 W. WIMMER  v8.2 - Robust Detection - Extended field
# HEARING RESEARCH LABORATORY, UNIVERSITY OF BERN (CH)
# EPIONE, INRIA SOPHIA ANTIPOLIS (FR)
###########################################################################
"""

import os
import numpy as np
import errno
import time
import configparser

import plot_utils as plot_utils
import modax_funs as modax_funs


class CochlearMesh:

    def __init__(self, base_dir, spec_name, config_dir=r'./'):
        # Adjustable parameters
        self.base_dir = base_dir
        self.spec_name = spec_name
        self.load_dicom = True
        self.smooth_mesh = False
        self.mask_cochlea = True
        self.min_len_verts = 5000
        self.plot_surf = False  # takes up to 1 min for big volumes
        self.verbose = False
        path_cfg = os.path.join(config_dir, 'modax_settings.ini')
        # Other members
        self._vertex_normals = None
        self._verts_normalized = None
        self._verts_loc = None
        self._faces_normalized = None
        self._faces_loc = None
        self._normalizer = None
        self._centralizer = None
        self._labyrinth_loc = None
        self._specimen_data = None
        self._translation = None
        self._rot_mat = None
        self._coord_sys = None
        self._side = ''
        self._spec_dir = self.get_spec_dir()
        self._spec_data_dir = self.get_spec_data_dir()
        # load config
        config = configparser.ConfigParser()
        config.read(path_cfg)
        self._iso_th = int(config['modax']['IsoThreshold'])
        print(f'IsoThreshold: {self.iso_th}')
        self._crop_radius = float(config['modax']['CropRadius'])
        print(f'CropRadius: {self.crop_radius}')
        self._filter_size = int(config['modax']['FilterSize'])
        print(f'FilterSize: {self.filter_size}')

    @property
    def side(self) -> int:
        return int(self._side)

    @property
    def coord_sys(self):
        return self._coord_sys

    @property
    def rot_mat(self):
        return self._rot_mat

    @property
    def translation(self):
        return self._translation

    @property
    def specimen_data(self):
        return self._specimen_data

    @property
    def labyrinth_loc(self):
        return self._labyrinth_loc

    @property
    def filter_size(self):
        return self._filter_size

    @property
    def iso_th(self):
        return self._iso_th

    @property
    def crop_radius(self):
        return self._crop_radius

    @property
    def vertex_normals(self):
        return self._vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, normals):
        self._vertex_normals = normals

    @property
    def verts_normalized(self):
        return self._verts_normalized

    @verts_normalized.setter
    def verts_normalized(self, verts):
        self._verts_normalized = verts

    @property
    def verts_loc(self):
        return self._verts_loc

    @verts_loc.setter
    def verts_loc(self, verts_loc):
        self._verts_loc = verts_loc

    @property
    def faces_normalized(self):
        return self._faces_normalized

    @faces_normalized.setter
    def faces_normalized(self, faces_normalized):
        self._faces_normalized = faces_normalized

    @property
    def faces_loc(self):
        return self._faces_loc

    @faces_loc.setter
    def faces_loc(self, faces_loc):
        self._faces_loc = faces_loc

    @property
    def normalizer(self):
        return self._normalizer

    @normalizer.setter
    def normalizer(self, normalizer):
        self._normalizer = normalizer

    @property
    def centralizer(self):
        return self._centralizer

    @centralizer.setter
    def centralizer(self, centralizer):
        self._centralizer = centralizer

    @property
    def spec_data_dir(self):
        return self._spec_data_dir

    @property
    def spec_dir(self):
        return self._spec_dir

    def get_spec_dir(self) -> str:
        spec_dir = os.path.join(self.base_dir, self.spec_name)
        if not os.path.isdir(spec_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), spec_dir)
        return str(spec_dir)

    def get_spec_data_dir(self):
        spec_data_dir = os.path.join(self.get_spec_dir(), 'data')
        if not os.path.isdir(spec_data_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), spec_data_dir)
        return spec_data_dir

    def init(self):
        self.init_specimen()
        self.init_mesh()

    def init_specimen(self):
        self._specimen_data = modax_funs.init_specimen(self.spec_dir, self.spec_name)
        self._translation, self._rot_mat, self._coord_sys, \
            self._side = modax_funs.get_local_coord_sys(self.specimen_data)

    def init_mesh(self):
        self._labyrinth_loc, \
            self._verts_loc, \
            self._faces_loc = self._extract_cochlea_polygon(
                self.specimen_data, self.translation,
                self.rot_mat, smooth_mesh=self.smooth_mesh)

        # Normalize mesh
        self.verts_normalized, \
            self.faces_normalized, \
            self.centralizer, \
            self.normalizer = modax_funs.normalize_mesh(self.verts_loc, self.faces_loc)
        self.vertex_normals = modax_funs.vertex_normals_from_mesh(self.verts_normalized,
                                                       self.faces_normalized)

    def _extract_cochlea_polygon(self, specimen_data, translation, rot_mat, smooth_mesh=False):
        start_time = time.time()

        if self.load_dicom:
            verts_transformed, faces_raw = modax_funs.mesh_from_dicom(
                self.spec_dir, self.spec_name, self.iso_th,
                filter_size=self.filter_size,
                mask_cochlea=self.mask_cochlea, show_mesh=self.plot_surf,
                verbose=self.verbose
            )
        else:
            verts_transformed, faces_raw = modax_funs.mesh_from_surf(specimen_data, self.spec_data_dir,
                                                          self.spec_name)

        modax_funs.update_mesh_data(specimen_data, verts_transformed, faces_raw)

        # Transform to local coordinate system
        labyrinth_loc = modax_funs.transform_labyrinth(specimen_data, translation, rot_mat)
        landmarks_loc = labyrinth_loc['landmarks'].copy()
        verts_loc = labyrinth_loc['vertices'].copy()
        faces_loc = labyrinth_loc['faces'].copy()

        # Extract cochlear polygon
        verts_cochlea_loc, faces_cochlea_loc = self._extract_cochlea(verts_loc, faces_loc,
                                                                     landmarks_loc)
        if smooth_mesh:
            verts_cochlea_loc, faces_cochlea_loc = modax_funs.smooth_laplacian(verts_cochlea_loc,
                                                                    faces_cochlea_loc)

        labyrinth_loc['vertices_processed_unnormalized'] = verts_cochlea_loc.copy()

        if self.verbose:
            print('Extraction took {:.2f}s\n'.format(time.time() - start_time))

        return labyrinth_loc, verts_cochlea_loc, faces_cochlea_loc

    def _extract_cochlea(self, verts_loc, faces_loc, landmarks_loc):
        if self.verbose:
            print('Extract Cochlea ...')

        if self.load_dicom:
            bool_rm_verts = modax_funs.get_verts_to_rm_dicom(verts_loc, landmarks_loc, self.crop_radius)
        else:
            bool_rm_verts = modax_funs.get_verts_to_rm_surf(verts_loc, landmarks_loc)

        n_keep = len(bool_rm_verts) - np.sum(bool_rm_verts)
        if n_keep == 0:
            raise ValueError(f'No vertices left after removal')
        if self.verbose:
            print(f'Remove {np.sum(bool_rm_verts):n} / {len(bool_rm_verts):n} '
                  f'vertices outside cochlear boundary sphere (keep {n_keep})')

        if self.plot_surf:
            plot_utils.iplot_mesh(verts_loc, faces_loc, landmarks=landmarks_loc,
                             title=f'Local mesh {self.spec_name}', alpha_surf=0.3,
                             orthographic=True)

        verts_fragments, faces_fragments = modax_funs.remove_vertices_by_mask(
            verts=verts_loc,
            faces=faces_loc,
            bool_rm_verts=bool_rm_verts
        )

        if self.plot_surf:
            plot_utils.iplot_mesh(verts_fragments, faces_fragments,
                             title=f'Mesh fragments {self.spec_name}', alpha_surf=0.3,
                             orthographic=True)

        # Select Cochlea from Fragments
        verts_cochlea_loc, faces_cochlea_loc = modax_funs.select_cochlea_from_fragments(
            verts_fragments,
            faces_fragments,
            min_len=self.min_len_verts)
        if self.plot_surf:
            plot_utils.iplot_mesh(verts_cochlea_loc, faces_cochlea_loc,
                             title=f'Extracted cochlea {self.spec_name}', alpha_surf=0.3,
                             orthographic=True)
        if self.verbose:
            print(f'Selected cochlea with {len(faces_cochlea_loc)} faces')

        return verts_cochlea_loc, faces_cochlea_loc
