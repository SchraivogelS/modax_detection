#!/usr/bin/env python3

"""
Class for robust modiolar axis detection
Author: WW, NH, SCS
Date: 26.07.2021

###########################################################################
# Spiral Shape Recognition using Equiform Motions
# SNF Postdoc.Mobility Fellowship
    
# (C) 2018-2019 W. WIMMER  v8.2 - Robust Detection - Extended field
# HEARING RESEARCH LABORATORY, UNIVERSITY OF BERN (CH)
# EPIONE, INRIA SOPHIA ANTIPOLIS (FR)
###########################################################################
"""

import CochlearMesh
import os
import sys
import logging
import configparser
import locale
locale.setlocale(locale.LC_ALL, '')  # Format of number printing {<number>:n}, Use '' for auto

from modax.modax_funs import *
import utils.plot_util as putil
import utils.io_util as ioutil
import utils.pydicom_util as pyutil


class ModiolarAxis:

    def __init__(self):
        # Adjustable parameters
        self.max_iter = 15
        self.plot_resulting_fit = True
        self.compare_results = False
        self.save_fit_to_file = False
        self.verbose = True
        path_cfg = r'modax_settings.ini'
        # load config
        config = configparser.ConfigParser()
        config.read(path_cfg)
        self.spec_name = config['itide']['SinglePatient']  # specimen for modiolar axis detection
        print(f'SinglePatient: {self.spec_name}')
        self.w_p = float(config['itide']['w_p'])
        print(f'w_p: {self.w_p}')
        study = config['itide']['Study']
        print(f'Study: {study}')
        self.base_dir = base_dir_for(study)
        # Other members
        self.Cochlea = CochlearMesh.CochlearMesh(self.base_dir, self.spec_name)
        self._initialized = False
        self._fit_c = None
        self._fit_c_bar = None
        self._fit_gamma = None
        self._kin_z_world = None
        self._kin_A_world = None
        self._generator = None
        self._spec_out_dir = ''

    @property
    def initialized(self):
        return self._initialized

    @property
    def fit_c(self):
        return self._fit_c

    @property
    def fit_c_bar(self):
        return self._fit_c_bar

    @property
    def fit_gamma(self):
        return self._fit_gamma

    @property
    def kin_z_world(self):
        return self._kin_z_world

    @property
    def kin_A_world(self):
        return self._kin_A_world

    @property
    def generator(self):
        return self._generator

    @property
    def spec_out_dir(self):
        return self._spec_out_dir

    def check_default_params(self):
        if self.max_iter != 15 or \
                self.w_p != 0.01 or \
                self.Cochlea.iso_th != 1000 or \
                self.Cochlea.crop_radius != 1.0 or \
                self.Cochlea.filter_size != 3:
            logging.warning('Parameters differ from recommended defaults')

    def get_spec_out_dir(self):
        if not self.initialized:
            raise ValueError('Missing initialization')
        spec_out_dir = os.path.join(self.Cochlea.base_dir, self.Cochlea.spec_name, '_desc')
        return spec_out_dir

    def init_cochlea(self):
        self.Cochlea.init()
        self._initialized = True
        self._spec_out_dir = self.get_spec_out_dir()

    def save_fit(self):
        fit_result = defaultdict(dict)

        fit_result['id'] = self.spec_name
        fit_result['side'] = side_to_str(self.Cochlea.side)
        fit_result['iso_th'] = self.Cochlea.iso_th
        fit_result['crop_radius'] = self.Cochlea.crop_radius
        fit_result['filter_size'] = self.Cochlea.filter_size
        fit_result['coord_sys'] = 'RAS'  # store fit in RAS anatomical coordinate system
        fit_result['mesh_from_matlab'] = self.Cochlea.mesh_from_matlab

        fit_result['landmarks']['RW'] = self.Cochlea.specimen_data['landmarks']['RW']
        fit_result['landmarks']['C'] = self.Cochlea.specimen_data['landmarks']['C']
        fit_result['landmarks']['A'] = self.Cochlea.specimen_data['landmarks']['A']
        fit_result['landmarks']['OW'] = self.Cochlea.specimen_data['landmarks']['OW']

        fit_result['fit']['P0'] = self.kin_z_world.T
        fit_result['fit']['AX'] = self.kin_A_world.T
        fit_result['fit']['normalizer'] = self.Cochlea.normalizer
        fit_result['fit']['centralizer'] = self.Cochlea.centralizer
        fit_result['fit']['fit_c'] = self.fit_c
        fit_result['fit']['fit_c_bar'] = self.fit_c_bar
        fit_result['fit']['fit_gamma'] = self.fit_gamma
        fit_result['fit']['max_iter'] = self.max_iter
        fit_result['fit']['w_p'] = self.w_p
        fit_result['fit']['min_len_verts'] = self.Cochlea.min_len_verts

        fit_result['mesh']['vertices_local'] = self.Cochlea.verts_loc
        fit_result['mesh']['faces'] = self.Cochlea.faces_loc

        # use correct coordinates in anatomical system
        if self.Cochlea.specimen_data['coord_sys'] == 'LPS':
            for key, val in fit_result['landmarks'].items():
                fit_result['landmarks'][key] = pyutil.swap_ras_lps(val)
            fit_result['fit']['P0'] = pyutil.swap_ras_lps(fit_result['fit']['P0'])
            fit_result['fit']['AX'] = pyutil.swap_ras_lps(fit_result['fit']['AX'])
            fit_result['mesh']['vertices_local'] = pyutil.swap_ras_lps(
                fit_result['mesh']['vertices_local'])

        # Output directory
        if not os.path.exists(self.spec_out_dir):
            os.mkdir(self.spec_out_dir)

        ioutil.save_np_values_to_json(fit_result,
                                      os.path.join(self.spec_out_dir,
                                                   self.Cochlea.spec_name + '-fit-modax.json'))

    def plot_fit(self, kin_A, center_v_zero):
        vnorm = self.Cochlea.verts_normalized
        fnorm = self.Cochlea.faces_normalized

        # --- matplotlib
        # mesh_normalized = trimesh.Trimesh(vertices=vnorm, faces=fnorm)
        # ax = putil.pyplot_mesh(vnorm, fnorm, face_normals=mesh_normalized.face_normals,
        #                        alpha_surf=0.8, equal_axis_limit=1)
        # arrow_len = 1 / 3 * (np.max(vnorm[:, 2]) - np.min(vnorm[:, 2]))
        # putil.plot_rotax_on_axes(kin_A, center_v_zero, ax, arrow_len=arrow_len, block=True)

        # --- plotly
        title = f'{self.spec_name}_P0{self.kin_z_world}_AX{self.kin_A_world}'
        fig = putil.iplot_mesh(vnorm, fnorm, orthographic=True, equal_axis_limit=1,
                               title=title, show=False)
        fig = putil.iplot_rotax_on_fig(kin_A, center_v_zero, fig=fig)

        if self.save_fit_to_file:
            fig_name = f'{self.spec_name}_axis_fitting'
            fig_path = os.path.join(self.get_spec_out_dir(), fig_name + '.html')
            fig.write_html(fig_path)
            print(f'Wrote figure to {fig_path}')

    def get_kin_z(self):
        kin_z = get_center_vzero(self.fit_c, self.fit_c_bar, self.fit_gamma)
        kin_z_unnormalized = kin_z * self.Cochlea.normalizer + self.Cochlea.centralizer
        kin_z_world = kin_z_unnormalized @ self.Cochlea.rot_mat + self.Cochlea.coord_sys[
            'Origin']  # back transformation from local
        if self.verbose:
            print(f'--- Center of velocity zero P0 = {kin_z_world} (local {kin_z})')
        return kin_z, kin_z_world

    def get_kin_A(self):
        # Note: Angle between 3D vectors: atan2d(norm(cross(a, b)), dot(a, b))
        kin_A, kin_A_bar = get_rotation_axis(self.fit_c, self.fit_c_bar, self.fit_gamma)
        kin_A_world = kin_A @ self.Cochlea.rot_mat  # back transformation from local
        kin_A_world /= np.linalg.norm(kin_A_world)
        if self.verbose:
            print(f'--- Rotation axis AX = {kin_A_world} (local {kin_A})')
        return kin_A, kin_A_world

    def fit_axis(self):
        if not self.initialized:
            raise ValueError('Missing initialization')
        if self.verbose:
            print('Fit modiolar axis ...')

        self.check_default_params()

        self._fit_c, \
        self._fit_c_bar, \
        self._fit_gamma = fit_cochlear_shape(self.Cochlea.verts_normalized,
                                             self.Cochlea.vertex_normals,
                                             self.max_iter, self.w_p, verbose=self.verbose)

        # Center of velocity zero
        kin_z, self._kin_z_world = self.get_kin_z()

        # Rotation axis
        kin_A, self._kin_A_world = self.get_kin_A()

        if self.verbose:
            print(f'Done for specimen {self.spec_name}')

        if self.compare_results:
            compare_results_to_matlab(self.Cochlea.spec_name, self.kin_A_world, self.kin_z_world)

        if self.plot_resulting_fit:
            self.plot_fit(kin_A, kin_z)

    # TODO: Fails for several (DICOM) subjects
    def generate_sample_curve(self):
        fit = ioutil.load_values_from_json(
            os.path.join(self.spec_out_dir, self.Cochlea.spec_name + '-fit-modax.json'))
        fit = fit['fit']

        try:
            # Slice polarly
            generator = generate_sample_curve(self.Cochlea.labyrinth_loc, fit,
                                              self.Cochlea.normalizer, self.Cochlea.centralizer)
            self._generator = generator
            save_as_csv(generator, self.spec_out_dir, self.Cochlea.spec_name + '-generator')
        except Exception as err:
            logging.error(f'Caught exception while generating sample curve:\n{err}')

    def plot_sample_curve(self):
        no_torsion = np.array([[0, 0, 0]])
        fit = ioutil.load_values_from_json(
            os.path.join(self.spec_out_dir, self.Cochlea.spec_name + '-fit-modax.json'))
        fit = fit['fit']
        curve, full_curve, curvature, torsion, arc_length = generate_shape(no_torsion,
                                                                           fit['fit_c'],
                                                                           fit['fit_c_bar'],
                                                                           fit['fit_gamma'],
                                                                           self.generator,
                                                                           fit['P0'],
                                                                           fit['normalizer'],
                                                                           fit['centralizer'],
                                                                           fit['AX'])
        return


if __name__ == "__main__":
    Modax = ModiolarAxis()
    Modax.init_cochlea()
    Modax.fit_axis()
    if Modax.save_fit_to_file:
        Modax.save_fit()
    # Modax.generate_sample_curve()
    # Modax.plot_sample_curve()
    sys.exit(0)
