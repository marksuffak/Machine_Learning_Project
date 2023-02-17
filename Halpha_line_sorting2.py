#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
""" Halpha_line_sorting2.py
Takes Halpha line data and assigns them either singly or doubly peaked.
Also writes data to a separate text file to use in machine learning project.
"""

__author__ = "Mark Suffak"
__contact__ = "msuffak@uwo.ca"
__date__ = "2023/02/22/"
__email__ =  "msuffak@uwo.ca"
__version__ = "0.0.1"

import numpy as np
import os as os
import scipy.interpolate as interp
from astropy.convolution import convolve, Box1DKernel
import tools

#list of redenned fullseds from HDUST for a range of model numbers
def redlist(modfolder, suffix, modrange):
    files = []
    for j in modrange:
        if j < 10:
            files.append(modfolder+'/fullsed/fullsed_mod0'+str(j)+suffix+'_red.txt')
        else:
             files.append(modfolder+'/fullsed/fullsed_mod'+str(j)+suffix+'_red.txt')
    return files

# Get list of all fullsed files in desires model numeber range
fullsed_files = redlist('hdust212/new_temp_grid/temp_models', 'a',range(67,87))

# center wavelength for line you're analyzing (Halpha = 0.6564 um)
center_wavelength = 0.6564

# define needed lists and loop over every file
train_flux_table = []
obslist = []
for j in range(len(fullsed_files)):
    print(j)
    # find model number and ending from file name
    suffix = fullsed_files[j].split('fullsed')[2].split('_')[1].split('mod')[1][-1]
    modnumber = fullsed_files[j].split('fullsed')[2].split('_')[1].split('mod')[1].split(suffix)[0]
    # load data
    redfullsed = np.loadtxt(fullsed_files[j])
    # get number of observers and data points per observer in the simulation
    nobs, ppo = tools.nobs_ppo(redfullsed)
    # get base density and file name of the simulation
    rho, file = tools.tilted_file_and_rho(fullsed_files[j])
    # loop over observing angles
    for m in range(nobs):
        # get normalized Halpha line data
        lambdatable, lambdavelotable, normfluxtable, ew = tools.norm_Halpha(m * ppo, m * ppo + ppo - 1,
                                                                            redfullsed)

        # convolve line with gaussian of fwhm 0.656 A to lower resolution and avoid noise
        fwhm = center_wavelength / 2500  # um (lambda/FWHM = 2500)
        st_dev = 0.5 * fwhm / np.sqrt(2.0 * np.log(2)) # standard deviation
        # create a gaussian kernel and convolve with normalized line
        diffs = []
        for i in range(1, len(lambdatable)):
            diffs.append(lambdatable[i] - lambdatable[i - 1])
        avgspace = sum(diffs) / len(diffs)
        kern_width = fwhm / avgspace
        gauss_kernel = Box1DKernel(kern_width)
        convflux = convolve(normfluxtable, gauss_kernel)
        # create interpolating function from convolved line
        f = interp.UnivariateSpline(lambdatable, convflux,
                                    s=0, k=4, ext=0)  # interpolates function of 1-flux over all lambdas
        # find roots of the derivative of interpolating function
        roots = (f.derivative().roots())
        # find half the maximum flux
        halfmax = (max(convflux) - min(convflux)) / 2 + min(convflux)
        # select only wavelengths around the half-maximum flux value
        lambdas = []
        for i in range(len(normfluxtable) - 1):
            if convflux[i] <= halfmax <= convflux[i + 1] or convflux[i] >= halfmax >= \
                    convflux[i + 1]:
                lambdas.append(lambdatable[i])
                lambdas.append(lambdatable[i + 1])
        # compute line center
        line_center = (lambdas[0] + lambdas[-1]) / 2

        # get center of line info
        center_idx = np.where(abs(roots - line_center) == min(abs(roots - line_center)))[0][0]
        center_flux = f(roots[center_idx])

        # find flux of first peak
        peak_1_flux = f(roots[center_idx - 1])
        peak_1_idx = center_idx - 1
        for i in range(2, center_idx):
            if f(roots[center_idx - i]) > peak_1_flux:
                peak_1_flux = f(roots[center_idx - i])
                peak_1_idx = center_idx - i
            else:
                break

        # find flux of second peak
        peak_2_flux = f(roots[center_idx + 1])
        peak_2_idx = center_idx + 1
        for idx in range(center_idx + 2, len(roots)):
            if f(roots[idx]) > peak_2_flux:
                peak_2_flux = f(roots[idx])
                peak_2_idx = idx
            else:
                break

        # find flux of first minimum
        min_1_flux = f(roots[center_idx - 1])
        min_1_idx = center_idx - 1
        for i in range(2, center_idx):
            if f(roots[center_idx - i]) < min_1_flux:
                min_1_flux = f(roots[center_idx - i])
                min_1_idx = center_idx - i
            else:
                break

        # find flux of second minimum
        min_2_flux = f(roots[center_idx + 1])
        min_2_idx = center_idx + 1
        for idx in range(center_idx + 2, len(roots)):
            if f(roots[idx]) < min_2_flux:
                min_2_flux = f(roots[idx])
                min_2_idx = idx
            else:
                break

        # determine if line is double peaked, singly peaked, or "V-shaped" (in absorption)
        if center_flux < peak_1_flux and center_flux < peak_2_flux and peak_1_flux < 1 and peak_2_flux < 1:
            if f(roots[peak_1_idx] - 0.0001) < peak_1_flux > f(roots[peak_1_idx] + 0.0001) and f(
                    roots[peak_2_idx] + 0.0001) < peak_2_flux > f(roots[peak_2_idx] - 0.0001):
                line_shape = "Double Peaked"
            elif center_flux == min(f(roots)):
                line_shape = "V Shape"
            else:
                line_shape = 'Double Peaked'
        elif center_flux < peak_1_flux and center_flux < peak_2_flux and peak_1_flux > 1 and peak_2_flux > 1:
            line_shape = "Double Peaked"
        elif center_flux > peak_1_flux and center_flux > peak_2_flux and center_flux > 1:
            if abs(center_flux - 1) < abs(peak_1_flux - 1):
                line_shape = "Double Peaked"
            else:
                line_shape = "Singly Peaked"
        elif center_flux > peak_1_flux and center_flux > peak_2_flux and center_flux < 1:
            line_shape = "Double Peaked"

        # write line shape, EW, and flux data to text file for use in machine learning code
        if not os.path.exists('hdust212/new_temp_grid/Halpha_text_files'):
            os.mkdir('hdust212/new_temp_grid/Halpha_text_files')
        with open('hdust212/new_temp_grid/Halpha_text_files/' + 'mod'+str(modnumber) + '_mu' +
                  str(np.int(redfullsed[m * ppo, 0])) + '_phi' + str(np.int(redfullsed[m * ppo, 1])) + '.txt', 'w') as f:
            f.write('mod'+str(modnumber) + ',' + line_shape + '\n')
            f.write(str(ew)+'\n')
            for j in range(len(convflux)):
                f.write(str(lambdavelotable[j]) + ',' + str(convflux[j]) + '\n')
