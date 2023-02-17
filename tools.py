#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
""" tools.py
Contains various functions to process HDUST output files.
Only a few of these functions are used in the Machine Learning project
"""

__author__ = "Mark Suffak"
__contact__ = "msuffak@uwo.ca"
__date__ = "2022/08/16/"
__email__ =  "msuffak@uwo.ca"
__version__ = "0.0.2"

import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integrate
import scipy.constants as const  
import warnings 
import math
import os

Pi = const.pi
Na = const.Avogadro

#gets values of n and rho_0 from mod files associated with fullsed files #NEEDS FILE NAME, NOT THE OPENED FILE
def n_and_rho(redfullsed, mu = 0.6):
	path =redfullsed.split('fullsed')[0]
	suffix = redfullsed.split('fullsed')[2].split('_')[1].split('mod')[1][-1]
	modnumber = redfullsed.split('fullsed')[2].split('_')[1].split('mod')[1].split(suffix)[0]
	modfile = path + 'mod'+modnumber+'/mod'+modnumber+suffix+'.txt'
	mod = open(modfile,'r')
	modlines = mod.readlines()
	nnotline = modlines[437]
	linelist = nnotline.split()
	nnot = float(linelist[2])
	rho = mu*nnot/Na
	rho = format(rho, '.2e')
	nline = modlines[156]
	linelist = nline.split()
	n = float(linelist[2])+1.5
	n  = format(n,'.1f')
	Rline = modlines[35]
	Rlinelist = Rline.split()
	Renv = float(Rlinelist[2])
	return str(rho), str(n), str(Renv)

def SPH_time_and_rho(redfullsed, mu = 0.6):
	path =redfullsed.split('fullsed')[0]
	suffix = redfullsed.split('fullsed')[2].split('_')[1].split('mod')[1][-1]
	modnumber = redfullsed.split('fullsed')[2].split('_')[1].split('mod')[1].split(suffix)[0]
	modfile = path + 'mod'+modnumber+'/mod'+modnumber+suffix+'.txt'
	mod = open(modfile,'r')
	modlines = mod.readlines()
	xdrline = modlines[136]
	# print(xdrline)
	# fileend = xdrline.split('data.')[1]
	# # fileend = xdrline.split('/')[-1]
	# if list(fileend)[8] == '.':
	# 	time = list(fileend)[0]+list(fileend)[1]+list(fileend)[2]+list(fileend)[3]+list(fileend)[4]+list(fileend)[5]+list(fileend)[6]+list(fileend)[7]
	# else:
	# 	time = list(fileend)[0]+list(fileend)[1]+list(fileend)[2]+list(fileend)[3]+list(fileend)[4]+list(fileend)[5]+list(fileend)[6]+list(fileend)[7]+list(fileend)[8]
	# time = format(float(time),'.2f')
	time = xdrline.split()[2]
	nnotline = modlines[430]
	linelist = nnotline.split()
	nnot = float(linelist[2])
	rho = mu * nnot / Na
	rho = format(rho, '.2e')
	return str(rho), str(time)

def tilted_file_and_rho(redfullsed, mu = 0.6):
	path =redfullsed.split('fullsed')[0]
	suffix = redfullsed.split('fullsed')[2].split('_')[1].split('mod')[1][-1]
	modnumber = redfullsed.split('fullsed')[2].split('_')[1].split('mod')[1].split(suffix)[0]
	modfile = path + 'mod'+modnumber+'/mod'+modnumber+suffix+'.txt'
	mod = open(modfile,'r')
	modlines = mod.readlines()
	xdrline = modlines[136]
	fileend = xdrline.split('/')[-1]
	nnotline = modlines[430]
	linelist = nnotline.split()
	nnot = float(linelist[2])
	rho = mu * nnot / Na
	rho = format(rho, '.2e')
	return str(rho), str("'"+fileend)
	
#gets number of observers and points per observer from redfullsed
def nobs_ppo(redfullsed):
	nobs = 1
	for i in range(1,len(redfullsed)):
		if redfullsed[i,0] != redfullsed[i-1,0] or redfullsed[i,1]!=redfullsed[i-1,1]:
			nobs = nobs+1
	if nobs == 1 :
		ppo = len(redfullsed)
	else:
		for i in range(len(redfullsed)):
			if redfullsed[i,0]!= redfullsed[i+1,0] or redfullsed[i,1]!=redfullsed[i+1,1]:
				ppo = i+1
				break
	return nobs, ppo

#normalizes Halpha line and finds EW
def norm_Halpha(start, stop, redfullsed):
	#define list names
	lambdatable = []
	Hfluxtable = []
	polylambda = []
	polyflux = []
	#sort points into different lists ('poly' lists are used for normalization)
	for i in range(start, stop):
		if redfullsed[i,2]>0.65 and redfullsed[i,2]<0.66:
			lambdatable.append(redfullsed[i,2])
			Hfluxtable.append(redfullsed[i,3])
			# if redfullsed[i,2]<0.6560 or redfullsed[i,2]>0.6570:
			if redfullsed[i,2]<0.6550 or redfullsed[i,2]>0.6580:
				polylambda.append(redfullsed[i,2])
				polyflux.append(redfullsed[i,3])
	#fit linear polynomial to 'poly' lists
	a = np.polyfit(polylambda, polyflux, 1)
	lambdatable = np.array(lambdatable)
	Hfluxtable = np.array(Hfluxtable)
	#normalize every flux value to polynomial 'a'
	normfluxtable = Hfluxtable/(a[1]+a[0]*lambdatable)
	lambdavelotable = ((lambdatable-0.656461)/0.656461)*2.998*10**(5)
#uses eqn(9.59) from carroll and ostille for EW calc (F_c = 1 here)
	ewfluxtable = 1-normfluxtable
	f = interp.interp1d(lambdatable,ewfluxtable,axis=0,fill_value='extrapolate')#interpolates function of 1-flux over all lambdas
	ew = integrate.quad(f,0.653977,0.658586)#chris's fixed integration limits
	# ew = integrate.quad(f,0.656374,0.656544)#chris's fixed integration limits
	ew = ew[0]*1000 #convert ew from micrometers to nanometers
	return lambdatable, lambdavelotable, normfluxtable, ew
	
#normalizes Halpha line and finds EW #USED FOR CONVOLVING LINE WITH GAUSSIAN....HAS LARGER WAVELENGTH RANGE NEEDED FOR GAUSSFOLD
def norm_conv_Halpha(start, stop, redfullsed):
	lambdatable = []
	Hfluxtable = []
	polylambda = []
	polyflux = []
	for i in range(start, stop):
		if redfullsed[i,2]>0.64 and redfullsed[i,2]<0.68:
			lambdatable.append(redfullsed[i,2])
			Hfluxtable.append(redfullsed[i,3])
			if redfullsed[i,2]<0.6555 or redfullsed[i,2]>0.6575:
				polylambda.append(redfullsed[i,2])
				polyflux.append(redfullsed[i,3])
	a = np.polyfit(polylambda, polyflux, 1)
	lambdatable = np.array(lambdatable)
	Hfluxtable = np.array(Hfluxtable)
	normfluxtable = Hfluxtable/(a[1]+a[0]*lambdatable)
	lambdavelotable = ((lambdatable-0.656461)/0.656461)*2.998*10**(5)
#uses eqn(9.59) from carroll and ostille for EW calc (F_c = 1 here)
	ewfluxtable = 1-normfluxtable
	f = interp.interp1d(lambdatable,ewfluxtable,axis=0,fill_value='extrapolate')
	ew = integrate.quad(f,0.653977,0.658586)#chris's fixed integration limits
	ew = ew[0]*1000
	return lambdatable, lambdavelotable, normfluxtable, ew
	
#normalizes H7 and H8 lines and finds their EWs
def norm_H7H8(start, stop, redfullsed):
	lambdatable = []
	fluxtable = []
	polylambda = []
	polyflux = []
	for i in range(start, stop):
		if redfullsed[i,2]>0.385 and redfullsed[i,2]<0.405:#redfullsed[i,2]>0.380 and redfullsed[i,2]<0.405:#redfullsed[i,2]>0.385 and redfullsed[i,2]<0.405:
			lambdatable.append(redfullsed[i,2])
			fluxtable.append(redfullsed[i,3])
			#if fullsed[i,2]<0.388 or 0.39<fullsed[i,2]<0.396 or fullsed[i,2]>0.398:
			if redfullsed[i,2]<0.388 or redfullsed[i,2]>0.398:#redfullsed[i,2]<0.387 or redfullsed[i,2]>0.399 or 0.391<redfullsed[i,2]<0.395:#redfullsed[i,2]<0.388 or redfullsed[i,2]>0.398:
				polylambda.append(redfullsed[i,2])
				polyflux.append(redfullsed[i,3])
	a = np.polyfit(polylambda, polyflux, 1)
	normfluxtable = []
	for i in range(0,len(fluxtable)):
		normfluxtable.append(fluxtable[i]/(a[1]+a[0]*lambdatable[i]))
	normfluxtable = np.array(normfluxtable)
	ewfluxtable = 1-normfluxtable
	f = interp.interp1d(lambdatable,ewfluxtable)
	ew8 = integrate.quad(f,0.3875,0.391)
	ew8 = ew8[0]*1000
	ew7 = integrate.quad(f,0.396,0.399)
	ew7 = ew7[0]*1000
	lambdatable = np.asarray(lambdatable)
	return lambdatable, normfluxtable, ew7, ew8, fluxtable

#finds V magnitude for given wavelength and flux points
def Vmag2(lambdas, fluxes):
	filter_file = np.loadtxt('V_Band.dat')
	# to read the wavelength from filter file
	Filter_lambda = filter_file[:, 0]/10000
	# to read the transmition ratio from filter file
	Filter_trans = filter_file[:, 1]
	'''
	Here we do interpolation to match the transmition info with the wavelengths of
	the fullsed file
	'''
	# fullsed = np.loadtxt(fullsed_file, skiprows = 5)
	SED_lambda = lambdas
	SED_total = fluxes
	Filter_trans_interploated = np.interp(lambdas, Filter_lambda, Filter_trans,
										  left=0, right=0)
	'''
	Here we calculate the magnitude in the desired filter
	'''
	# vega_zpt_V = 3.62e-9
	vega_zpt_V = (3636E-23) * 3E14 / (0.545)**2
	# print(np.where(Filter_trans_interploated!=0))
	filtered_flux_total = np.trapz(Filter_trans_interploated * SED_total,
								   SED_lambda) / np.trapz(Filter_trans_interploated, SED_lambda)
	print(filtered_flux_total, vega_zpt_V)
	magnitude_total = -2.5 * np.log10(filtered_flux_total / vega_zpt_V)
	print(magnitude_total)
	return str(magnitude_total)

#finds V magnitude for given wavelength and flux points
def Vmag(lambdas, fluxes):
	f1 = interp.interp1d(lambdas, fluxes) #interpolate whole SED
	# effective wavelength is 0.545um
	Vflux = f1(0.545)
	# 3636 is flux value at m=0 for B band in Janskys (http://www.astronomy.ohio-state.edu/~martini/usefuldata.html)
	vegaVflux = (3636E-23) * 3E14 / (0.545) ** 2  # converts flux from janskys to wavelength space: erg/s/cm^2/um
	calcVmag = -2.5*math.log(Vflux/(vegaVflux),10) #uses eqn 3.4 from Carroll and Ostille with Vega being the reference star
	return str(calcVmag)

#finds B magnitude for given wavelength and flux points
def Bmag(lambdas, fluxes):
	f1 = interp.interp1d(lambdas,fluxes)
	#effective wavelength is 0.44um
	Bflux = f1(0.438)
	# 4260 is flux value at m=0 for B band in Janskys (https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html)
	vegaBflux = (4063E-23) *3E14/(0.438)**2 # converts flux from janskys to wavelength space: erg/s/cm^2/um
	calcBmag = -2.5*math.log(Bflux/(vegaBflux),10) #uses eqn 3.4 from Carroll and Ostille with Vega being the reference star
	return str(calcBmag)

#finds U magnitude for given wavelength and flux points
def Umag(lambdas, fluxes):
	f1 = interp.interp1d(lambdas,fluxes)
	#effective wavelength is 0.36um
	Uflux = f1(0.36)
	# 1810 is flux value at m=0 for U band in Janskys (https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html)
	vegaUflux = 1790E-23 *3E14/(0.36)**2 # converts flux from janskys to wavelength space: erg/s/cm^2/um
	calcUmag = -2.5*math.log(Uflux/(vegaUflux),10) #uses eqn 3.4 from Carroll and Ostille with Vega being the reference star
	return str(calcUmag)

#finds R magnitude for given wavelength and flux points
def Rmag(lambdas, fluxes):
	f1 = interp.interp1d(lambdas,fluxes)
	#effective wavelength is 0.64um
	Rflux = f1(0.641)
	# 3080 is flux value at m=0 for R band in Janskys (https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html)
	vegaRflux = 3064E-23 *3E14/(0.641)**2 # converts flux from janskys to wavelength space: erg/s/cm^2/um
	calcRmag = -2.5*math.log(Rflux/(vegaRflux),10) #uses eqn 3.4 from Carroll and Ostille with Vega being the reference star
	return str(calcRmag)

#finds I magnitude for given wavelength and flux points
def Imag(lambdas, fluxes):
	f1 = interp.interp1d(lambdas,fluxes)
	#effective wavelength is 0.79um
	Iflux = f1(0.798)
	# 2550 is flux value at m=0 for I band in Janskys (https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html)
	vegaIflux = 2416E-23 *3E14/(0.798)**2 # converts flux from janskys to wavelength space: erg/s/cm^2/um
	calcImag = -2.5*math.log(Iflux/(vegaIflux),10) #uses eqn 3.4 from Carroll and Ostille with Vega being the reference star
	return str(calcImag)

#finds I magnitude for given wavelength and flux points
def Kmag(lambdas, fluxes):
	f1 = interp.interp1d(lambdas,fluxes)
	#effective wavelength is 0.79um
	Kflux = f1(2.22)
	# 670 is flux value at m=0 for K band in Janskys (https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html)
	vegaKflux = 670E-23 *3E14/(2.22)**2 # converts flux from janskys to wavelength space: erg/s/cm^2/um
	calcKmag = -2.5*math.log(Kflux/(vegaKflux),10) #uses eqn 3.4 from Carroll and Ostille with Vega being the reference star
	return str(calcKmag)

#calculates MJD at midnight UT(universal time) on day specified
def MJD(year, month, day):
	JD = 367*year-math.floor(7*(year+math.floor((month+9)/12))/4) -math.floor(3*(math.floor((year+(month-9)/7)/100)+1)/4) + math.floor(275*month/9) + day + 1721028.5 +0/24
	mjd = JD - 2400000.5
	return mjd

def HaLambda0(waves, fluxes):
	halfmax = (max(fluxes)-1)/2 +1
	lambdas = []
	for i in range(len(fluxes)-1):
		if fluxes[i] <= halfmax <= fluxes[i+1] or fluxes[i] >= halfmax >= fluxes[i+1]:
			lambdas.append(waves[i])
			lambdas.append(waves[i+1])
	return (lambdas[0]+lambdas[-1])/2

def HaFWHM(waves, fluxes, lambda_0):
	halfmax = (max(fluxes)-1)/2 +1
	lambdas = []
	for i in range(len(fluxes)-1):
		if fluxes[i] <= halfmax <= fluxes[i+1] or fluxes[i] >= halfmax >= fluxes[i+1]:
			lambdas.append(waves[i])
			lambdas.append(waves[i+1])
	FWHM = abs(lambda_0-lambdas[0])+abs(lambda_0-lambdas[-1])
	print(lambdas[0], lambdas[-1])
	return FWHM

def get_modfolder_path(path):
	head,tail = os.path.split(path)
	tail_len = len(list(tail))
	while tail_len !=5:
		new_head, new_tail = os.path.split(head)
		tail_len = len(list(new_tail))
		head = new_head
		tail = new_tail
	return head, tail

def get_temp_grids(data,nLTE,Rstar):
	ncr, ncmu, ncphi = data[0].shape
	x = np.zeros([ncphi, ncmu, ncr])
	y = np.zeros([ncphi, ncmu, ncr])
	z = np.zeros([ncphi, ncmu, ncr])
	temp = np.zeros([ncphi, ncmu, ncr])
	rho = np.zeros([ncphi, ncmu, ncr])
	ion_frac = np.zeros([ncphi, ncmu, ncr])
	num_dens = np.zeros([ncphi, ncmu, ncr])
	rgrid = np.zeros([ncphi, ncmu, ncr])
	mugrid = np.zeros([ncphi, ncmu, ncr])
	phigrid = np.zeros([ncphi, ncmu, ncr])
	for ridx in range(0, ncr):
		for phiidx in range(0, ncphi):
			for muidx in range(0, ncmu):
				r = data[0, ridx, muidx, phiidx] / Rstar
				costheta = data[1, ridx, muidx, phiidx]
				phi = data[2, ridx, muidx, phiidx]
				rgrid[phiidx,muidx,ridx] = r
				mugrid[phiidx, muidx, ridx] = costheta
				phigrid[phiidx, muidx, ridx] = phi
				if 0.6 * data[5 + nLTE, ridx, muidx, phiidx] / (6.022 * 10 ** (23)) == 0:
					temp[phiidx, muidx, ridx] = 0
					rho[phiidx, muidx, ridx] = 0
					ion_frac[phiidx, muidx, ridx] = 0
					num_dens[phiidx,muidx,ridx] = data[5 + nLTE, ridx, muidx, phiidx]
				else:
					temp[phiidx, muidx, ridx] = data[3, ridx, muidx, phiidx]
					rho[phiidx, muidx, ridx] = 0.6 * data[5 + nLTE, ridx, muidx, phiidx] / (6.022 * 10 ** (23))
					ion_frac[phiidx, muidx, ridx] = data[29, ridx, muidx, phiidx]
					num_dens[phiidx, muidx, ridx] = data[5 + nLTE, ridx, muidx, phiidx]
				# x[phiidx, muidx, ridx] = r * np.sin(np.arccos(costheta)) * np.sin(phi)
				# y[phiidx, muidx, ridx] = -r * np.sin(np.arccos(costheta)) * np.cos(phi)
				# z[phiidx, muidx, ridx] = r * costheta
				x[phiidx, muidx, ridx] = r * np.sin(np.arccos(costheta)) * np.cos(phi)
				y[phiidx, muidx, ridx] = r * np.sin(np.arccos(costheta)) * np.sin(phi)
				z[phiidx, muidx, ridx] = r * costheta
	return rgrid,mugrid,phigrid,x,y,z,temp,rho,ion_frac, num_dens
