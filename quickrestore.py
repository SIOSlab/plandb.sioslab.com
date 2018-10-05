import requests
import pandas
from StringIO import StringIO
import astropy.units as u
import astropy.constants as const
import EXOSIMS.PlanetPhysicalModel.Forecaster
from sqlalchemy import create_engine
import getpass,keyring
import numpy as np
import os
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
import sqlalchemy.types 
import re
import scipy.integrate
import scipy.interpolate as interpolate
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.deltaMag import deltaMag
import EXOSIMS.Prototypes.PlanetPhysicalModel
from astropy.time import Time

%pylab --no-import-all
t0 = Time('2026-01-01T00:00:00', format='isot', scale='utc')


data = pandas.read_pickle('data3_080718.pkl')

tmp = np.load('allphotdata_2015.npz')
allphotdata = tmp['allphotdata']
clouds = tmp['clouds']
cloudstr = tmp['cloudstr']
wavelns = tmp['wavelns']
betas = tmp['betas']
dists = tmp['dists']
metallicities = tmp['metallicities']
def makeninterp(vals):
    ii =  interp1d(vals,vals,kind='nearest',bounds_error=False,fill_value=(vals.min(),vals.max()))
    return ii

distinterp = makeninterp(dists)
betainterp = makeninterp(betas)
feinterp = makeninterp(metallicities)
cloudinterp = makeninterp(clouds)


photinterps2 = {}
quadinterps = {}
for i,fe in enumerate(metallicities):
    photinterps2[fe] = {}
    quadinterps[fe] = {}
    for j,d in enumerate(dists):
        photinterps2[fe][d] = {}
        quadinterps[fe][d] = {}
        for k,cloud in enumerate(clouds):
            if np.any(np.isnan(allphotdata[i,j,k,:,:])):
                #remove whole rows of betas
                goodbetas = np.array(list(set(range(len(betas))) - set(np.unique(np.where(np.isnan(allphotdata[i,j,k,:,:]))[0]))))
                photinterps2[fe][d][cloud] = RectBivariateSpline(betas[goodbetas],wavelns,allphotdata[i,j,k,goodbetas,:])
                #photinterps2[fe][d][cloud] = interp2d(betas[goodbetas],wavelns,allphotdata[i,j,k,goodbetas,:].transpose(),kind='cubic')
            else:
                #photinterps2[fe][d][cloud] = interp2d(betas,wavelns,allphotdata[i,j,k,:,:].transpose(),kind='cubic')
                photinterps2[fe][d][cloud] = RectBivariateSpline(betas,wavelns,allphotdata[i,j,k,:,:])
            quadinterps[fe][d][cloud] = interp1d(wavelns,allphotdata[i,j,k,9,:].flatten())



orbdata = pandas.read_pickle('orbdata2_080718.pkl') 

lambdas = [575,  660, 730, 760, 825] #nm
bps = [10,18,18,18,10] #percent
bands = []
bandws = []
bandwsteps = []

for lam,bp in zip(lambdas,bps):
    band = np.array([-1,1])*float(lam)/1000.*bp/200.0 + lam/1000.
    bands.append(band)
    [ws,wstep] = np.linspace(band[0],band[1],100,retstep=True)
    bandws.append(ws)
    bandwsteps.append(wstep)

bands = np.vstack(bands) #um
bws = np.diff(bands,1).flatten() #um
bandws = np.vstack(bandws)
bandwsteps = np.array(bandwsteps)



plannames = data['pl_name'].values


def RfromM(m):
    m = np.array(m,ndmin=1)
    R = np.zeros(m.shape)


    S = np.array([0.2790,0,0,0,0.881])
    C = np.array([np.log10(1.008), 0, 0, 0, 0])
    T = np.array([2.04,95.16,(u.M_jupiter).to(u.M_earth),((0.0800*u.M_sun).to(u.M_earth)).value])

    Rj = u.R_jupiter.to(u.R_earth)
    Rs = 8.522 #saturn radius

    S[1] = (np.log10(Rs) - (C[0] + np.log10(T[0])*S[0]))/(np.log10(T[1]) - np.log10(T[0]))
    C[1] = np.log10(Rs) - np.log10(T[1])*S[1]

    S[2] = (np.log10(Rj) - np.log10(Rs))/(np.log10(T[2]) - np.log10(T[1]))
    C[2] = np.log10(Rj) - np.log10(T[2])*S[2]

    C[3] = np.log10(Rj)

    C[4] = np.log10(Rj) - np.log10(T[3])*S[4]


    inds = np.digitize(m,np.hstack((0,T,np.inf)))
    for j in range(1,inds.max()+1):
        R[inds == j] = 10.**(C[j-1] + np.log10(m[inds == j])*S[j-1])

    return R


altorbdata = pandas.read_pickle('altorbdata_080718.pkl') 

out2 = pandas.read_pickle('completeness2_080718.pkl')
tmp = np.load('completeness2_080718.npz')
goodinds = tmp['goodinds']
minCdMag = tmp['minCdMag']
maxCWA = tmp['maxCWA']
minCWA = tmp['minCWA']
maxCdMag = tmp['maxCdMag']
cs = tmp['cs']


out3 = pandas.read_pickle('aliases_080718.pkl') 


