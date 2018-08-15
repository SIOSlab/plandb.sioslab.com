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

%pylab --no-import-all


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

