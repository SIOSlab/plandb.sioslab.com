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
from getDataFromIPAC_extended import substitute_data

%pylab --no-import-all


t0 = Time('2026-01-01T00:00:00', format='isot', scale='utc')

#grab the data
query = """https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=compositepars&select=*&format=csv"""
r = requests.get(query)
data = pandas.read_csv(StringIO(r.content))

query2 = """https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=*&format=csv"""
r2 = requests.get(query2)
data2 = pandas.read_csv(StringIO(r2.content))

#strip leading 'f' on data colnames
colmap = {k:k[1:] if (k.startswith('fst_') | k.startswith('fpl_')) else k for k in data.keys()}
data = data.rename(columns=colmap)
#sma, eccen, metallicity cols were renamed so name them back for merge
data = data.rename(columns={'pl_smax':'pl_orbsmax',
                            'pl_smaxerr1':'pl_orbsmaxerr1',
                            'pl_smaxerr2':'pl_orbsmaxerr2',
                            'pl_smaxlim':'pl_orbsmaxlim',
                            'pl_smaxreflink':'pl_orbsmaxreflink',
                            'pl_eccen':'pl_orbeccen',
                            'pl_eccenerr1':'pl_orbeccenerr1',
                            'pl_eccenerr2':'pl_orbeccenerr2',
                            'pl_eccenlim':'pl_orbeccenlim',
                            'pl_eccenreflink':'pl_orbeccenreflink',
                            'st_met':'st_metfe',
                            'st_meterr1':'st_metfeerr1',
                            'st_meterr2':'st_metfeerr2',
                            'st_metreflink':'st_metfereflink',
                            'st_metlim':'st_metfelim',
                            })

#sort by planet name
data = data.sort_values(by=['pl_name']).reset_index(drop=True)
data2 = data2.sort_values(by=['pl_name']).reset_index(drop=True)

#merge data sets
data = data.combine_first(data2)

# substitute data from the extended table.
data = substitute_data(data)

#sort by planet name 
data = data.sort_values(by=['pl_name']).reset_index(drop=True)

###############################
#some sanity checking
# data3 = data.combine_first(data2)
#
# ccols = np.array(list(set(data.keys()) & set(data2.keys())))
# ncols = np.array(list(set(data2.keys()) - set(data.keys())))
#
# #compare redundant cols
# diffcs = []
# diffinds = []
# for c in ccols:
#     tmp = (data[c].values == data2[c].values) | (data[c].isnull().values & data2[c].isnull().values)
#     if not np.all( tmp ):
#         diffcs.append(c)
#         diffinds.append(np.where(~tmp)[0])
#
# for c,inds in zip(diffcs,diffinds):
#     print c
#     tmp = data[c][inds].isnull().values & ~(data2[c][inds].isnull().values)
#     assert np.all(data3[c][inds][tmp] == data2[c][inds][tmp])
###############################


## filter rows:
# we need:
# distance AND
# (sma OR (period AND stellar mass)) AND
# (radius OR mass (either true or m\sin(i)))
keep = ~np.isnan(data['st_dist'].values) & (~np.isnan(data['pl_orbsmax'].values) | \
        (~np.isnan(data['pl_orbper'].values) & ~np.isnan(data['st_mass'].values))) & \
       (~np.isnan(data['pl_bmassj'].values) | ~np.isnan(data['pl_radj'].values))
data = data[keep]
data = data.reset_index(drop=True)


##fill in missing smas from period & star mass
nosma = np.isnan(data['pl_orbsmax'].values)
p2sma = lambda mu,T: ((mu*T**2/(4*np.pi**2))**(1/3.)).to('AU')
GMs = const.G*(data['st_mass'][nosma].values*u.solMass) # units of solar mass
T = data['pl_orbper'][nosma].values*u.day
tmpsma = p2sma(GMs,T)
data['pl_orbsmax'][nosma] = tmpsma
data['pl_orbsmaxreflink'][nosma] = "Calculated from stellar mass and orbital period."

##update all WAs based on sma
WA = np.arctan((data['pl_orbsmax'].values*u.AU)/(data['st_dist'].values*u.pc)).to('mas')
data['pl_angsep'] = WA.value


###################################################################
#devel (skip)
#forecaster original
#S = np.array([0.2790,0.589,-0.044,0.881]) #orig coeffs
#C0 = np.log10(1.008)
#T = np.array([2.04,((0.414*u.M_jupiter).to(u.M_earth)).value,((0.0800*u.M_sun).to(u.M_earth)).value])
#C = np.hstack((C0, C0 + np.cumsum(-np.diff(S)*np.log10(T))))

#modify neptune and jupiter leg with new transition point at saturn mass and then flat leg past jupiter mass
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

##forecaster sanity check:
m1 = np.array([1e-3,T[0]])
r1 = 10.**(C[0] + np.log10(m1)*S[0])

m2 = T[0:2]
r2 = 10.**(C[1] + np.log10(m2)*S[1])

m3 = T[1:3]
r3 = 10.**(C[2] + np.log10(m3)*S[2])

m4 = T[2:4]
r4 = 10.**(C[3] + np.log10(m4)*S[3])

m5 = np.array([T[3],1e6])
r5 = 10.**(C[4] + np.log10(m5)*S[4])

plt.figure()
plt.loglog(m1,r1)
plt.loglog(m2,r2)
plt.loglog(m3,r3)
plt.loglog(m4,r4)
plt.loglog(m5,r5)
plt.xlabel('Mass ($M_\oplus$)')
plt.ylabel('Radius ($R_\oplus$)')
plt.loglog(m,Rf,'.',zorder=0)

##################################################################


#drop all other radius columns
data = data.drop(columns=['pl_rade',  'pl_radelim',  'pl_radserr2', 'pl_radeerr1', 'pl_rads', 'pl_radslim', 'pl_radeerr2', 'pl_radserr1']) 

#fill in radius based on mass
noR = ((data['pl_radreflink'] == '<a refstr="CALCULATED VALUE" href="/docs/composite_calc.html" target=_blank>Calculated Value</a>') |\
        data['pl_radj'].isnull()).values

m = ((data['pl_bmassj'][noR].values*u.M_jupiter).to(u.M_earth)).value

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

R = RfromM(m)

#create mod forecaster radius column
data = data.assign(pl_radj_forecastermod=data['pl_radj'].values)
data['pl_radj_forecastermod'][noR] = ((R*u.R_earth).to(u.R_jupiter)).value
     

## now the Fortney model
from EXOSIMS.PlanetPhysicalModel.FortneyMarleyCahoyMix1 import FortneyMarleyCahoyMix1
fortney = FortneyMarleyCahoyMix1()

ml10 = m <= 17
Rf = np.zeros(m.shape)
Rf[ml10] = fortney.R_ri(0.67,m[ml10])

mg10 = m > 17
tmpsmas = data['pl_orbsmax'][noR].values
tmpsmas = tmpsmas[mg10]
tmpsmas[tmpsmas < fortney.giant_pts2[:,1].min()] = fortney.giant_pts2[:,1].min()
tmpsmas[tmpsmas > fortney.giant_pts2[:,1].max()] = fortney.giant_pts2[:,1].max()

tmpmass = m[mg10]
tmpmass[tmpmass > fortney.giant_pts2[:,2].max()] = fortney.giant_pts2[:,2].max()

Rf[mg10] = interpolate.griddata(fortney.giant_pts2, fortney.giant_vals2,( np.array([10.]*np.where(mg10)[0].size), tmpsmas, tmpmass))

data = data.assign(pl_radj_fortney=data['pl_radj'].values)
data['pl_radj_fortney'][noR] = ((Rf*u.R_earth).to(u.R_jupiter)).value


#######
#quick fig for docs
plt.figure()
plt.plot(R,Rf,'.')
plt.plot([0,12],[0,12])
plt.xlim([0,12])
plt.ylim([0,12])
plt.xlabel('Modified Forecster Fit ($R_\oplus$)')
plt.ylabel('Fortney et al. (2007) Fit ($R_\oplus$)')
#######


##populate max WA based on available eccentricity data (otherwise maxWA = WA)
hase = ~np.isnan(data['pl_orbeccen'].values)
maxWA = WA[:]
maxWA[hase] = np.arctan((data['pl_orbsmax'][hase].values*(1 + data['pl_orbeccen'][hase].values)*u.AU)/(data['st_dist'][hase].values*u.pc)).to('mas')
data = data.assign(pl_maxangsep=maxWA.value)

#populate min WA based on eccentricity & inclination data (otherwise minWA = WA)
hasI =  ~np.isnan(data['pl_orbincl'].values)
s = data['pl_orbsmax'].values*u.AU
s[hase] *= (1 - data['pl_orbeccen'][hase].values)
s[hasI] *= np.cos(data['pl_orbincl'][hasI].values*u.deg)
s[~hasI] = 0
minWA = np.arctan(s/(data['st_dist'].values*u.pc)).to('mas')
data = data.assign(pl_minangsep=minWA.value)


#data.to_pickle('data_062818.pkl')
##############################
##restore from disk:
data = pandas.read_pickle('data_080718.pkl')


##############################################################################################################################
# grab photometry data 
#enginel = create_engine('sqlite:///' + os.path.join(os.getenv('HOME'),'Documents','AFTA-Coronagraph','ColorFun','AlbedoModels.db'))
enginel = create_engine('sqlite:///' + os.path.join(os.getenv('HOME'),'Documents','AFTA-Coronagraph','ColorFun','AlbedoModels_2015.db'))

# getting values
meta_alb = pandas.read_sql_table('header',enginel)
metallicities = meta_alb.metallicity.unique()
metallicities.sort()
betas = meta_alb.phase.unique()
betas.sort()
dists = meta_alb.distance.unique()
dists.sort()
clouds = meta_alb.cloud.unique()
clouds.sort()
cloudstr = clouds.astype(str)
for j in range(len(cloudstr)):
    cloudstr[j] = 'f'+cloudstr[j]
cloudstr[cloudstr == 'f0.0'] = 'NC'
cloudstr[cloudstr == 'f1.0'] = 'f1'
cloudstr[cloudstr == 'f3.0'] = 'f3'
cloudstr[cloudstr == 'f6.0'] = 'f6'

tmp = pandas.read_sql_table('g25_t150_m0.0_d0.5_NC_phang000',enginel)
wavelns = tmp.WAVELN.values

##################
#unnecessary if pulling all phot data
photdata550 = np.zeros((meta_alb.metallicity.unique().size,meta_alb.distance.unique().size, meta_alb.phase.unique().size))
for i,fe in enumerate(meta_alb.metallicity.unique()):
    basename = 'g25_t150_m'+str(fe)+'_d'
    print(basename)
    for j,d in enumerate(meta_alb.distance.unique()):
        for k,beta in enumerate(meta_alb.phase.unique()):
            name = basename+str(d)+'_NC_phang'+"%03d"%beta
            try:
                tmp = pandas.read_sql_table(name,enginel)
            except:
                photdata550[i,j,k] = np.nan
                continue
            ind = np.argmin(np.abs(tmp['WAVELN']-0.550))
            pval = tmp['GEOMALB'][ind]
            photdata550[i,j,k] = pval

photinterps = {}
for i,fe in enumerate(meta_alb.metallicity.unique()):
    photinterps[fe] = {}
    for j,d in enumerate(meta_alb.distance.unique()):
        photinterps[fe][d] = interp1d(betas[np.isfinite(photdata550[i,j,:])],photdata550[i,j,:][np.isfinite(photdata550[i,j,:])],kind='cubic')
#################

allphotdata = np.zeros((metallicities.size, dists.size, clouds.size, betas.size, wavelns.size))
for i,fe in enumerate(metallicities):
    basename = 'g25_t150_m'+str(fe)+'_d'
    for j,d in enumerate(dists):
        basename2 = basename+str(d)+'_'
        for k,cloud in enumerate(clouds):
            basename3 = basename2+cloudstr[k]+'_phang'
            print(basename3)
            for l,beta in enumerate(betas):
                name = basename3+"%03d"%beta
                try:
                    tmp = pandas.read_sql_table(name,enginel)
                except:
                    print("Missing: %s"%name)
                    allphotdata[i,j,k,l,:] = np.nan
                    continue
                pvals = tmp['GEOMALB'].values
                if len(tmp) != len(wavelns):
                    missing = list(set(wavelns) - set(tmp.WAVELN.values))
                    inds  = np.searchsorted(tmp['WAVELN'].values,missing)
                    pvals = np.insert(pvals,inds,np.nan)
                    assert np.isnan(pvals[wavelns==missing[0]])
                    print("Filled value: %s, %s"%(name,missing))
                allphotdata[i,j,k,l,:] = pvals



#patch individual nans
for i,fe in enumerate(metallicities):
    for j,d in enumerate(dists):
        for k,cloud in enumerate(clouds):
            for l,beta in enumerate(betas):
                nans = np.isnan(allphotdata[i,j,k,l,:])
                if np.any(nans) & ~np.all(nans):
                    tmp = interp1d(wavelns[~nans],allphotdata[i,j,k,l,~nans],kind='cubic')
                    allphotdata[i,j,k,l,nans] = tmp(wavelns[nans])


##np.savez('allphotdata',metallicities=metallicities,dists=dists,clouds=clouds,cloudstr=cloudstr,betas=betas,wavelns=wavelns,allphotdata=allphotdata)
#np.savez('allphotdata_2015',metallicities=metallicities,dists=dists,clouds=clouds,cloudstr=cloudstr,betas=betas,wavelns=wavelns,allphotdata=allphotdata)


#######
# visualization:
wind = np.argmin(np.abs(wavelns - 0.575))
dind = np.argmin(np.abs(dists - 1))


wind = np.argmin(np.abs(wavelns - 0.825))
dind = np.argmin(np.abs(dists - 5))


ls = ["-","--","-.",":","o-","s-","d-","h-"]

plt.figure()
for j in range(clouds.size):
    plt.plot(betas,allphotdata[0,dind,j,:,wind],ls[j],label=cloudstr[j])

plt.ylabel('$p\Phi(\\beta)$')
plt.xlabel('Phase (deg)')
plt.xlim([0,180])
plt.legend()
plt.title('Phase Curves for %4.4f $\mu$m at %3.1f AU'%(wavelns[wind],dists[dind]))

########
#restore photdata fromdisk
#tmp = np.load('allphotdata.npz')
tmp = np.load('allphotdata_2015.npz')
allphotdata = tmp['allphotdata']
clouds = tmp['clouds']
cloudstr = tmp['cloudstr']
wavelns = tmp['wavelns']
betas = tmp['betas']
dists = tmp['dists']
metallicities = tmp['metallicities']
#########


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



##############################################################################################################################

## quadrature columns
#wavelengths of interest
#lambdas = np.array([575, 635, 660, 706, 760, 825])
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

smas = data['pl_orbsmax'].values
fes = data['st_metfe'].values
fes[np.isnan(fes)] = 0.0
Rps = data['pl_radj_forecastermod'].values
inc = data['pl_orbincl'].values
eccen = data['pl_orbeccen'].values
arg_per = data['pl_orblper'].values

tmpout = {}
for c in clouds:
    for l in lambdas:
        tmpout['quad_pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"] = np.zeros(smas.shape)
        tmpout['quad_dMag_'+"%03dC_"%(c*100)+str(l)+"NM"] = np.zeros(smas.shape)
        tmpout['quad_radius_' + "%03dC_" % (c * 100) + str(l) + "NM"] = np.zeros(smas.shape)


for j, (Rp, fe,a, I, e, w) in enumerate(zip(Rps, fes,smas, inc, eccen, arg_per)):
    print(j)
    for c in clouds:
        for l,band,bw,ws,wstep in zip(lambdas,bands,bws,bandws,bandwsteps):
            #pphi = photinterps2[float(feinterp(fe))][float(distinterp(a))][c](90.0,float(l)/1000.).flatten()
            #pphi = scipy.integrate.quad(quadinterps[float(feinterp(fe))][float(distinterp(a))][c],band[0],band[1])[0]/bw

            #Only calc quadrature distance if known eccentricity and argument of periaps, and not face-on orbit
            if not np.isnan(e) and not np.isnan(w) and I != 0:
                nu1 = -w
                nu2 = np.pi - w

                r1 = a * (1.0 - e ** 2.0) / (1.0 + e * np.cos(nu1))
                r2 = a * (1.0 - e ** 2.0) / (1.0 + e * np.cos(nu2))

                pphi1 = quadinterps[float(feinterp(fe))][float(distinterp(r1))][c](ws).sum() * wstep / bw
                pphi2 = quadinterps[float(feinterp(fe))][float(distinterp(r2))][c](ws).sum() * wstep / bw
                if np.isinf(pphi1):
                    print("Inf value encountered in pphi")
                    pphi1 = np.nan
                if np.isinf(pphi2):
                    print("Inf value encountered in pphi")
                    pphi2 = np.nan
                # pphi[np.isinf(pphi)] = np.nan
                # pphi = pphi[0]

                dMag1 = deltaMag(1, Rp * u.R_jupiter, r1 * u.AU, pphi1)
                dMag2 = deltaMag(1, Rp * u.R_jupiter, r2 * u.AU, pphi2)
                if np.isnan(dMag2) or dMag1 < dMag2:
                    dMag = dMag1
                    pphi = pphi1
                    r = r1
                else:
                    dMag = dMag2
                    pphi = pphi2
                    r = r2
                tmpout['quad_pPhi_' + "%03dC_" % (c * 100) + str(l) + "NM"][j] = pphi
                tmpout['quad_radius_' + "%03dC_" % (c * 100) + str(l) + "NM"][j] = r
            else:
                pphi = quadinterps[float(feinterp(fe))][float(distinterp(a))][c](ws).sum()*wstep/bw
                if np.isinf(pphi):
                    print("Inf value encountered in pphi")
                    pphi = np.nan
                #pphi[np.isinf(pphi)] = np.nan
                #pphi = pphi[0]
                tmpout['quad_pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"][j] = pphi
                dMag = deltaMag(1, Rp*u.R_jupiter, a*u.AU, pphi)
                tmpout['quad_radius_' + "%03dC_" % (c * 100) + str(l) + "NM"][j] = a
            if np.isinf(dMag): 
                print("Inf value encountered in dmag")
                dMag = np.nan
            tmpout['quad_dMag_'+"%03dC_"%(c*100)+str(l)+"NM"][j] = dMag



#collect min/max/med for every wavelength
for l in lambdas:
    tmp = []
    for c in clouds:
        tmp.append(tmpout['quad_dMag_'+"%03dC_"%(c*100)+str(l)+"NM"])
    tmp = np.vstack(tmp)
    tmpout["quad_dMag_min_"+str(l)+"NM"] = np.nanmin(tmp,axis=0)
    tmpout["quad_dMag_max_"+str(l)+"NM"] = np.nanmax(tmp,axis=0)
    tmpout["quad_dMag_med_"+str(l)+"NM"] = np.nanmedian(tmp,axis=0)

data = data.join(pandas.DataFrame(tmpout))


#data.to_pickle('data2_080718.pkl') #incorrect bw calculation
#data.to_pickle('data3_080718.pkl') #corrected bw calculation

##############################################################################################################################
#PPMod = EXOSIMS.Prototypes.PlanetPhysicalModel.PlanetPhysicalModel()

#orbit info
M = np.linspace(0,2*np.pi,100)
plannames = data['pl_name'].values

minWA = data['pl_minangsep'].values*u.mas
maxWA = data['pl_maxangsep'].values*u.mas

orbdata = None
#row = data.iloc[71] 
for j in range(len(plannames)):
    row = data.iloc[j] 

    a = row['pl_orbsmax']
    e = row['pl_orbeccen'] 
    if np.isnan(e): e = 0.0
    I = row['pl_orbincl']*np.pi/180.0
    if np.isnan(I): I = np.pi/2.0
    w = row['pl_orblper']*np.pi/180.0
    if np.isnan(w): w = 0.0
    E = eccanom(M, e)                      
    Rp = row['pl_radj_forecastermod']
    dist = row['st_dist']
    fe = row['st_metfe']
    if np.isnan(fe): fe = 0.0

#    a1 = np.cos(w) 
#    a2 = np.cos(I)*np.sin(w)
#    a3 = np.sin(I)*np.sin(w)
#    A = a*np.vstack((a1, a2, a3))
#
#    b1 = -np.sqrt(1 - e**2)*np.sin(w)
#    b2 = np.sqrt(1 - e**2)*np.cos(I)*np.cos(w)
#    b3 = np.sqrt(1 - e**2)*np.sin(I)*np.cos(w)
#    B = a*np.vstack((b1, b2, b3))
#    r1 = np.cos(E) - e
#    r2 = np.sin(E)
#
#    r = (A*r1 + B*r2).T
#    d = np.linalg.norm(r, axis=1)
#    s = np.linalg.norm(r[:,0:2], axis=1)
#    beta = np.arccos(r[:,2]/d)*u.rad

    nu = 2*np.arctan(np.sqrt((1.0 + e)/(1.0 - e))*np.tan(E/2.0));
    d = a*(1.0 - e**2.0)/(1 + e*np.cos(nu))
    s = d*np.sqrt(4.0*np.cos(2*I) + 4*np.cos(2*nu + 2.0*w) - 2.0*np.cos(-2*I + 2.0*nu + 2*w) - 2*np.cos(2*I + 2*nu + 2*w) + 12.0)/4.0
    beta = np.arccos(np.sin(I)*np.sin(nu+w))*u.rad


    WA = np.arctan((s*u.AU)/(dist*u.pc)).to('mas').value
    print(j,plannames[j],WA.min() - minWA[j].value, WA.max() - maxWA[j].value)

    outdict = {'Name': [plannames[j]]*len(M),
                'M': M,
                'r': d,
                's': s,
                'WA': WA,
                'beta': beta.to(u.deg).value}

    inds = np.argsort(beta)
    for c in clouds:
        for l,band,bw,ws,wstep in zip(lambdas,bands,bws,bandws,bandwsteps):
            #pphi = photinterps2[float(feinterp(fe))][float(distinterp(a))][c](beta.to(u.deg).value[inds],float(l)/1000.)[np.argsort(inds)].flatten()
            pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a))][c](beta.to(u.deg).value[inds],ws).sum(1)*wstep/bw)[np.argsort(inds)]
            pphi[np.isinf(pphi)] = np.nan
            outdict['pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"] = pphi 
            dMag = deltaMag(1, Rp*u.R_jupiter, d*u.AU, pphi)
            dMag[np.isinf(dMag)] = np.nan
            outdict['dMag_'+"%03dC_"%(c*100)+str(l)+"NM"] = dMag

    #pphi = np.array([ photinterps[float(feinterp(fe))][float(distinterp(di))](bi) for di,bi in zip(d,beta.to(u.deg).value) ])
    #dMag = deltaMag(1, Rp*u.R_jupiter, d*u.AU, pphi)

    #phi = PPMod.calc_Phi(np.arccos(r[:,2]/d)*u.rad) 
    #dMag = deltaMag(0.5, Rp*u.R_jupiter, d*u.AU, phi)


    out = pandas.DataFrame(outdict)
    
    if orbdata is None:
        orbdata = out.copy()
    else:
        orbdata = orbdata.append(out)


#orbdata.to_pickle('orbdata_080718.pkl') #incorrect bw calculation
#orbdata.to_pickle('orbdata2_080718.pkl')  #corrected bw calculation
#############################################################################################################################

##variable inclination orbits
plannames = data['pl_name'].values
Isglob = np.array([90,60,30])
(l,band,bw,ws,wstep) = (lambdas[0],bands[0],bws[0],bandws[0],bandwsteps[0])
c = 3.0

altorbdata = None
#row = data.iloc[71] 
for j in range(len(plannames)):
    row = data.iloc[j]
    print(j,plannames[j])

    if not np.isnan(row['pl_orbincl']):
        continue
    
    if row['pl_bmassprov'] == 'Msini':
            Icrit = np.arcsin( ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value/((0.0800*u.M_sun).to(u.M_earth)).value )
    else:
        Icrit = 10*np.pi/180.0

    Is = np.hstack((Isglob*np.pi/180.0,Icrit))

    a = row['pl_orbsmax']
    e = row['pl_orbeccen'] 
    if np.isnan(e): e = 0.0
    w = row['pl_orblper']*np.pi/180.0
    if np.isnan(w): w = 0.0                    
    Rp = row['pl_radj_forecastermod']
    dist = row['st_dist']
    fe = row['st_metfe']
    if np.isnan(fe): fe = 0.0

    Tp = row['pl_orbper'] #days
    Mstar = row['st_mass'] #solar masses
    taup = row['pl_orbtper']

    if (np.isnan(Tp) or Tp == 0.0) and np.isnan(Mstar):
        print("No period or star mass for: %s")%(plannames[j])
        continue

    mu = const.G*(Mstar*u.solMass).decompose()
    if np.isnan(Tp) or (Tp == 0.0):
        Tp = (2*np.pi*np.sqrt(((a*u.AU)**3.0)/mu)).decompose().to(u.d).value

    if Tp > 10*365.25:
        print("Too long period for: %s")%(plannames[j])
        continue

    if np.isnan(mu):
        mu = ( (a*u.AU)**3.0 * (2*np.pi/(Tp*u.d))**2. ).decompose() 

    n = 2*np.pi/Tp

    #M = np.arange(0,Tp,30)*n
    ttmp = np.arange(t0.jd,t0.jd+Tp,30)
    M = np.mod(ttmp*n,2*np.pi)


    E = eccanom(M, e)  
    nu = 2*np.arctan(np.sqrt((1.0 + e)/(1.0 - e))*np.tan(E/2.0));
    d0 = a*(1.0 - e**2.0)/(1 + e*np.cos(nu))

    # a1 = np.cos(w)
    # b1 = -np.sqrt(1 - e**2)*np.sin(w)

    outdict = {'Name': [plannames[j]]*len(M),
                'M': M,
                'r': d0,
                'Icrit': [Icrit]*len(M)
               }


    for k,I in enumerate(Is):
    
        # a2 = np.cos(I)*np.sin(w)
        # a3 = np.sin(I)*np.sin(w)
        # A = a*np.vstack((a1, a2, a3))
        #
        # b2 = np.sqrt(1 - e**2)*np.cos(I)*np.cos(w)
        # b3 = np.sqrt(1 - e**2)*np.sin(I)*np.cos(w)
        # B = a*np.vstack((b1, b2, b3))
        # r1 = np.cos(E) - e
        # r2 = np.sin(E)
        #
        # r = (A*r1 + B*r2).T
        # d = np.linalg.norm(r, axis=1)
        # s = np.linalg.norm(r[:,0:2], axis=1)
        # beta = np.arccos(r[:,2]/d)*u.rad

        nu = 2 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E / 2.0));
        d = a * (1.0 - e ** 2.0) / (1 + e * np.cos(nu))
        s = d * np.sqrt(4.0 * np.cos(2 * I) + 4 * np.cos(2 * nu + 2.0 * w) - 2.0 * np.cos(-2 * I + 2.0 * nu + 2 * w) - 2 * np.cos(2 * I + 2 * nu + 2 * w) + 12.0) / 4.0
        beta = np.arccos(np.sin(I) * np.sin(nu + w)) * u.rad

        WA = np.arctan((s*u.AU)/(dist*u.pc)).to('mas').value

        if I == Icrit:
            Itag = "crit"
        else:
            Itag = "%02d"%(Isglob[k])

        outdict["s_I"+Itag] = s
        outdict["WA_I"+Itag] =  WA
        outdict["beta_I"+Itag] = beta.to(u.deg).value

        inds = np.argsort(beta)
        pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a))][c](beta.to(u.deg).value[inds],ws).sum(1)*wstep/bw)[np.argsort(inds)]
        pphi[np.isinf(pphi)] = np.nan
        outdict['pPhi_'+"%03dC_"%(c*100)+str(l)+"NM_I"+Itag] = pphi 
        dMag = deltaMag(1, Rp*u.R_jupiter, d*u.AU, pphi)
        dMag[np.isinf(dMag)] = np.nan
        outdict['dMag_'+"%03dC_"%(c*100)+str(l)+"NM_I"+Itag] = dMag


    out = pandas.DataFrame(outdict)
    
    if altorbdata is None:
        altorbdata = out.copy()
    else:
        altorbdata = altorbdata.append(out)


#altorbdata.to_pickle('altorbdata_080718.pkl')



#############################################################################################################################

#from Mark: 
#f_sed             Frequency
#0.000000         0.099
#0.010000         0.001
#0.030000         0.005
#0.100000         0.010
#0.300000         0.025
#1.000000         0.280
#3.000000         0.300
#6.000000         0.280

# Generates fsed based on a random number: 0 <= num < 1
def get_fsed(num):
    if num < .099:
        r = 0
    elif num < .1:
        r = .01
    elif num < .105:
        r = .03
    elif num < .115:
        r = .1
    elif num < .14:
        r = .3
    elif num < .42:
        r = 1
    elif num < .72:
        r = 3
    else:
        r = 6
    return float(r)


wfirstcontr = np.genfromtxt('WFIRST_pred_imaging.txt')
contr = wfirstcontr[:,1]
angsep = wfirstcontr[:,0] #l/D
angsep = (angsep * (575.0*u.nm)/(2.37*u.m)*u.rad).decompose().to(u.mas).value #mas
wfirstc = interp1d(angsep,contr,bounds_error = False, fill_value = 'extrapolate')


## completeness calculation
minangsep = 150
maxangsep = 450

inds = np.where((data['pl_maxangsep'].values > minangsep) & (data['pl_minangsep'].values < maxangsep))[0]

WAbins0 = np.arange(minangsep,maxangsep+1,1)
WAbins = np.hstack((0, WAbins0, np.inf))
dMagbins0 = np.arange(0,26.1,0.1)
dMagbins = np.hstack((dMagbins0,np.inf))

WAc,dMagc = np.meshgrid(WAbins0[:-1]+np.diff(WAbins0)/2.0,dMagbins0[:-1]+np.diff(dMagbins0)/2.0)
WAc = WAc.T
dMagc = dMagc.T

WAinds = np.arange(WAbins0.size-1)
dMaginds = np.arange(dMagbins0.size-1)
WAinds,dMaginds = np.meshgrid(WAinds,dMaginds)
WAinds = WAinds.T
dMaginds = dMaginds.T

dMaglimsc = wfirstc(WAc[:,0])

names = []
WAcs = []
dMagcs = []
iinds = []
jinds = []
hs = []
cs = []
goodinds = []
for j in inds:
    row = data.iloc[j] 
    print(j, row['pl_name'])
    
    amu = row['pl_orbsmax']
    astd = (row['pl_orbsmaxerr1'] - row['pl_orbsmaxerr2'])/2.
    if np.isnan(astd): astd = 0.01*amu
    gena = lambda n: np.clip(np.random.randn(n)*astd + amu,0,np.inf)

    emu = row['pl_orbeccen'] 
    if np.isnan(emu):
        gene = lambda n: 0.175/np.sqrt(np.pi/2.)*np.sqrt(-2.*np.log(1 - np.random.uniform(size=n)))
    else:
        estd = (row['pl_orbeccenerr1'] - row['pl_orbeccenerr2'])/2.
        if np.isnan(estd) or (estd == 0):
            estd = 0.01*emu
        gene = lambda n: np.clip(np.random.randn(n)*estd + emu,0,0.99)

    Imu = row['pl_orbincl']*np.pi/180.0
    if np.isnan(Imu):
        if row['pl_bmassprov'] == 'Msini':
            Icrit = np.arcsin( ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value/((0.0800*u.M_sun).to(u.M_earth)).value )
            Irange = [Icrit, np.pi - Icrit]
            C = 0.5*(np.cos(Irange[0])-np.cos(Irange[1]))
            genI = lambda n: np.arccos(np.cos(Irange[0]) - 2.*C*np.random.uniform(size=n))

        else:
            genI = lambda n: np.arccos(1 - 2.*np.random.uniform(size=n))
    else:
        Istd = (row['pl_orbinclerr1'] - row['pl_orbinclerr2'])/2.*np.pi/180.0
        if np.isnan(Istd) or (Istd == 0): 
            Istd = Imu*0.01
        genI = lambda n: np.random.randn(n)*Istd + Imu
    

    wbarmu = row['pl_orblper']*np.pi/180.0
    if np.isnan(wbarmu):
        genwbar = lambda n: np.random.uniform(size=n,low=0.0,high=2*np.pi)
    else:
        wbarstd = (row['pl_orblpererr1'] - row['pl_orblpererr2'])/2.*np.pi/180.0
        if np.isnan(wbarstd) or (wbarstd == 0): 
            wbarstd = wbarmu*0.01
        genwbar = lambda n: np.random.randn(n)*wbarstd + wbarmu

    fe = row['st_metfe']
    if np.isnan(fe): fe = 0.0


    n = int(1e6)
    c = 0.
    h = np.zeros((len(WAbins)-3, len(dMagbins)-2))
    k = 0.0
    cprev = 0.0
    pdiff = 1.0

    while (pdiff > 0.0001) | (k <3):
    #for blah in range(100):
        print("%d \t %5.5e \t %5.5e"%( k,pdiff,c))
        a = gena(n)
        e = gene(n)
        I = genI(n)
        O = np.random.uniform(size=n,low=0.0,high=2*np.pi)
        wbar = genwbar(n)
        w = O - wbar

        # cl = cloudinterp(np.random.randn(n)*2 + 3)
        vget_fsed = np.vectorize(get_fsed)
        cl = vget_fsed(np.random.rand(n))

        if (row['pl_radreflink'] == '<a refstr="CALCULATED VALUE" href="/docs/composite_calc.html" target=_blank>Calculated Value</a>'):
            if row['pl_bmassprov'] == 'Msini':
                Mp = ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value
                Mp = Mp/np.sin(I)
            else:
                Mstd = (((row['pl_bmassjerr1'] - row['pl_bmassjerr2'])*u.M_jupiter).to(u.M_earth)).value
                if np.isnan(Mstd):
                    Mstd = ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value * 0.1
                Mp = np.random.randn(n)*Mstd + ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value

            R = (RfromM(Mp)*u.R_earth).to(u.R_jupiter).value
            R[R > 1.0] = 1.0
        else:
            Rmu = row['pl_radj']
            Rstd = (row['pl_radjerr1'] - row['pl_radjerr2'])/2.
            if np.isnan(Rstd): Rstd = Rmu*0.1
            R = np.random.randn(n)*Rstd + Rmu
        
        M0 = np.random.uniform(size=n,low=0.0,high=2*np.pi)
        E = eccanom(M0, e)
        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

        # a1 = np.cos(O)*np.cos(w) - np.sin(O)*np.cos(I)*np.sin(w)
        # a2 = np.sin(O)*np.cos(w) + np.cos(O)*np.cos(I)*np.sin(w)
        # a3 = np.sin(I)*np.sin(w)
        # A = a*np.vstack((a1, a2, a3))
        # b1 = -np.sqrt(1 - e**2)*(np.cos(O)*np.sin(w) + np.sin(O)*np.cos(I)*np.cos(w))
        # b2 = np.sqrt(1 - e**2)*(-np.sin(O)*np.sin(w) + np.cos(O)*np.cos(I)*np.cos(w))
        # b3 = np.sqrt(1 - e**2)*np.sin(I)*np.cos(w)
        # B = a*np.vstack((b1, b2, b3))
        # r1 = np.cos(E) - e
        # r2 = np.sin(E)
        #
        # rvec = (A*r1 + B*r2).T
        # rnorm = np.linalg.norm(rvec, axis=1)
        # s = np.linalg.norm(rvec[:,0:2], axis=1)
        # beta = np.arccos(rvec[:,2]/rnorm)*u.rad

        d = a * (1.0 - e ** 2.0) / (1 + e * np.cos(nu))
        s = d * np.sqrt(4.0 * np.cos(2 * I) + 4 * np.cos(2 * nu + 2.0 * w) - 2.0 * np.cos(-2 * I + 2.0 * nu + 2 * w) - 2 * np.cos(2 * I + 2 * nu + 2 * w) + 12.0) / 4.0
        beta = np.arccos(np.sin(I) * np.sin(nu + w)) * u.rad
        rnorm = d

        #phi = PPMod.calc_Phi(np.arccos(rvec[:,2]/rnorm)*u.rad)    # planet phase
        #dMag = deltaMag(0.5, R*u.R_jupiter, rnorm*u.AU, phi)     # delta magnitude
        #pphi = photinterps[float(feinterp(fe))][float(distinterp(np.mean(rnorm)))](beta.to(u.deg).value)
                    
        pphi = np.zeros(n)
        for clevel in np.unique(cl):
            tmpinds = cl == clevel
            betatmp = beta[tmpinds]
            binds = np.argsort(betatmp)
            pphi[tmpinds] = photinterps2[float(feinterp(fe))][float(distinterp(np.mean(rnorm)))][clevel](betatmp.to(u.deg).value[binds],575./1000.)[np.argsort(binds)].flatten()
        
        pphi[pphi <= 0.0] = 1e-16

        #binds = np.argsort(beta)
        #pphi = photinterps2[float(feinterp(fe))][float(distinterp(np.mean(rnorm)))][0.0](beta.to(u.deg).value[binds],575./1000.)[np.argsort(binds)].flatten()

        dMag = deltaMag(1, R*u.R_jupiter, rnorm*u.AU, pphi)

        WA = np.arctan((s*u.AU)/(row['st_dist']*u.pc)).to('mas').value # working angle

        h += np.histogram2d(WA,dMag,bins=(WAbins,dMagbins))[0][1:-1,0:-1]
        k += 1.0

        dMaglimtmp = -2.5*np.log10(wfirstc(WA))
        currc = float(len(np.where((WA >= minangsep) & (WA <= maxangsep) & (dMag <= dMaglimtmp))[0]))/n

        #currc = float(len(np.where((WA >= minangsep) & (WA <= maxangsep) & (dMag <= 22.5))[0]))/n
        cprev = c
        if k == 1.0:
            c = currc
        else:
            c = ((k-1)*c + currc)/k
        if c == 0:
            pdiff = 1.0
        else:
            pdiff = np.abs(c - cprev)/c

        if (c == 0.0) & (k > 2):
            break

        if (c < 1e-5) & (k > 25):
            break

    if c != 0.0:
        h = h/float(n*k)
        names.append(np.array([row['pl_name']]*h.size))
        WAcs.append(WAc.flatten())
        dMagcs.append(dMagc.flatten())
        hs.append(h.flatten())
        iinds.append(WAinds.flatten())
        jinds.append(dMaginds.flatten())
        cs.append(c)
        goodinds.append(j)

    print("\n\n\n\n")

cs = np.array(cs)
goodinds = np.array(goodinds)


out2 = pandas.DataFrame({'Name': np.hstack(names),
                         'alpha': np.hstack(WAcs),
                         'dMag': np.hstack(dMagcs),
                         'H':    np.hstack(hs),
                         'iind': np.hstack(iinds),
                         'jind': np.hstack(jinds)
                         })
out2 = out2[out2['H'].values != 0.]
out2['H'] = np.log10(out2['H'].values)



minCWA = []
maxCWA = []
minCdMag = []
maxCdMag = []

for j in range(len(goodinds)):
    minCWA.append(np.floor(np.min(WAcs[j][hs[j] != 0])))
    maxCWA.append(np.ceil(np.max(WAcs[j][hs[j] != 0])))
    minCdMag.append(np.floor(np.min(dMagcs[j][hs[j] != 0])))
    maxCdMag.append(np.ceil(np.max(dMagcs[j][hs[j] != 0])))

#np.savez('completeness_080718',cs=cs,goodinds=goodinds,minCWA=minCWA,maxCWA=maxCWA,minCdMag=minCdMag,maxCdMag=maxCdMag)
#out2.to_pickle('completeness_080718.pkl')


#####
#restore
out2 = pandas.read_pickle('completeness2_080718.pkl')
tmp = np.load('completeness2_080718.npz')
goodinds = tmp['goodinds']
minCdMag = tmp['minCdMag']
maxCWA = tmp['maxCWA']
minCWA = tmp['minCWA']
maxCdMag = tmp['maxCdMag']
cs = tmp['cs']

###################################################################
#build alias table
from astroquery.simbad import Simbad

starnames = data['pl_hostname'].unique()

s = Simbad()
s.add_votable_fields('ids')
baseurl = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"

ids = []
aliases = []
noipacalias = []
nosimbadalias = []
badstars = []
priname = []
for j,star in enumerate(starnames):
    print(j,star)
    #get aliases from IPAC
    r = requests.get(baseurl,{'table':'aliastable','objname':star})
    if "ERROR" not in r.content: 
        tmp = r.content.strip().split("\n")
    else:
        noipacalias.append(star)
        tmp = [star]
    
    #get aliases from SIMBAD
    r = s.query_object(star)
    if r:
        tmp += r['IDS'][0].split('|')
    else:
        if (len(noipacalias) == 0) or (noipacalias[-1] != star):
            for t in tmp:
                if (t not in['aliasdis',star]):
                    r = s.query_object(t)
                    if r:
                        tmp += r['IDS'][0].split('|')
                        break
            if not r:
                nosimbadalias.append(star)
        else:
            nosimbadalias.append(star)

    #track stars with no records
    if (len(noipacalias) > 0) and (len(nosimbadalias) > 0) and (noipacalias[-1] == star) and (nosimbadalias[-1] == star):
        badstars.append(star)

    if star not in tmp: tmp.append(star)
    if 'aliasdis' in tmp: tmp.remove('aliasdis')
    tmp = list(np.unique(tmp))
        
    ids.append([j]*len(tmp))
    aliases.append(tmp)
    priname.append(list((np.array(tmp) == star).astype(int)))


#toggleoff = ['notesel','messel','bibsel','fluxsel','sizesel','mtsel','spsel','rvsel','pmsel','cooN','otypesel']
#url = """http://simbad.u-strasbg.fr/simbad/sim-id?output.format=ASCII&Ident=%s"""%starnames[j]
#for t in toggleoff:
#    url += "&obj.%s=off"%t


out3 = pandas.DataFrame({'SID': np.hstack(ids),
                         'Alias': np.hstack(aliases),
                         'NEAName':np.hstack(priname)
                         })
#out3.to_pickle('aliases_080718.pkl')

###################################################################


#------write to db------------
namemxchar = np.array([len(n) for n in plannames]).max()

#testdb
engine = create_engine('mysql+pymysql://ds264@127.0.0.1/dsavrans_plandb',echo=False)

#proddb#################################################################################################
username = 'dsavrans_admin'
passwd = keyring.get_password('plandb_sql_login', username)
if passwd is None:
    passwd = getpass.getpass("Password for mysql user %s:\n"%username)
    keyring.set_password('plandb_sql_login', username, passwd)

engine = create_engine('mysql+pymysql://'+username+':'+passwd+'@sioslab.com/dsavrans_plandb',echo=False)
#proddb#################################################################################################


##cleanup as necessary
result = engine.execute("DROP TABLE IF EXISTS PlanetOrbits")
result = engine.execute("DROP TABLE IF EXISTS Completeness")
result = engine.execute("DROP TABLE IF EXISTS AltPlanetOrbits")

result = engine.execute("UPDATE KnownPlanets SET completeness=NULL")
result = engine.execute("UPDATE KnownPlanets SET compMinWA=NULL")
result = engine.execute("UPDATE KnownPlanets SET compMaxWA=NULL")
result = engine.execute("UPDATE KnownPlanets SET compMindMag=NULL")
result = engine.execute("UPDATE KnownPlanets SET compMaxdMag=NULL")


##write KnownPlanets
data.to_sql('KnownPlanets',engine,chunksize=100,if_exists='replace',
        dtype={'pl_name':sqlalchemy.types.String(namemxchar),
               'pl_hostname':sqlalchemy.types.String(namemxchar-2),
               'pl_letter':sqlalchemy.types.CHAR(1)})
        
result = engine.execute("ALTER TABLE KnownPlanets ENGINE=InnoDB")
result = engine.execute("ALTER TABLE KnownPlanets ADD INDEX (pl_name)")
result = engine.execute("ALTER TABLE KnownPlanets ADD INDEX (pl_hostname)")
result = engine.execute("ALTER TABLE KnownPlanets ADD completeness double COMMENT 'completeness in 0.1 to 0.5 as bin'")
result = engine.execute("UPDATE KnownPlanets SET completeness=NULL where completeness is not NULL")
result = engine.execute("ALTER TABLE KnownPlanets ADD compMinWA double COMMENT 'min non-zero completeness WA'")
result = engine.execute("ALTER TABLE KnownPlanets ADD compMaxWA double COMMENT 'max non-zero completeness WA'")
result = engine.execute("ALTER TABLE KnownPlanets ADD compMindMag double COMMENT 'min non-zero completeness dMag'")
result = engine.execute("ALTER TABLE KnownPlanets ADD compMaxdMag double COMMENT 'max non-zero completeness dMag'")

for ind,c in zip(goodinds,cs):
    result = engine.execute("UPDATE KnownPlanets SET completeness=%f where pl_name = '%s'"%(c,plannames[ind]))

for ind,minw,maxw,mind,maxd in zip(goodinds,minCWA,maxCWA,minCdMag,maxCdMag):
    result = engine.execute("UPDATE KnownPlanets SET compMinWA=%f,compMaxWA=%f,compMindMag=%f,compMaxdMag=%f where pl_name = '%s'"%(minw,maxw,mind,maxd,plannames[ind]))


#add comments
coldefs = pandas.ExcelFile('coldefs.xlsx')
coldefs = coldefs.parse('Sheet1')
cols = coldefs['Column'][coldefs['Definition'].notnull()].values
cdefs = coldefs['Definition'][coldefs['Definition'].notnull()].values
cnames =  coldefs['Name'][coldefs['Definition'].notnull()].values


result = engine.execute("show create table KnownPlanets")
res = result.fetchall()
res = res[0]['Create Table']
res = res.split("\n")

p = re.compile('`(\S+)`[\s\S]+')
keys = []
defs = []
for r in res:
  r = r.strip().strip(',')
  if "COMMENT" in r: continue
  m = p.match(r)
  if m:
    keys.append(m.groups()[0])
    defs.append(r)

for key,d in zip(keys,defs):
  if not key in cols: continue
  comm =  """ALTER TABLE `KnownPlanets` CHANGE `%s` %s COMMENT "%s %s";"""%(key,d,cnames[cols == key][0].strip('"'),cdefs[cols == key][0])
  print comm
  r = engine.execute(comm)

#---------------------------------------------
#write planetorbits table
orbdata.to_sql('PlanetOrbits',engine,chunksize=100,if_exists='replace',dtype={'Name':sqlalchemy.types.String(namemxchar)})
result = engine.execute("ALTER TABLE PlanetOrbits ENGINE=InnoDB")
result = engine.execute("ALTER TABLE PlanetOrbits ADD INDEX (Name)")
result = engine.execute("ALTER TABLE PlanetOrbits ADD FOREIGN KEY (Name) REFERENCES KnownPlanets(pl_name) ON DELETE NO ACTION ON UPDATE NO ACTION");


#---------------------------------------------
#write completeness table
out2.to_sql('Completeness',engine,chunksize=100,if_exists='replace',dtype={'Name':sqlalchemy.types.String(namemxchar)})
result = engine.execute("ALTER TABLE Completeness ENGINE=InnoDB")
result = engine.execute("ALTER TABLE Completeness ADD INDEX (Name)")
result = engine.execute("ALTER TABLE Completeness ADD FOREIGN KEY (Name) REFERENCES KnownPlanets(pl_name) ON DELETE NO ACTION ON UPDATE NO ACTION");


#---------------------------------------------
#write altplanetorbits table
altorbdata.to_sql('AltPlanetOrbits',engine,chunksize=100,if_exists='replace',dtype={'Name':sqlalchemy.types.String(namemxchar)})
result = engine.execute("ALTER TABLE AltPlanetOrbits ENGINE=InnoDB")
result = engine.execute("ALTER TABLE AltPlanetOrbits ADD INDEX (Name)")
result = engine.execute("ALTER TABLE AltPlanetOrbits ADD FOREIGN KEY (Name) REFERENCES KnownPlanets(pl_name) ON DELETE NO ACTION ON UPDATE NO ACTION");


#---------------------------------------------------
#write alias table
aliasmxchar = np.array([len(n) for n in out3['Alias'].values]).max()


out3.to_sql('Aliases',engine,chunksize=100,if_exists='replace',dtype={'Alias':sqlalchemy.types.String(aliasmxchar)})
result = engine.execute("ALTER TABLE Aliases ENGINE=InnoDB")
result = engine.execute("ALTER TABLE Aliases ADD INDEX (Alias)")
result = engine.execute("ALTER TABLE Aliases ADD INDEX (SID)")


