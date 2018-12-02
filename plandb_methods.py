import requests
import pandas
from StringIO import StringIO
import astropy.units as u
import astropy.constants as const
import EXOSIMS.PlanetPhysicalModel.Forecaster
import numpy as np
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline


def RfromM(m):
    '''
    Given masses m (in Earth masses) return radii (in Earth radii) \
    based on modified forecaster

    '''
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



def getIPACdata():
    '''
    grab everything from exoplanet and composite tables, merge, 
    add additional column info and return/save to disk
    '''

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
    #data = substitute_data(data)

    #sort by planet name 
    data = data.sort_values(by=['pl_name']).reset_index(drop=True)

    # filter rows:
    # we need:
    # distance AND
    # (sma OR (period AND stellar mass)) AND
    # (radius OR mass (either true or m\sin(i)))
    keep = ~np.isnan(data['st_dist'].values) & (~np.isnan(data['pl_orbsmax'].values) | \
            (~np.isnan(data['pl_orbper'].values) & ~np.isnan(data['st_mass'].values))) & \
           (~np.isnan(data['pl_bmassj'].values) | ~np.isnan(data['pl_radj'].values))
    data = data[keep]
    data = data.reset_index(drop=True)

    #remove extraneous columns
    data = data.drop(columns=['pl_rade',
                              'pl_radelim',
                              'pl_radserr2',
                              'pl_radeerr1',
                              'pl_rads',
                              'pl_radslim',
                              'pl_radeerr2',
                              'pl_radserr1',
                              'pl_masse',
                              'pl_masseerr1',
                              'pl_masseerr2',
                              'pl_masselim',
                              'pl_msinie',
                              'pl_msinieerr1',
                              'pl_msinieerr2',
                              'pl_msinielim'
                              ])



    #fill in missing smas from period & star mass
    nosma = np.isnan(data['pl_orbsmax'].values)
    p2sma = lambda mu,T: ((mu*T**2/(4*np.pi**2))**(1/3.)).to('AU')
    GMs = const.G*(data['st_mass'][nosma].values*u.solMass) # units of solar mass
    T = data['pl_orbper'][nosma].values*u.day
    tmpsma = p2sma(GMs,T)
    data.loc[nosma,'pl_orbsmax'] = tmpsma
    data.loc[nosma,'pl_orbsmaxreflink'] = "Calculated from stellar mass and orbital period."

    #propagate filled in sma errors
    GMerrs = ((data['st_masserr1'][nosma] - data['st_masserr2'][nosma])/2.).values*u.solMass*const.G
    Terrs = ((data['pl_orbpererr1'][nosma] - data['pl_orbpererr2'][nosma])/2.).values*u.day

    smaerrs = np.sqrt((2.0*T**2.0*GMs)**(2.0/3.0)/(9*np.pi**(4.0/3.0)*T**2.0)*Terrs**2.0 +\
            (2.0*T**2.0*GMs)**(2.0/3.0)/(36*np.pi**(4.0/3.0)*GMs**2.0)*GMerrs**2.0).to('AU')
    data.loc[nosma,'pl_orbsmaxerr1'] = smaerrs
    data.loc[nosma,'pl_orbsmaxerr2'] = -smaerrs


    #update all WAs (and errors) based on sma
    WA = np.arctan((data['pl_orbsmax'].values*u.AU)/(data['st_dist'].values*u.pc)).to('mas')
    data['pl_angsep'] = WA.value
    sigma_a = ((data['pl_orbsmaxerr1']- data['pl_orbsmaxerr2'])/2.).values*u.AU
    sigma_d = ((data['st_disterr1']- data['st_disterr2'])/2.).values*u.pc
    sigma_wa = (np.sqrt(( (data['pl_orbsmax'].values*u.AU)**2.0*sigma_d**2 + (data['st_dist'].values*u.pc)**2.0*sigma_a**2)/\
            ((data['pl_orbsmax'].values*u.AU)**2.0 + (data['st_dist'].values*u.pc)**2.0)**2.0).decompose()*u.rad).to(u.mas)
    data['pl_angseperr1'] = sigma_wa
    data['pl_angseperr2'] = -sigma_wa


    #fill in radius based on mass
    noR = ((data['pl_radreflink'] == '<a refstr="CALCULATED VALUE" href="/docs/composite_calc.html" target=_blank>Calculated Value</a>') |\
            data['pl_radj'].isnull()).values

    m = ((data['pl_bmassj'][noR].values*u.M_jupiter).to(u.M_earth)).value
    merr = (((data['pl_bmassjerr1'][noR].values - data['pl_bmassjerr2'][noR].values)/2.0)*u.M_jupiter).to(u.M_earth).value
    R = RfromM(m)

    #create mod forecaster radius column
    data = data.assign(pl_radj_forecastermod=data['pl_radj'].values)
    #data['pl_radj_forecastermod'][noR] = ((R*u.R_earth).to(u.R_jupiter)).value
    data.loc[noR,'pl_radj_forecastermod'] = ((R*u.R_earth).to(u.R_jupiter)).value

    # now the Fortney model
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
    #data['pl_radj_fortney'][noR] = ((Rf*u.R_earth).to(u.R_jupiter)).value
    data.loc[noR,'pl_radj_fortney'] = ((Rf*u.R_earth).to(u.R_jupiter)).value

    #populate max WA based on available eccentricity data (otherwise maxWA = WA)
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

    return data


