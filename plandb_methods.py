from __future__ import print_function
from __future__ import division
import requests
import pandas
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO
import astropy.units as u
import astropy.constants as const
import EXOSIMS.PlanetPhysicalModel.Forecaster
import numpy as np
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, griddata
from astropy.time import Time
import sqlalchemy.types 
from sqlalchemy import create_engine
import re
import os
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.eccanom import eccanom
from astroquery.simbad import Simbad
from requests.exceptions import ConnectionError
import math
from MeanStars import MeanStars


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

    print("Querying IPAC for all data.")
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
    data = data.rename(columns={# 'pl_smax':'pl_orbsmax',
                                # 'pl_smaxerr1':'pl_orbsmaxerr1',
                                # 'pl_smaxerr2':'pl_orbsmaxerr2',
                                # 'pl_smaxlim':'pl_orbsmaxlim',
                                'pl_smaxreflink':'pl_orbsmaxreflink',
                                # 'pl_eccen':'pl_orbeccen',
                                # 'pl_eccenerr1':'pl_orbeccenerr1',
                                # 'pl_eccenerr2':'pl_orbeccenerr2',
                                # 'pl_eccenlim':'pl_orbeccenlim',
                                'pl_eccenreflink':'pl_orbeccenreflink',
                                # 'st_met':'st_metfe',
                                # 'st_meterr1':'st_metfeerr1',
                                # 'st_meterr2':'st_metfeerr2',
                                'st_metreflink':'st_metfereflink',
                                # 'st_metlim':'st_metfelim',
                                })

    composite_cols = [col for col in data.columns if ('reflink' in col) or (col == 'pl_name')]
    composite_cols.extend(['pl_radj'])

    data = data[composite_cols]

    #sort by planet name
    data = data.sort_values(by=['pl_name']).reset_index(drop=True)
    data2 = data2.sort_values(by=['pl_name']).reset_index(drop=True)

    #merge data sets
    data = data.combine_first(data2)

    #############################################################
    pl_names_df = pandas.read_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\output.csv")
    pl_names = pl_names_df["pl_name"]

    data = data.loc[data['pl_name'].isin(pl_names)]
    #############################################################

    # substitute data from the extended table.
    print("Querying extended data table.")
    #grab extended data table
    query_ext = """https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exomultpars&select=*&format=csv"""
    r_ext = requests.get(query_ext)
    data_ext = pandas.read_csv(StringIO(r_ext.content))

    #############################################################
    data_ext = data_ext.loc[data_ext['mpl_name'].isin(pl_names)]
    #############################################################

    extended = (data_ext.sort_values('mpl_name')).reset_index(drop=True)
    
    authregex = re.compile(r"(<a.*f>)|(</a>)")

    #create columns for short references and publication years, and separate columns of
    #ref links for all attributes to be updated
    shortrefs = [authregex.sub("", extended['mpl_reflink'][j]).strip() for j in range(len(extended))]
    refs = extended["mpl_reflink"].values
    refyrs = [int(re.findall('(\d{4})', ref)[0]) for ref in refs]
    extended = extended.assign(ref_author=shortrefs,\
                               publication_year=refyrs,\
                               best_data=np.zeros(len(extended)), \
                               mpl_orbperreflink=refs,\
                               mpl_orbsmaxreflink=refs,\
                               mpl_orbeccenreflink=refs,\
                               mpl_bmassreflink=refs,\
                               mpl_radreflink=refs,\
                               mst_massreflink=refs)

    #pick best attribute row for each planet
    print("Choosing best attributes for all planets.")
    for j,name in enumerate(data['pl_name'].values):
        print("%s: %d/%d"%(name,j+1,len(data)))

        planet_rows = extended.loc[extended["mpl_name"] == name]

        sorted_rows = planet_rows.sort_values(by=["publication_year"], axis=0, ascending=False)
        good_idx = sorted_rows.index[0]
        good_lvl = 0
        for index, row in sorted_rows.iterrows():
            base_need = (not pandas.isnull(row["mpl_orbsmax"]) or not pandas.isnull(row["mpl_orbper"])) and \
                        (not pandas.isnull(row["mpl_bmassj"]) or not pandas.isnull(row["mpl_radj"]))

            # Has everything
            if good_lvl < 4 and (base_need
                                 and not pandas.isnull(row["mpl_orbeccen"]) and not pandas.isnull(row["mpl_orbtper"])
                                 and not pandas.isnull(row["mpl_orblper"]) and not pandas.isnull(row["mpl_orbincl"])):
                good_idx = index
                good_lvl = 4
                break

            # Has everything except inclination
            if good_lvl < 3 and (base_need
                                 and not pandas.isnull(row["mpl_orbeccen"]) and not pandas.isnull(row["mpl_orbtper"])
                                 and not pandas.isnull(row["mpl_orblper"])):
                good_idx = index
                good_lvl = 3

            # Has either periapsis time or argument of pariapsis
            elif good_lvl < 2 and (base_need
                                   and not pandas.isnull(row["mpl_orbeccen"]) and (not pandas.isnull(row["mpl_orbtper"])
                                                                          or not pandas.isnull(row["mpl_orblper"]))):
                good_idx = index
                good_lvl = 2
            # Has eccentricity
            elif good_lvl < 1 and (base_need
                                   and not pandas.isnull(row["mpl_orbeccen"])):
                good_idx = index
                good_lvl = 1
            # 1st doesn't have basic info
            elif index == good_idx and not base_need:
                good_idx = -1
            # Previous row needed to be replaced
            elif good_idx == -1 and base_need:
                good_idx = index
                good_lvl = 1

        if good_idx == -1:
            good_idx = sorted_rows.index[0]

        extended.at[good_idx, "best_data"] = 1

    #strip leading 'm' from all col names
    colmap = {k: k[1:] if (k.startswith('mst_') | k.startswith('mpl_')) else k for k in extended.keys()}
    extended = extended.rename(columns=colmap)

    columns_to_replace = ["pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_orbperlim", "pl_orbsmax",
                          "pl_orbsmaxerr1", "pl_orbsmaxerr2", "pl_orbsmaxlim", "pl_orbeccen",
                          "pl_orbeccenerr1", "pl_orbeccenerr2", "pl_orbeccenlim", "pl_orbtper",
                          "pl_orbtpererr1", "pl_orbtpererr2", "pl_orbtperlim", "pl_orblper",
                          "pl_orblpererr1", "pl_orblpererr2", "pl_orblperlim", "pl_bmassj",
                          "pl_bmassjerr1", "pl_bmassjerr2", "pl_bmassjlim", "pl_radj", "pl_radjerr1",
                          "pl_radjerr2", "pl_radjlim", "pl_orbincl", "pl_orbinclerr1", "pl_orbinclerr2",
                          "pl_orbincllim", "pl_orbperreflink", "pl_orbsmaxreflink", "pl_orbeccenreflink",
                          "pl_bmassreflink", "pl_radreflink", "pl_bmassprov"]
    columns_to_replace_st = columns_to_replace + ["st_mass","st_masserr1","st_masserr2","st_masslim","st_massreflink"]


    #update rows as needed
    print("Updating planets with best attributes.")
    data = data.assign(pl_def_override=np.zeros(len(data)))
    for j,name in enumerate(data['pl_name'].values):
        row_extended = extended.loc[(extended["best_data"] == 1) & (extended["pl_name"] == name)]
        idx = data.loc[(data["pl_name"] == name)].index
        if int(row_extended["pl_def"]) == 0:
            final_columns = columns_to_replace
            
            row_extended.index = idx
            row_extended_replace = row_extended[final_columns]

            data.loc[(data["pl_name"] == name,final_columns)] = np.nan
            data.update(row_extended_replace,overwrite=True)
            data.loc[idx,'pl_def_override'] = 1
            print("%s: %d/%d"%(name,j+1,len(data)))


    #sort by planet name 
    data = data.sort_values(by=['pl_name']).reset_index(drop=True)

    print("Filtering to useable planets and calculating additional properties.")

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
           (data['pl_radreflink'] == '<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>') |\
            data['pl_radj'].isnull()).values

    m = ((data['pl_bmassj'][noR].values*u.M_jupiter).to(u.M_earth)).value
    merr = (((data['pl_bmassjerr1'][noR].values - data['pl_bmassjerr2'][noR].values)/2.0)*u.M_jupiter).to(u.M_earth).value
    R = RfromM(m)   
    Rerr = [RfromM(np.random.normal(loc=m[j], scale=merr[j], size=int(1e4))).std() if not(np.isnan(merr[j])) else np.nan for j in range(len(m))]

    #create mod forecaster radius column and error cols
    data = data.assign(pl_radj_forecastermod=data['pl_radj'].values)
    data.loc[noR,'pl_radj_forecastermod'] = ((R*u.R_earth).to(u.R_jupiter)).value
    data = data.assign(pl_radj_forecastermoderr1=data['pl_radjerr1'].values)
    data.loc[noR,'pl_radj_forecastermoderr1'] = ((Rerr*u.R_earth).to(u.R_jupiter)).value
    data = data.assign(pl_radj_forecastermoderr2=data['pl_radjerr2'].values)
    data.loc[noR,'pl_radj_forecastermoderr2'] = -((Rerr*u.R_earth).to(u.R_jupiter)).value


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

    Rf[mg10] = griddata(fortney.giant_pts2, fortney.giant_vals2,( np.array([10.]*np.where(mg10)[0].size), tmpsmas, tmpmass))

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


def packagePhotometryData(dbfile=None):
    """
    Read photometry data from database of grid models and repackage into single ndarray of data
    """
    if dbfile is None:
        dbfile = os.path.join(os.getenv('HOME'),'Documents','AFTA-Coronagraph','ColorFun','AlbedoModels_2015.db')

    # grab photometry data 
    enginel = create_engine('sqlite:///' + dbfile)

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

    outname = os.path.split(dbfile)[-1].split(os.extsep)[0]
    np.savez(outname,metallicities=metallicities,dists=dists,clouds=clouds,cloudstr=cloudstr,betas=betas,wavelns=wavelns,allphotdata=allphotdata)

def loadPhotometryData(infile='allphotdata_2015.npz'):
    """
    Read stored photometry data from disk and generate interpolants over data
    """

    tmp = np.load(infile)
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

    return {'allphotdata':allphotdata,
            'clouds':clouds,
            'cloudstr':cloudstr,
            'wavelns':wavelns,
            'betas':betas,
            'dists':dists,
            'metallicities':metallicities,
            'distinterp':distinterp,
            'betainterp':betainterp,
            'feinterp':feinterp,
            'cloudinterp':cloudinterp,
            'photinterps':photinterps2,
            'quadinterps':quadinterps}


def genBands():
    """
    Generate central wavelengths and bandpasses describing bands of interest
    """

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

    return zip(lambdas,bands,bws,bandws,bandwsteps)


def calcQuadratureVals(data, bandzip, photdict):
    """
    Calculate quadrature photometry values for planets in data (output of getIPACdata)
    for bands in bandzip (output of genBands) assuming photometry info from photdict
    (output from loadPhotometryData)

    """

    smas = data['pl_orbsmax'].values
    fes = data['st_metfe'].values
    fes[np.isnan(fes)] = 0.0
    Rps = data['pl_radj_forecastermod'].values
    inc = data['pl_orbincl'].values
    eccen = data['pl_orbeccen'].values
    arg_per = data['pl_orblper'].values
    lum = data['st_lum'].values
    temps = data['st_teff'].values

    tmpout = {}

    quadinterps = photdict['quadinterps']
    feinterp = photdict['feinterp']
    distinterp = photdict['distinterp']
    lambdas = []

    ms = MeanStars()

    #iterate over all data rows
    for j, (Rp, fe,a, I, e, w, lm, teff) in enumerate(zip(Rps, fes,smas, inc, eccen, arg_per, lum, temps)):
        print("%d/%d"%(j+1,len(Rps)))
        for c in photdict['clouds']:
            for l,band,bw,ws,wstep in bandzip: 
                if j == 0:
                    lambdas.append(l)
                    #allocate output arrays
                    tmpout['quad_pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"] = np.zeros(smas.shape)
                    tmpout['quad_dMag_'+"%03dC_"%(c*100)+str(l)+"NM"] = np.zeros(smas.shape)
                    tmpout['quad_radius_' + "%03dC_" % (c * 100) + str(l) + "NM"] = np.zeros(smas.shape)

                if np.isnan(lm):
                    if np.isnan(teff):
                        lum_fix = 1
                    else:
                        lum_fix = (10 ** ms.TeffOther('logL', teff)) ** .5
                else:
                    lum_fix = (10 ** lm) ** .5  # Since lum is log base 10 of solar luminosity

                #Only calculate quadrature distance if known eccentricity and argument of periaps, 
                #and not face-on orbit
                if not np.isnan(e) and not np.isnan(w) and I != 0:
                    nu1 = -w
                    nu2 = np.pi - w

                    if np.isnan(lm):
                        if np.isnan(teff):
                            lum_fix = 1
                        else:
                            lum_fix = (10 ** ms.TeffOther('logL', teff)) ** .5
                    else:
                        lum_fix = (10 ** lm) ** .5  # Since lum is log base 10 of solar luminosity

                    r1 = a * (1.0 - e ** 2.0) / (1.0 + e * np.cos(nu1)) * lum_fix
                    r2 = a * (1.0 - e ** 2.0) / (1.0 + e * np.cos(nu2)) * lum_fix

                    pphi1 = quadinterps[float(feinterp(fe))][float(distinterp(r1))][c](ws).sum() * wstep / bw
                    pphi2 = quadinterps[float(feinterp(fe))][float(distinterp(r2))][c](ws).sum() * wstep / bw
                    if np.isinf(pphi1):
                        print("Inf value encountered in pphi")
                        pphi1 = np.nan
                    if np.isinf(pphi2):
                        print("Inf value encountered in pphi")
                        pphi2 = np.nan

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
                    pphi = quadinterps[float(feinterp(fe))][float(distinterp(a*lum_fix))][c](ws).sum()*wstep/bw
                    if np.isinf(pphi):
                        print("Inf value encountered in pphi")
                        pphi = np.nan

                    tmpout['quad_pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"][j] = pphi
                    dMag = deltaMag(1, Rp*u.R_jupiter, a*u.AU, pphi)
                    tmpout['quad_radius_' + "%03dC_" % (c * 100) + str(l) + "NM"][j] = a * lum_fix

                if np.isinf(dMag): 
                    print("Inf value encountered in dmag")
                    dMag = np.nan
                tmpout['quad_dMag_'+"%03dC_"%(c*100)+str(l)+"NM"][j] = dMag

    for l in lambdas:
        tmp = []
        for c in photdict['clouds']:
            tmp.append(tmpout['quad_dMag_'+"%03dC_"%(c*100)+str(l)+"NM"])
        tmp = np.vstack(tmp)
        tmpout["quad_dMag_min_"+str(l)+"NM"] = np.nanmin(tmp,axis=0)
        tmpout["quad_dMag_max_"+str(l)+"NM"] = np.nanmax(tmp,axis=0)
        tmpout["quad_dMag_med_"+str(l)+"NM"] = np.nanmedian(tmp,axis=0)

    data = data.join(pandas.DataFrame(tmpout))

    return data


def genOrbitData(data, bandzip, photdict, t0=None):
    """ Generate data for one orbital period starting at absolute time t0 (if time
    of periapsis is known) for all planets in data (output of getIPACdata)
    based on photometric data in photdict (output of loadPhotometryData) for bands
    in bandzip (output of genBands)
    """

    if t0 is None:
        t0 = Time('2026-01-01T00:00:00', format='isot', scale='utc')

    plannames = data['pl_name'].values
    minWA = data['pl_minangsep'].values*u.mas
    maxWA = data['pl_maxangsep'].values*u.mas

    photinterps2 = photdict['photinterps']
    feinterp = photdict['feinterp']
    distinterp = photdict['distinterp']
    lambdas = []


    orbdata = None
    for j in range(len(plannames)):
        row = data.iloc[j]

        #orbital parameters
        a = row['pl_orbsmax']
        e = row['pl_orbeccen']
        if np.isnan(e): e = 0.0
        I = row['pl_orbincl']*np.pi/180.0
        if np.isnan(I): I = np.pi/2.0
        w = row['pl_orblper']*np.pi/180.0
        if np.isnan(w): w = 0.0
        Rp = row['pl_radj_forecastermod']
        dist = row['st_dist']
        fe = row['st_metfe']
        if np.isnan(fe): fe = 0.0

        #time
        M = np.linspace(0,2*np.pi,100)
        t = np.zeros(100)*np.nan
        tau = row['pl_orbtper'] #jd
        if not np.isnan(tau):
            Tp = row['pl_orbper'] #days
            if Tp == 0:
                Tp = np.nan
            if np.isnan(Tp):
                Mstar = row['st_mass'] #solar masses
                if not np.isnan(Mstar):
                    mu = const.G*(Mstar*u.solMass).decompose()
                    Tp = (2*np.pi*np.sqrt(((a*u.AU)**3.0)/mu)).decompose().to(u.d).value
            if not np.isnan(Tp):
                n = 2*np.pi/Tp
                t = np.linspace(t0.jd,t0.jd+Tp,100)
                M = np.mod((t - tau)*n,2*np.pi)

        #calculate orbital values
        E = eccanom(M, e)
        nu = 2*np.arctan(np.sqrt((1.0 + e)/(1.0 - e))*np.tan(E/2.0));
        d = a*(1.0 - e**2.0)/(1 + e*np.cos(nu))
        s = d*np.sqrt(4.0*np.cos(2*I) + 4*np.cos(2*nu + 2.0*w) - 2.0*np.cos(-2*I + 2.0*nu + 2*w) - 2*np.cos(2*I + 2*nu + 2*w) + 12.0)/4.0
        beta = np.arccos(np.sin(I)*np.sin(nu+w))*u.rad

        WA = np.arctan((s*u.AU)/(dist*u.pc)).to('mas').value
        print(j,plannames[j],WA.min() - minWA[j].value, WA.max() - maxWA[j].value)

        outdict = {'Name': [plannames[j]]*len(M),
                    'M': M,
                    't': t-t0.jd,
                    'r': d,
                    's': s,
                    'WA': WA,
                    'beta': beta.to(u.deg).value}

        lum = row['st_lum']
        teff = row['st_teff']

        ms = MeanStars()

        if np.isnan(lum):
            if np.isnan(teff):
                lum_fix = 1
            else:
                lum_fix = (10 ** ms.TeffOther('logL', teff)) ** .5
        else:
            lum_fix = (10 ** lum) ** .5  # Since lum is log base 10 of solar luminosity

        inds = np.argsort(beta)
        for c in photdict['clouds']:
            for l,band,bw,ws,wstep in bandzip:
                pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a * lum_fix))][c](beta.to(u.deg).value[inds],ws).sum(1)*wstep/bw)[np.argsort(inds)]
                pphi[np.isinf(pphi)] = np.nan
                outdict['pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"] = pphi
                dMag = deltaMag(1, Rp*u.R_jupiter, d*u.AU, pphi)
                dMag[np.isinf(dMag)] = np.nan
                outdict['dMag_'+"%03dC_"%(c*100)+str(l)+"NM"] = dMag


        out = pandas.DataFrame(outdict)

        if orbdata is None:
            orbdata = out.copy()
        else:
            orbdata = orbdata.append(out)


    orbdata = orbdata.sort_values(by=['Name','t','M']).reset_index(drop=True)
    tmpout = {}
    for l in lambdas:
        tmpout["dMag_min_"+str(l)+"NM"] = np.zeros(len(orbdata))
        tmpout["dMag_max_"+str(l)+"NM"] = np.zeros(len(orbdata))
        tmpout["dMag_med_"+str(l)+"NM"] = np.zeros(len(orbdata))
        tmpout["pPhi_min_"+str(l)+"NM"] = np.zeros(len(orbdata))
        tmpout["pPhi_max_"+str(l)+"NM"] = np.zeros(len(orbdata))
        tmpout["pPhi_med_"+str(l)+"NM"] = np.zeros(len(orbdata))



    for name in np.unique(orbdata['Name']):
        print(name)
        loc = orbdata['Name'] == name
        for l in lambdas:
            tmp = []
            tmp2 = []
            for c in photdict['clouds']:
                tmp.append(orbdata.loc[loc,'dMag_'+"%03dC_"%(c*100)+str(l)+"NM"].values)
                tmp2.append(orbdata.loc[loc,'pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"].values)
            tmp = np.vstack(tmp)
            tmp2 = np.vstack(tmp2)
            tmpout["dMag_min_"+str(l)+"NM"][loc.values] = np.nanmin(tmp,axis=0)
            tmpout["dMag_max_"+str(l)+"NM"][loc.values] = np.nanmax(tmp,axis=0)
            tmpout["dMag_med_"+str(l)+"NM"][loc.values] = np.nanmedian(tmp,axis=0)
            tmpout["pPhi_min_"+str(l)+"NM"][loc.values] = np.nanmin(tmp2,axis=0)
            tmpout["pPhi_max_"+str(l)+"NM"][loc.values] = np.nanmax(tmp2,axis=0)
            tmpout["pPhi_med_"+str(l)+"NM"][loc.values] = np.nanmedian(tmp2,axis=0)

    orbdata = orbdata.join(pandas.DataFrame(tmpout))

    return orbdata


def genAltOrbitData(data, bandzip, photdict, t0=None):
    """ Generate data for one orbital period starting at absolute time t0 (if time
    of periapsis is known) for all planets in data (output of getIPACdata)
    based on photometric data in photdict (output of loadPhotometryData) for bands
    in bandzip (output of genBands) for varying values of inclination.
    """

    if t0 is None:
        t0 = Time('2026-01-01T00:00:00', format='isot', scale='utc')

    plannames = data['pl_name'].values
    photinterps2 = photdict['photinterps']
    feinterp = photdict['feinterp']
    distinterp = photdict['distinterp']

    Isglob = np.array([90,60,30])
    (l,band,bw,ws,wstep) = bandzip[0]
    c = 3.0

    altorbdata = None
    for j in range(len(plannames)):
        row = data.iloc[j]
        print("%d/%d  %s"%(j+1,len(plannames),plannames[j]))


        #if there is an inclination error, then we're going to skip the row altogether
        if not(np.isnan(row['pl_orbinclerr1'])) and not(np.isnan(row['pl_orbinclerr1'])):
            continue


        if row['pl_bmassprov'] == 'Msini':
                Icrit = np.arcsin( ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value/((0.0800*u.M_sun).to(u.M_earth)).value )
        else:
            Icrit = 10*np.pi/180.0

        Is = np.hstack((Isglob*np.pi/180.0,Icrit))

        #orbital parameters
        a = row['pl_orbsmax']
        e = row['pl_orbeccen']
        if np.isnan(e): e = 0.0
        w = row['pl_orblper']*np.pi/180.0
        if np.isnan(w): w = 0.0
        Rp = row['pl_radj_forecastermod']
        dist = row['st_dist']
        fe = row['st_metfe']
        if np.isnan(fe): fe = 0.0

        #time
        Tp = row['pl_orbper'] #days
        Mstar = row['st_mass'] #solar masses
        tau = row['pl_orbtper']

        if np.isnan(Tp) or (Tp == 0.0):
            if np.isnan(Mstar):
                continue
            mu = const.G*(Mstar*u.solMass).decompose()
            Tp = (2*np.pi*np.sqrt(((a*u.AU)**3.0)/mu)).decompose().to(u.d).value

        n = 2*np.pi/Tp
        if Tp > 10*365.25:
            tend = 10*365.25
        else:
            tend = Tp
        t = np.arange(t0.jd,t0.jd+tend,30)
        if np.isnan(tau):
            tau = 0.0
        M = np.mod((t - tau)*n,2*np.pi)

        E = eccanom(M, e)
        nu = 2*np.arctan(np.sqrt((1.0 + e)/(1.0 - e))*np.tan(E/2.0));
        d = a*(1.0 - e**2.0)/(1 + e*np.cos(nu))

        outdict = {'Name': [plannames[j]]*len(M),
                    'M': M,
                    't': t-t0.jd,
                    'r': d,
                    'Icrit': [Icrit]*len(M)
                   }

        lum = row['st_lum']
        teff = row['st_teff']

        ms = MeanStars()

        if np.isnan(lum):
            if np.isnan(teff):
                lum_fix = 1
            else:
                lum_fix = (10 ** ms.TeffOther('logL', teff)) ** .5
        else:
            lum_fix = (10 ** lum) ** .5  # Since lum is log base 10 of solar luminosity

        for k,I in enumerate(Is):
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
            pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a * lum_fix))][c](beta.to(u.deg).value[inds],ws).sum(1)*wstep/bw)[np.argsort(inds)]
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

    return altorbdata


def genOrbitData_2(data, bandzip, photdict, t0=None):
    if t0 is None:
        t0 = Time('2026-01-01T00:00:00', format='isot', scale='utc')

    plannames = data['pl_name'].values
    minWA = data['pl_minangsep'].values * u.mas
    maxWA = data['pl_maxangsep'].values * u.mas

    photinterps2 = photdict['photinterps']
    feinterp = photdict['feinterp']
    distinterp = photdict['distinterp']
    lambdas = []


    (l, band, bw, ws, wstep) = bandzip[0]
    c = 3.0

    orbitfits_dict = []
    orbitfit_ids = []

    max_pl_id = -1
    max_evalorbit_id = -1

    orbdata = None
    orbitfits = None
    evalorbits = None
    for j in range(len(plannames)):
        row = data.iloc[j]

        # orbital parameters
        a = row['pl_orbsmax']
        e = row['pl_orbeccen']
        if np.isnan(e): e = 0.0
        I = row['pl_orbincl'] * np.pi / 180.0
        if np.isnan(I): I = np.pi / 2.0
        w = row['pl_orblper'] * np.pi / 180.0
        if np.isnan(w): w = 0.0
        Rp = row['pl_radj_forecastermod']
        dist = row['st_dist']
        fe = row['st_metfe']
        if np.isnan(fe): fe = 0.0

        # time
        M = np.linspace(0, 2 * np.pi, 100)
        t = np.zeros(100) * np.nan
        tau = row['pl_orbtper']  # jd
        Tp = row['pl_orbper']  # days

        if Tp == 0:
            Tp = np.nan
        if np.isnan(Tp):
            Mstar = row['st_mass']  # solar masses
            if not np.isnan(Mstar):
                mu = const.G * (Mstar * u.solMass).decompose()
                Tp = (2 * np.pi * np.sqrt(((a * u.AU) ** 3.0) / mu)).decompose().to(u.d).value
        else:
            if not np.isnan(tau):
                n = 2 * np.pi / Tp
                t = np.linspace(t0.jd, t0.jd + Tp, 100)
                M = np.mod((t - tau) * n, 2 * np.pi)



        # calculate orbital values
        E = eccanom(M, e)
        nu = 2 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E / 2.0));
        d = a * (1.0 - e ** 2.0) / (1 + e * np.cos(nu))
        s = d * np.sqrt(
            4.0 * np.cos(2 * I) + 4 * np.cos(2 * nu + 2.0 * w) - 2.0 * np.cos(-2 * I + 2.0 * nu + 2 * w) - 2 * np.cos(
                2 * I + 2 * nu + 2 * w) + 12.0) / 4.0
        beta = np.arccos(np.sin(I) * np.sin(nu + w)) * u.rad

        WA = np.arctan((s * u.AU) / (dist * u.pc)).to('mas').value
        print(j, plannames[j], WA.min() - minWA[j].value, WA.max() - maxWA[j].value)

        # Calculate periaps after current date and after 1/1/2026
        if not np.isnan(tau):
            if not np.isnan(Tp):
                period_nums_cur = (Time.now().jd - tau) * 1.0 / Tp
                period_nums_2026 = (t0.jd - tau) * 1.0 / Tp

                tper_next = math.ceil(period_nums_cur) * Tp + tau
                tper_2026 = math.ceil(period_nums_2026) * Tp + tau
            else:
                tper_next = tau
                tper_2026 = tau
        else:
            tper_next = 0.0
            tper_2026 = 0.0

        if np.isnan(tau):
            tau = 0.0

        st_cols = [col for col in data.columns if ('st_' in col) or ('gaia_' in col)]
        st_cols.extend(["dec", "dec_str", "hd_name", "hip_name", "ra", "ra_str"])
        pl_cols = np.asarray(np.setdiff1d(data.columns, st_cols))
        np.append(pl_cols, 'st_name')
        included = ['pl_name', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl', 'pl_orblper', 'pl_orblper', 'pl_orbper'
                                                                                           'pl_orbtper']
        pl_cols2 = [entry for entry in pl_cols if entry not in included]

        if np.isnan(row['pl_orbincl']):
            Is_1 = np.array([np.nan, I])
            defaults = np.array([1, 0])
        else:
            Is_1 = np.array([I])
            defaults = np.array([1])
            
        orbitfits_dict = {'pl_id': [j] * len(Is_1),
                          'pl_name': [plannames[j]] * len(Is_1),
                          'orbsmax': [row['pl_orbsmax']] * len(Is_1),
                          'orbeccen': [row['pl_orbeccen']] * len(Is_1),
                          'orbincl': Is_1 * 180 / np.pi,
                          'orblper': [row['pl_orblper']] * len(Is_1),
                          'orbper': [row['pl_orbper']] * len(Is_1),
                          'orbtper': [row['pl_orbtper']] * len(Is_1),
                          'orbtper_next': [tper_next] * len(Is_1),
                          'orbtper_2026': [tper_2026] * len(Is_1),
                          'is_Icrit': [0] * len(Is_1),
                          'def_pl': defaults}
        for col_name in pl_cols2:
            if col_name in ['completeness', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag'] and len(Is_1) > 1:
                orbitfits_dict[col_name.replace('pl_', '')] = [row[col_name], np.nan]
            else:
                orbitfits_dict[col_name.replace('pl_', '')] = row[col_name]

        min_pl_id = max_pl_id + 1
        max_pl_id = min_pl_id + len(Is_1) - 1

        evalorbit_dict = {'pl_id': [j],
                          'pl_fit_id': [max_pl_id],
                          'pl_name': [plannames[j]],
                          'orbsmax': [a],
                          'orbeccen': [e],
                          'orbincl': [I * 180 / np.pi],
                          'orblper': [w],
                          'orbper': [Tp],
                          'orbtper': [tau],
                          'is_Icrit': [0],
                          'def_eval': [1]}

        min_evalorbit_id = max_evalorbit_id + 1
        max_evalorbit_id = min_evalorbit_id

        outdict = {'pl_id': [j] * len(M),
                   'pl_fit_id': [max_pl_id] * len(M),
                   'pl_name': [plannames[j]] * len(M),
                   'evalorbit_id': [max_evalorbit_id] * len(M),
                   'M': M,
                   't': t - t0.jd,
                   'r': d,
                   's': s,
                   'WA': WA,
                   'beta': beta.to(u.deg).value,
                   'def_orb': [1] * len(M)}

        inds = np.argsort(beta)

        lum = row['st_lum']
        teff = row['st_teff']

        ms = MeanStars()

        if np.isnan(lum):
            if np.isnan(teff):
                lum_fix = 1
            else:
                lum_fix = (10 ** ms.TeffOther('logL', teff)) ** .5
        else:
            lum_fix = (10 ** lum) ** .5  # Since lum is log base 10 of solar luminosity

        for c in photdict['clouds']:
            for l, band, bw, ws, wstep in bandzip:
                pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a*lum_fix))][c](beta.to(u.deg).value[inds], ws).sum(
                    1) * wstep / bw)[np.argsort(inds)]
                pphi[np.isinf(pphi)] = np.nan
                outdict['pPhi_' + "%03dC_" % (c * 100) + str(l) + "NM"] = pphi
                dMag = deltaMag(1, Rp * u.R_jupiter, d * u.AU, pphi)
                dMag[np.isinf(dMag)] = np.nan
                outdict['dMag_' + "%03dC_" % (c * 100) + str(l) + "NM"] = dMag

        out = pandas.DataFrame(outdict)

        if orbdata is None:
            orbdata = out.copy()
        else:
            orbdata = orbdata.append(out)

        if orbitfits is None:
            orbitfits = pandas.DataFrame(orbitfits_dict).copy()
        else:
            orbitfits = orbitfits.append(pandas.DataFrame(orbitfits_dict))

        if evalorbits is None:
            evalorbits = pandas.DataFrame(evalorbit_dict).copy()
        else:
            evalorbits = evalorbits.append(pandas.DataFrame(evalorbit_dict))


        if row['pl_bmassprov'] == 'Msini':
            Icrit = np.arcsin(
                ((row['pl_bmassj'] * u.M_jupiter).to(u.M_earth)).value / ((0.0800 * u.M_sun).to(u.M_earth)).value)
        else:
            Icrit = 10 * np.pi / 180.0

        # Alt orb data
        Isglob = np.array([60, 30])
        c = 3.0
        (l, band, bw, ws, wstep) = bandzip[0]
        if I != np.pi / 2.0: # To account if 90 was not chosen as the primary orbit
            Isglob = np.insert(Isglob, 0, 90)
            is_icrit = [0, 0, 0, 1]
        else: # If 90 was chosen for the primary orbit
            is_icrit = [0, 0, 1]

        # Condition to generate altorbits with inclinations of 30, 60, and 90
        if not (np.isnan(row['pl_orbinclerr1'])) and not (np.isnan(row['pl_orbinclerr2'])):
            Isglob = np.array([row['pl_orbinclerr1'] + row['pl_orbincl'], row['pl_orbinclerr2'] + row['pl_orbincl']])
            skip_orbitfits = True
            is_icrit = [0, 0]
        else:
            skip_orbitfits = False

        if not skip_orbitfits:
            Is = np.hstack((Isglob * np.pi / 180.0, Icrit))
        else:
            Is = np.hstack(Isglob * np.pi / 180.0)

        # time
        Tp = row['pl_orbper']  # days
        Mstar = row['st_mass']  # solar masses
        tau = row['pl_orbtper']

        if np.isnan(Tp) or (Tp == 0.0):
            if np.isnan(Mstar):
                continue
            mu = const.G * (Mstar * u.solMass).decompose()
            Tp = (2 * np.pi * np.sqrt(((a * u.AU) ** 3.0) / mu)).decompose().to(u.d).value

        n = 2 * np.pi / Tp
        if Tp > 10 * 365.25:
            tend = 10 * 365.25
        else:
            tend = Tp
        t = np.arange(t0.jd, t0.jd + tend, 30)
        if np.isnan(tau):
            tau = 0.0
        M = np.mod((t - tau) * n, 2 * np.pi)

        E = eccanom(M, e)
        nu = 2 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E / 2.0));
        d = a * (1.0 - e ** 2.0) / (1 + e * np.cos(nu))

        if not skip_orbitfits:
            orbitfits_dict = {'pl_id': [j] * len(Is),
                              'pl_name': [plannames[j]] * len(Is),
                              'orbsmax': [row['pl_orbsmax']] * len(Is),
                              'orbeccen': [row['pl_orbeccen']] * len(Is),
                              'orbincl': Is * 180 / np.pi,
                              'orblper': [row['pl_orblper']] * len(Is),
                              'orbper': [row['pl_orbper']] * len(Is),
                              'orbtper': [row['pl_orbtper']] * len(Is),
                              'orbtper_next': [tper_next] * len(Is),
                              'orbtper_2026': [tper_2026] * len(Is),
                              'is_Icrit': is_icrit,
                              'def_pl': [0] * len(Is)}

            for col_name in pl_cols2:
                if col_name not in ['completeness', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag']:
                    orbitfits_dict[col_name.replace('pl_', '')] = row[col_name]

            min_pl_id = max_pl_id + 1
            max_pl_id = min_pl_id + len(Is) - 1
            pl_ids = np.linspace(min_pl_id, max_pl_id, num=len(Is))
            pl_ids = pl_ids.tolist()
        else:
            pl_ids = [max_pl_id] * len(Is)

        evalorbit_dict = {'pl_id': [j] * len(Is),
                          'pl_fit_id': pl_ids,
                          'pl_name': [plannames[j]] * len(Is),
                          'orbsmax': [a] * len(Is),
                          'orbeccen': [e] * len(Is),
                          'orbincl': Is * 180 / np.pi,
                          'orblper': [w] * len(Is),
                          'orbper': [Tp] * len(Is),
                          'orbtper': [tau] * len(Is),
                          'is_Icrit': is_icrit,
                          'def_eval': [0] * len(Is)}

        min_evalorbit_id = max_evalorbit_id + 1
        max_evalorbit_id = min_evalorbit_id + len(Is) - 1
        evalorbit_ids = np.linspace(min_evalorbit_id, max_evalorbit_id, num=len(Is))

        # outdict = {'Name': [plannames[j]] * len(M),
        #            'M': M,
        #            't': t - t0.jd,
        #            'r': d,
        #            'Icrit': [Icrit] * len(M)
        #            }

        slist = []
        WAlist = []
        betaList = []
        pPhiList = []
        dMagList = []

        for k, I in enumerate(Is):
            s = d * np.sqrt(4.0 * np.cos(2 * I) + 4 * np.cos(2 * nu + 2.0 * w) - 2.0 * np.cos(-2 * I + 2.0 * nu + 2 * w) - 2 * np.cos(2 * I + 2 * nu + 2 * w) + 12.0) / 4.0
            beta = np.arccos(np.sin(I) * np.sin(nu + w)) * u.rad

            WA = np.arctan((s * u.AU) / (dist * u.pc)).to('mas').value

            # if I == Icrit:
            #     Itag = "crit"
            # else:
            #     Itag = "%02d" % (Isglob[k])

            # outdict["s_I" + Itag] = s
            # outdict["WA_I" + Itag] = WA
            # outdict["beta_I" + Itag] = beta.to(u.deg).value
            slist.append(s)
            WAlist.append(WA)
            betaList.append(beta)

            inds = np.argsort(beta)
            pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a*lum_fix))][c](beta.to(u.deg).value[inds], ws).sum(
                1) * wstep / bw)[np.argsort(inds)]
            pphi[np.isinf(pphi)] = np.nan
            # outdict['pPhi_' + "%03dC_" % (c * 100) + str(l) + "NM_I" + Itag] = pphi
            pPhiList.append(pphi)
            dMag = deltaMag(1, Rp * u.R_jupiter, d * u.AU, pphi)
            dMag[np.isinf(dMag)] = np.nan
            # outdict['dMag_' + "%03dC_" % (c * 100) + str(l) + "NM_I" + Itag] = dMag
            dMagList.append(dMag)

            outdict = {'pl_id': [j] * len(M),
                       'pl_fit_id': [pl_ids[k]] * len(M),
                       'pl_name': [plannames[j]] * len(M),
                       'evalorbit_id': [evalorbit_ids[k]] * len(M),
                       'M': M,
                       't': t - t0.jd,
                       'r': d,
                       's': s * len(M),
                       'WA': WA * len(M),
                       'beta': beta * len(M),
                       'pPhi_' + "%03dC_" % (c * 100) + str(l) + "NM": pphi * len(M),
                       'dMag_' + "%03dC_" % (c * 100) + str(l) + "NM": dMag * len(M),
                       'def_orb': [0] * len(M)}

            out = pandas.DataFrame(outdict)

            if orbdata is None:
                orbdata = out.copy()
            else:
                orbdata = orbdata.append(out)

        # print([plannames[j]] * len(M) * len(slist))
        # print(M * len(slist))
        # print(slist * len(M))

        if orbitfits is None and not skip_orbitfits:
            orbitfits = pandas.DataFrame(orbitfits_dict).copy()
        elif not skip_orbitfits:
            orbitfits = orbitfits.append(pandas.DataFrame(orbitfits_dict))

        if evalorbits is None:
            evalorbits = pandas.DataFrame(evalorbit_dict).copy()
        else:
            evalorbits = evalorbits.append(pandas.DataFrame(evalorbit_dict))

    orbdata = orbdata.sort_values(by=['pl_id', 'evalorbit_id', 't', 'M']).reset_index(drop=True)
    tmpout = {}
    for l in lambdas:
        tmpout["dMag_min_" + str(l) + "NM"] = np.zeros(len(orbdata))
        tmpout["dMag_max_" + str(l) + "NM"] = np.zeros(len(orbdata))
        tmpout["dMag_med_" + str(l) + "NM"] = np.zeros(len(orbdata))
        tmpout["pPhi_min_" + str(l) + "NM"] = np.zeros(len(orbdata))
        tmpout["pPhi_max_" + str(l) + "NM"] = np.zeros(len(orbdata))
        tmpout["pPhi_med_" + str(l) + "NM"] = np.zeros(len(orbdata))

    for name in np.unique(orbdata['pl_name']):
        print(name)
        loc = orbdata['pl_name'] == name
        for l in lambdas:
            tmp = []
            tmp2 = []
            for c in photdict['clouds']:
                tmp.append(orbdata.loc[loc, 'dMag_' + "%03dC_" % (c * 100) + str(l) + "NM"].values)
                tmp2.append(orbdata.loc[loc, 'pPhi_' + "%03dC_" % (c * 100) + str(l) + "NM"].values)
            tmp = np.vstack(tmp)
            tmp2 = np.vstack(tmp2)
            tmpout["dMag_min_" + str(l) + "NM"][loc.values] = np.nanmin(tmp, axis=0)
            tmpout["dMag_max_" + str(l) + "NM"][loc.values] = np.nanmax(tmp, axis=0)
            tmpout["dMag_med_" + str(l) + "NM"][loc.values] = np.nanmedian(tmp, axis=0)
            tmpout["pPhi_min_" + str(l) + "NM"][loc.values] = np.nanmin(tmp2, axis=0)
            tmpout["pPhi_max_" + str(l) + "NM"][loc.values] = np.nanmax(tmp2, axis=0)
            tmpout["pPhi_med_" + str(l) + "NM"][loc.values] = np.nanmedian(tmp2, axis=0)

    orbdata = orbdata.join(pandas.DataFrame(tmpout))

    orbitfits = orbitfits.reset_index(drop=True)
    evalorbits = evalorbits.reset_index(drop=True)

    return orbdata, orbitfits, evalorbits

# Generates fsed based on a random number: 0 <= num < 1
def get_fsed(num):
    """ Generate random value of f_sed based on distribution provided by Mark Marley:
    f_sed             Frequency
    0.000000         0.099
    0.010000         0.001
    0.030000         0.005
    0.100000         0.010
    0.300000         0.025
    1.000000         0.280
    3.000000         0.300
    6.000000         0.280

    Input num is a uniform random value between 0 and 1
    """

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


def calcPlanetCompleteness(data, bandzip, photdict, minangsep=150,maxangsep=450,contrfile='WFIRST_pred_imaging.txt'):
    """ For all known planets in data (output from getIPACdata), calculate obscurational 
    and photometric completeness for those cases where obscurational completeness is 
    non-zero.
    """

    (l,band,bw,ws,wstep) = bandzip[0]
    photinterps2 = photdict['photinterps']
    feinterp = photdict['feinterp']
    distinterp = photdict['distinterp']

    wfirstcontr = np.genfromtxt(contrfile)
    contr = wfirstcontr[:,1]
    angsep = wfirstcontr[:,0] #l/D
    angsep = (angsep * (575.0*u.nm)/(2.37*u.m)*u.rad).decompose().to(u.mas).value #mas
    wfirstc = interp1d(angsep,contr,bounds_error = False, fill_value = 'extrapolate')

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

    #define vectorized f_sed sampler
    vget_fsed = np.vectorize(get_fsed)

    names = []
    WAcs = []
    dMagcs = []
    iinds = []
    jinds = []
    hs = []
    cs = []
    goodinds = []
    for i,j in enumerate(inds):
        row = data.iloc[j] 
        print("%d/%d  %s"%(i+1,len(inds),row['pl_name']))

        #sma distribution        
        amu = row['pl_orbsmax']
        astd = (row['pl_orbsmaxerr1'] - row['pl_orbsmaxerr2'])/2.
        if np.isnan(astd): astd = 0.01*amu
        gena = lambda n: np.clip(np.random.randn(n)*astd + amu,0,np.inf)

        #eccentricity distribution
        emu = row['pl_orbeccen'] 
        if np.isnan(emu):
            gene = lambda n: 0.175/np.sqrt(np.pi/2.)*np.sqrt(-2.*np.log(1 - np.random.uniform(size=n)))
        else:
            estd = (row['pl_orbeccenerr1'] - row['pl_orbeccenerr2'])/2.
            if np.isnan(estd) or (estd == 0):
                estd = 0.01*emu
            gene = lambda n: np.clip(np.random.randn(n)*estd + emu,0,0.99)

        #inclination distribution
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
        
        #arg. of periastron distribution
        wmu = row['pl_orblper']*np.pi/180.0
        if np.isnan(wmu):
            genw = lambda n: np.random.uniform(size=n,low=0.0,high=2*np.pi)
        else:
            wstd = (row['pl_orblpererr1'] - row['pl_orblpererr2'])/2.*np.pi/180.0
            if np.isnan(wstd) or (wstd == 0): 
                wstd = wmu*0.01
            genw = lambda n: np.random.randn(n)*wstd + wmu

        #just a single metallicity
        fe = row['st_metfe']
        if np.isnan(fe): fe = 0.0

        #initialize loops vars
        n = int(1e6)
        c = 0.
        h = np.zeros((len(WAbins)-3, len(dMagbins)-2))
        k = 0.0
        cprev = 0.0
        pdiff = 1.0

        while (pdiff > 0.0001) | (k <3):
            print("%d \t %5.5e \t %5.5e"%( k,pdiff,c))

            #sample orbital parameters
            a = gena(n)
            e = gene(n)
            I = genI(n)
            w = genw(n)

            #sample cloud vals
            cl = vget_fsed(np.random.rand(n))

            #define mass/radius distribution depending on data provenance
            if ((row['pl_radreflink'] ==\
                    '<a refstr="CALCULATED VALUE" href="/docs/composite_calc.html" target=_blank>Calculated Value</a>') | \
                    (row['pl_radreflink'] == \
                    '<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>')):
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

            d = a * (1.0 - e ** 2.0) / (1 + e * np.cos(nu))
            s = d * np.sqrt(4.0 * np.cos(2 * I) + 4 * np.cos(2 * nu + 2.0 * w) - 2.0 * np.cos(-2 * I + 2.0 * nu + 2 * w) - 2 * np.cos(2 * I + 2 * nu + 2 * w) + 12.0) / 4.0
            beta = np.arccos(np.sin(I) * np.sin(nu + w)) * u.rad
            rnorm = d

            lum = row['st_lum']
            teff = row['st_teff']

            ms = MeanStars()

            if np.isnan(lum):
                if np.isnan(teff):
                    lum_fix = 1
                else:
                    lum_fix = (10 ** ms.TeffOther('logL', teff)) ** .5
            else:
                lum_fix = (10 ** lum) ** .5  # Since lum is log base 10 of solar luminosity

            pphi = np.zeros(n)
            for clevel in np.unique(cl):
                tmpinds = cl == clevel
                betatmp = beta[tmpinds]
                binds = np.argsort(betatmp)
                pphi[tmpinds] = (photinterps2[float(feinterp(fe))][float(distinterp(np.mean(rnorm) * lum_fix))][clevel](betatmp.to(u.deg).value[binds],ws).sum(1)*wstep/bw)[np.argsort(binds)].flatten()

            pphi[np.isinf(pphi)] = np.nan 
            pphi[pphi <= 0.0] = 1e-16

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

    outdict = {'cs':cs,
               'goodinds':goodinds,
               'minCWA':minCWA,
               'maxCWA':maxCWA,
               'minCdMag':minCdMag,
               'maxCdMag':maxCdMag}

    tmp = np.full(len(data),np.nan)
    tmp[goodinds] = cs
    data = data.assign(completeness=tmp)

    tmp = np.full(len(data),np.nan)
    tmp[goodinds] = minCWA
    data = data.assign(compMinWA=tmp)
    
    tmp = np.full(len(data),np.nan)
    tmp[goodinds] = maxCWA
    data = data.assign(compMaxWA=tmp)
    
    tmp = np.full(len(data),np.nan)
    tmp[goodinds] = minCdMag
    data = data.assign(compMindMag=tmp)

    tmp = np.full(len(data),np.nan)
    tmp[goodinds] = maxCdMag
    data = data.assign(compMaxdMag=tmp)

    return out2,outdict,data


def genAliases(starnames):
    """ Grab all available aliases for list of targets. """
    
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
        print("%d/%d  %s"%(j+1,len(starnames),star))
        
        #get aliases from IPAC
        r = requests.get(baseurl,{'table':'aliastable','objname':star})
        if "ERROR" not in r.content: 
            tmp = r.content.strip().split("\n")
        else:
            noipacalias.append(star)
            tmp = [star]
        
        #get aliases from SIMBAD
        try:
            r = s.query_object(star)
        except ConnectionError as e:
            try:
                r = s.query_object(star)
            except ConnectionError as e:
                raise e

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


    out3 = pandas.DataFrame({'SID': np.hstack(ids),
                             'Alias': np.hstack(aliases),
                             'NEAName':np.hstack(priname)
                             })

    return out3

def genAllAliases(data):
    starnames = data['pl_hostname'].unique()
    return genAliases(starnames)

def fillMissingAliases(engine,data):
    result = engine.execute("select Alias from Aliases where NEAName=1")
    dbnames = np.hstack(result.fetchall())
    starnames = data['pl_hostname'].unique()
    missing = list(set(starnames) - set(dbnames))

    return genAliases(missing)


def writeSQL(engine,data=None,orbdata=None,altorbdata=None,comps=None,aliases=None):
    """write outputs to sql database via engine"""

    if data is not None:
        print("Writing KnownPlanets")
        namemxchar = np.array([len(n) for n in data['pl_name'].values]).max()
        data.to_sql('KnownPlanets',engine,chunksize=100,if_exists='replace',
                    dtype={'pl_name':sqlalchemy.types.String(namemxchar),
                            'pl_hostname':sqlalchemy.types.String(namemxchar-2),
                            'pl_letter':sqlalchemy.types.CHAR(1)})
        #set indexes
        result = engine.execute("ALTER TABLE KnownPlanets ADD INDEX (pl_name)")
        result = engine.execute("ALTER TABLE KnownPlanets ADD INDEX (pl_hostname)")

        #add comments
        addSQLcomments(engine,'KnownPlanets')
    
    if orbdata is not None:
        print("Writing PlanetOrbits")
        namemxchar = np.array([len(n) for n in orbdata['Name'].values]).max()
        orbdata.to_sql('PlanetOrbits',engine,chunksize=100,if_exists='replace',dtype={'Name':sqlalchemy.types.String(namemxchar)})
        result = engine.execute("ALTER TABLE PlanetOrbits ADD INDEX (Name)")
        result = engine.execute("ALTER TABLE PlanetOrbits ADD FOREIGN KEY (Name) REFERENCES KnownPlanets(pl_name) ON DELETE NO ACTION ON UPDATE NO ACTION");

        addSQLcomments(engine,'PlanetOrbits')

    if altorbdata is not None:
        print("Writing AltPlanetOrbits")
        namemxchar = np.array([len(n) for n in altorbdata['Name'].values]).max()
        altorbdata.to_sql('AltPlanetOrbits',engine,chunksize=100,if_exists='replace',dtype={'Name':sqlalchemy.types.String(namemxchar)})
        result = engine.execute("ALTER TABLE AltPlanetOrbits ADD INDEX (Name)")
        result = engine.execute("ALTER TABLE AltPlanetOrbits ADD FOREIGN KEY (Name) REFERENCES KnownPlanets(pl_name) ON DELETE NO ACTION ON UPDATE NO ACTION");

        addSQLcomments(engine,'AltPlanetOrbits')

    if comps is not None:
        print("Writing Completeness")
        namemxchar = np.array([len(n) for n in comps['Name'].values]).max()
        comps.to_sql('Completeness',engine,chunksize=100,if_exists='replace',dtype={'Name':sqlalchemy.types.String(namemxchar)})
        result = engine.execute("ALTER TABLE Completeness ADD INDEX (Name)")
        result = engine.execute("ALTER TABLE Completeness ADD FOREIGN KEY (Name) REFERENCES KnownPlanets(pl_name) ON DELETE NO ACTION ON UPDATE NO ACTION");

        addSQLcomments(engine,'Completeness')


    if aliases is not None:
        print("Writing Alias")
        aliasmxchar = np.array([len(n) for n in aliases['Alias'].values]).max()
        aliases.to_sql('Aliases',engine,chunksize=100,if_exists='replace',dtype={'Alias':sqlalchemy.types.String(aliasmxchar)})
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (Alias)")
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (SID)")


def addSQLcomments(engine,tablename):
        """Add comments to table schema based on entries in spreadsheet"""
        
        #read in spreadsheet and grab data from appropriate sheet
        coldefs = pandas.ExcelFile('coldefs.xlsx')
        coldefs = coldefs.parse(tablename)
        cols = coldefs['Column'][coldefs['Definition'].notnull()].values
        cdefs = coldefs['Definition'][coldefs['Definition'].notnull()].values
        cnames =  coldefs['Name'][coldefs['Definition'].notnull()].values

        result = engine.execute("show create table %s"%tablename)
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
          comm =  """ALTER TABLE `%s` CHANGE `%s` %s COMMENT "%s %s";"""%(tablename,key,d,cnames[cols == key][0].strip('"'),cdefs[cols == key][0])
          print(comm)
          r = engine.execute(comm)


def writeSQL2(engine, data=None, stdata=None, orbdata=None, evalorbits=None, comps=None, aliases=None):
    """write outputs to sql database via engine"""

    if stdata is not None:
        print("Writing Stars")
        namemxchar = np.array([len(n) for n in stdata['st_name'].values]).max()
        stdata = stdata.rename_axis('st_id')
        stdata.to_sql('Stars', engine, chunksize=100, if_exists='replace',
                    dtype={'st_id': sqlalchemy.types.INT,
                           'st_name': sqlalchemy.types.String(namemxchar)})
        # set indexes
        result = engine.execute('ALTER TABLE Stars ADD INDEX (st_id)')

        # add comments
        # addSQLcomments(engine, 'StarProps')

    if data is not None:
        print("Writing Planets")
        namemxchar = np.array([len(n) for n in data['pl_name'].values]).max()
        data = data.rename_axis('pl_fit_id')
        data.to_sql('Planets',engine,chunksize=100,if_exists='replace',
                    dtype={'pl_id':sqlalchemy.types.INT,
                           'pl_fit_id':sqlalchemy.types.INT,
                            'pl_name':sqlalchemy.types.String(namemxchar),
                            'st_name':sqlalchemy.types.String(namemxchar-2),
                            'pl_letter':sqlalchemy.types.CHAR(1),
                            'st_id': sqlalchemy.types.INT})
        #set indexes
        result = engine.execute("ALTER TABLE Planets ADD INDEX (pl_id)")
        result = engine.execute("ALTER TABLE Planets ADD INDEX (pl_fit_id)")
        result = engine.execute("ALTER TABLE Planets ADD INDEX (st_id)")
        result = engine.execute("ALTER TABLE Planets ADD FOREIGN KEY (st_id) REFERENCES StarProps(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

        #add comments
        # addSQLcomments(engine,'KnownPlanets')

    # if orbitfits is not None:
    #     print("Writing OrbitFits")
    #     namemxchar = np.array([len(n) for n in orbitfits['pl_name'].values]).max()
    #     orbitfits = orbitfits.rename_axis('orbitfits_id')
    #     orbitfits.to_sql('OrbitFits',engine,chunksize=100,if_exists='replace',
    #                       dtype={'pl_id':sqlalchemy.types.INT,
    #                           'pl_name':sqlalchemy.types.String(namemxchar),
    #                              'orbitfits_id':sqlalchemy.types.BIGINT},
    #                       index=True)
    #     result = engine.execute("ALTER TABLE OrbitFits ADD INDEX (orbitfits_id)")
    #     result = engine.execute("ALTER TABLE OrbitFits ADD INDEX (pl_id)")
    #     result = engine.execute("ALTER TABLE OrbitFits ADD FOREIGN KEY (pl_id) REFERENCES KnownPlanets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

        # addSQLcomments(engine,'AltPlanetOrbits')

    if evalorbits is not None:
        print("Writing EvalOrbits")
        evalorbits = evalorbits.rename_axis('evalorbit_id')
        evalorbits.to_sql('EvalOrbits',engine,chunksize=100,if_exists='replace',
                          dtype={'pl_id': sqlalchemy.types.INT,
                                 'pl_fit_id': sqlalchemy.types.INT,
                                 'evalorbit_id': sqlalchemy.types.BIGINT},
                       index=True)
        result = engine.execute("ALTER TABLE EvalOrbits ADD INDEX (evalorbit_id)")
        result = engine.execute("ALTER TABLE EvalOrbits ADD INDEX (pl_id)")
        result = engine.execute("ALTER TABLE EvalOrbits ADD INDEX (pl_fit_id)")
        result = engine.execute("ALTER TABLE EvalOrbits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

        # addSQLcomments(engine,'PlanetOrbits')

    if orbdata is not None:
        print("Writing PlanetOrbits")
        namemxchar = np.array([len(n) for n in orbdata['pl_name'].values]).max()
        orbdata = orbdata.rename_axis('orbit_id')
        orbdata.to_sql('PlanetOrbits',engine,chunksize=100,if_exists='replace',
                       dtype={'pl_name':sqlalchemy.types.String(namemxchar),
                              'pl_id': sqlalchemy.types.INT,
                              'pl_fit_id': sqlalchemy.types.INT,
                              'orbit_id': sqlalchemy.types.BIGINT,
                              'evalorbit_id': sqlalchemy.types.BIGINT},
                       index=True)
        result = engine.execute("ALTER TABLE PlanetOrbits ADD INDEX (orbit_id)")
        result = engine.execute("ALTER TABLE PlanetOrbits ADD INDEX (pl_id)")
        result = engine.execute("ALTER TABLE PlanetOrbits ADD INDEX (pl_fit_id)")
        result = engine.execute("ALTER TABLE PlanetOrbits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION");
        result = engine.execute("ALTER TABLE PlanetOrbits ADD FOREIGN KEY (evalorbit_id) REFERENCES EvalOrbits(evalorbit_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

        # addSQLcomments(engine,'PlanetOrbits')

    if comps is not None:
        print("Writing Completeness")
        namemxchar = np.array([len(n) for n in comps['pl_name'].values]).max()
        comps = comps.rename_axis('completeness_id')
        comps.to_sql('Completeness',engine,chunksize=100,if_exists='replace',
                     dtype={'Name':sqlalchemy.types.String(namemxchar),
                            'pl_id': sqlalchemy.types.INT,
                            'pl_fit_id': sqlalchemy.types.INT})
        result = engine.execute("ALTER TABLE Completeness ADD INDEX (pl_fit_id)")
        result = engine.execute("ALTER TABLE Completeness ADD INDEX (pl_id)")
        result = engine.execute("ALTER TABLE Completeness ADD INDEX (completeness_id)")
        result = engine.execute("ALTER TABLE Completeness ADD FOREIGN KEY (pl_fit_id) REFERENCES Planets(pl_fit_id) ON DELETE NO ACTION ON UPDATE NO ACTION");
        result = engine.execute("ALTER TABLE Completeness ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

        # addSQLcomments(engine,'Completeness')

    if aliases is not None:
        print("Writing Alias")
        aliasmxchar = np.array([len(n) for n in aliases['Alias'].values]).max()
        aliases.to_sql('Aliases',engine,chunksize=100,if_exists='replace',dtype={'Alias':sqlalchemy.types.String(aliasmxchar)})
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (Alias)")
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (SID)")
        result = engine.execute("ALTER TABLE Aliases ADD FOREIGN KEY (Name) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION");


def addSQLcomments2(engine,tablename):
        """Add comments to table schema based on entries in spreadsheet"""

        #read in spreadsheet and grab data from appropriate sheet
        coldefs = pandas.ExcelFile('coldefs.xlsx')
        coldefs = coldefs.parse(tablename)
        cols = coldefs['Column'][coldefs['Definition'].notnull()].values
        cdefs = coldefs['Definition'][coldefs['Definition'].notnull()].values
        cnames =  coldefs['Name'][coldefs['Definition'].notnull()].values

        result = engine.execute("show create table %s"%tablename)
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
          comm =  """ALTER TABLE `%s` CHANGE `%s` %s COMMENT "%s %s";"""%(tablename,key,d,cnames[cols == key][0].strip('"'),cdefs[cols == key][0])
          print(comm)
          r = engine.execute(comm)

# Separates data into pl_data and st_data
def generateTables(data, orbitfits_data):
    print('Splitting IPAC data into Star and Planet data')
    # Isolate the stellar columns from the planet columns
    st_cols = [col for col in data.columns if ('st_' in col) or ('gaia_' in col)]
    st_cols.extend(["dec", "dec_str", "hd_name", "hip_name", "ra", "ra_str"])
    # print(st_cols)
    pl_cols = np.setdiff1d(data.columns, st_cols)
    st_cols.append("pl_hostname")
    st_cols.remove("pl_st_npar")
    st_cols.remove("pl_st_nref")
    st_data = data[st_cols]
    st_data.drop_duplicates(keep='first', inplace=True)
    st_data.reset_index(drop=True)

    # pl_cols = [col for col in pl_cols if 'pl_' not in col]
    # pl_cols = np.append(pl_cols, ['pl_name', 'pl_letter', 'pl_hostname'])

    # pl_cols = ['pl_name', 'pl_letter', 'pl_hostname']

    # print(pl_cols)5

    # pl_data = data[pl_cols]
    st_data = st_data.rename(columns={'pl_hostname':'st_name'})
    orbitfits_data = orbitfits_data.rename(columns={'hostname':'st_name'})

    # print(orbitfits_data['pl_name'].values)

    pl_data = orbitfits_data
    # print(pl_data.columns)
    # [print(col) for col in pl_data.columns]

    num_rows = len(st_data.index)
    stars = pl_data['st_name']
    stars.drop_duplicates(keep='first', inplace=True)

    num_stars = len(stars.index)
    # If these aren't equal, then two sources of data for a star don't agree.
    assert num_rows == num_stars, "Stellar data for different planets not consistent"

    idx_list = []
    for idx, row in pl_data.iterrows():
        st_idx = st_data[st_data['st_name'] == row['st_name']].index.values
        # If there's no star associated with a planet, should have null index
        if len(st_idx) == 0:
            idx_list.append(None)
        else:
            idx_list.append(st_idx[0])

    pl_data = pl_data.assign(st_id=idx_list)

    # colmap_pl = {k: k[1:] if (k.startswith('pl_') and (k != 'pl_id' or k != 'pl_name')) else k for k in pl_data.keys()}
    colmap_st = {k: k[3:] if (k.startswith('st_') and (k != 'st_id' and k != 'st_name')) else k for k in st_data.keys()}
    # pl_data = pl_data.rename(columns=colmap_pl)
    st_data = st_data.rename(columns=colmap_st)

    return pl_data, st_data


# Adds a pl_id column to data using the planet data specified by pl_data whose original column name is pl_name
def add_pl_idx(data, pl_data, pl_name):
    idx_list = []
    data = data.rename(columns={pl_name: 'pl_name'})
    for idx, row in data.iterrows():
        pl_idx_int = pl_data[pl_data['pl_name'] == row['pl_name']]#.index.values
        pl_idx = pl_idx_int[pl_idx_int['def_pl'] == 1].index.values
        # pl_idx_orig = pl_idx_int[pl_idx_int['def_pl'] == 1]['pl_id'].values
        if len(pl_idx) == 0:
            idx_list.append(None)
        else:
            idx_list.append(pl_idx[0])

    data = data.assign(pl_fit_id=idx_list)

    idx_list_2 = []
    for idx, row in data.iterrows():
        pl_idx_int = pl_data[pl_data['pl_name'] == row['pl_name']]  # .index.values
        pl_idx_row = pl_idx_int[pl_idx_int['def_pl'] == 1]
        pl_idx = pl_idx_row['pl_id'].values
        # pl_idx_orig = pl_idx_int[pl_idx_int['def_pl'] == 1]['pl_id'].values
        if len(pl_idx) == 0:
            idx_list_2.append(None)
        else:
            idx_list_2.append(pl_idx[0])

    data = data.assign(pl_id=idx_list_2)
    return data