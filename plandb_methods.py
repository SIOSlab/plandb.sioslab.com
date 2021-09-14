from __future__ import division, print_function

import math
import os
import re
from io import BytesIO

import astropy.constants as const
import astropy.units as u
import EXOSIMS.PlanetPhysicalModel.Forecaster
import numpy as np
import pandas as pd
import requests
import sqlalchemy.types
from astropy.time import Time
from astroquery.simbad import Simbad
from EXOSIMS.PlanetPhysicalModel.ForecasterMod import ForecasterMod
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.getExoplanetArchive import (getExoplanetArchivePS,
                                              getExoplanetArchivePSCP)
from MeanStars import MeanStars
from requests.exceptions import ConnectionError
from scipy.interpolate import RectBivariateSpline, griddata, interp1d, interp2d
from sqlalchemy import create_engine
from tqdm import trange, tqdm

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO


def getIPACdata():
    '''
    grab everything from exoplanet and composite tables, merge,
    add additional column info and return/save to disk
    '''

    print("Querying IPAC for all data.")
    pscp_data = getExoplanetArchivePSCP() # Composite table
    ps_data = getExoplanetArchivePS()

    #only keep stuff related to the star from composite
    composite_cols = [col for col in pscp_data.columns if (col == 'pl_name') or
                      str.startswith(col, 'st_')]
    # composite_cols.extend(['pl_radj', 'pl_radj_reflink'])
    pscp_data = pscp_data[composite_cols]

    #sort by planet name
    pscp_data = pscp_data.sort_values(by=['pl_name']).reset_index(drop=True)
    ps_data = ps_data.sort_values(by=['pl_name']).reset_index(drop=True)

    #merge data sets
    merged_data = ps_data.copy()
    blank_col = [None]*len(ps_data)
    for col in composite_cols:
        if col not in merged_data.columns:
            merged_data[col] = blank_col
    # Writing as a loop for now...
    t_bar = trange(len(merged_data), leave=False)
    last_pl = None
    for i, row in merged_data.iterrows():
        t_bar.update(1)
        pl = row.pl_name

        if pl != last_pl:
            # Get the composite row
            c_row = pscp_data.loc[pscp_data.pl_name == row.pl_name]

        # Now take all the stellar data from the composite table
        for col in composite_cols:
            if col != 'pl_name':
                merged_data.at[i, col] = c_row[col].values[0]

        last_pl = pl
    t_bar.close()
    #create columns for short references and publication years
    authregex = re.compile(r"(<a.*f>)|(</a>)")
    merged_data['publication_year'] = merged_data.pl_refname.str.extract('(\d{4})')
    merged_data['shortrefs'] = [authregex.sub("", merged_data['pl_refname'][j]).strip() for j in range(len(merged_data))]
    merged_data['refs'] = merged_data["pl_refname"].values
    merged_data['best_data'] = np.zeros(len(merged_data))

    #pick best attribute row for each planet
    print("Choosing best attributes for all planets.")
    # This is used for typesetting the progress bar
    max_justification = merged_data.pl_name.str.len().max()
    t_bar = trange(len(merged_data), leave=False)
    for j,name in enumerate(merged_data['pl_name'].values):
        # print("%s: %d/%d"%(name,j+1,len(data)))
        t_bar.set_description(name.ljust(max_justification))
        t_bar.update(1)

        planet_rows = merged_data.loc[merged_data["pl_name"] == name]

        sorted_rows = planet_rows.sort_values(by=["publication_year"], axis=0, ascending=False)
        good_idx = sorted_rows.index[0]
        good_lvl = 0
        for index, row in sorted_rows.iterrows():
            base_need = (not pd.isnull(row["pl_orbsmax"]) or not pd.isnull(row["pl_orbper"])) and \
                        (not pd.isnull(row["pl_bmassj"]) or not pd.isnull(row["pl_radj"]))

            # Has everything
            if good_lvl < 8 and (base_need
                                 and not pd.isnull(row["pl_orbeccen"]) and not pd.isnull(row["pl_orbtper"])
                                 and not pd.isnull(row["pl_orblper"]) and not pd.isnull(row["pl_orbincl"])):
                if not pd.isnull(row["pl_radj"]):
                    good_idx = index
                    good_lvl = 8
                    break
                elif good_lvl < 7:
                    good_idx = index
                    good_lvl = 7

            # Has everything except inclination
            if good_lvl < 6 and (base_need
                                 and not pd.isnull(row["pl_orbeccen"]) and not pd.isnull(row["pl_orbtper"])
                                 and not pd.isnull(row["pl_orblper"])):
                if not pd.isnull(row["pl_radj"]):
                    good_idx = index
                    good_lvl = 6
                elif good_lvl < 5:
                    good_idx = index
                    good_lvl = 5

            # Has either periapsis time or argument of pariapsis
            elif good_lvl < 4 and (base_need
                                   and not pd.isnull(row["pl_orbeccen"]) and (not pd.isnull(row["pl_orbtper"])
                                                                          or not pd.isnull(row["pl_orblper"]))):
                if not pd.isnull(row["pl_radj"]):
                    good_idx = index
                    good_lvl = 4
                elif good_lvl < 3:
                    good_idx = index
                    good_lvl = 3

            # Has eccentricity
            elif good_lvl < 2 and (base_need
                                   and not pd.isnull(row["pl_orbeccen"])):
                if not pd.isnull(row["pl_radj"]):
                    good_idx = index
                    good_lvl = 2
                elif good_lvl < 1:
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

        merged_data.at[good_idx, "best_data"] = 1

    columns_to_null = ["pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_orbperlim", "pl_orbsmax",
                          "pl_orbsmaxerr1", "pl_orbsmaxerr2", "pl_orbsmaxlim", "pl_orbeccen",
                          "pl_orbeccenerr1", "pl_orbeccenerr2", "pl_orbeccenlim", "pl_orbtper",
                          "pl_orbtpererr1", "pl_orbtpererr2", "pl_orbtperlim", "pl_orblper",
                          "pl_orblpererr1", "pl_orblpererr2", "pl_orblperlim", "pl_bmassj",
                          "pl_bmassjerr1", "pl_bmassjerr2", "pl_bmassjlim",  "pl_orbincl", "pl_orbinclerr1", "pl_orbinclerr2",
                          "pl_orbincllim", "pl_bmassprov"]
    rad_columns = ["pl_radj", "pl_radjerr1", "pl_radjerr2", "pl_radjlim"]
    final_replace_columns = list(columns_to_null)
    final_replace_columns.extend(rad_columns)

    #update rows as needed
    print("Updating planets with best attributes.")
    max_justification = pscp_data.pl_name.str.len().max()
    t_bar = trange(len(pscp_data), leave=True)
    merged_data = merged_data.assign(pl_def_override=np.zeros(len(merged_data)))
    # return ps_data, pscp_data, merged_data
    # for j,name in enumerate(merged_data['pl_name'].values):
        # # print("%s: %d/%d"%(name,j+1,len(data)))
        # t_bar.set_description(name.ljust(max_justification))
        # t_bar.update(1)

        # row_data = merged_data.loc[(merged_data["best_data"] == 1) & (merged_data["pl_name"] == name)]
        # idx = merged_data.loc[(merged_data["pl_name"] == name)].index

        # # row_data.index = idx
        # row_data_replace = row_data[final_replace_columns]
        # # Want to keep radius vals from composite table instead of replacing with null, so we don't null radius columns
        # merged_data.loc[(merged_data["pl_name"] == name,columns_to_null)] = np.nan
        # merged_data.update(row_data_replace,overwrite=True)
        # merged_data.loc[idx,'pl_def_override'] = 1
        # merged_data.loc[idx, 'disc_refname'] = row_data['disc_refname'].values[0]

        # if not np.isnan(row_data['pl_radj'].values[0]):
            # merged_data.loc[idx, 'pl_radreflink'] = row_data['disc_refname'].values[0]

    # Drop rows that aren't marked as the best data
    merged_data = merged_data.loc[merged_data.best_data == 1]
    # return ps_data, pscp_data, merged_data

    #sort by planet name
    merged_data = merged_data.sort_values(by=['pl_name']).reset_index(drop=True)

    print("Filtering to useable planets and calculating additional properties.")

    # filter rows:
    # we need:
    # distance AND
    # (sma OR (period AND stellar mass)) AND
    # (radius OR mass (either true or m\sin(i)))
    # return ps_data, pscp_data, merged_data
    keep = (~np.isnan(merged_data['sy_dist'].values)) & (~np.isnan(merged_data['pl_orbsmax'].values) | \
            (~np.isnan(merged_data['pl_orbper'].values) & ~np.isnan(merged_data['st_mass'].values))) & \
           (~np.isnan(merged_data['pl_bmassj'].values) | ~np.isnan(merged_data['pl_radj'].values))
    merged_data = merged_data[keep]
    merged_data = merged_data.reset_index(drop=True)

    #remove extraneous columns
    merged_data = merged_data.drop(columns=['pl_rade',
                              'pl_radelim',
                              # 'pl_radserr2',
                              'pl_radeerr1',
                              # 'pl_rads',
                              # 'pl_radslim',
                              'pl_radeerr2',
                              # 'pl_radserr1',
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
    nosma = np.isnan(merged_data['pl_orbsmax'].values)
    p2sma = lambda mu,T: ((mu*T**2/(4*np.pi**2))**(1/3.)).to('AU')
    GMs = const.G*(merged_data['st_mass'][nosma].values*u.solMass) # units of solar mass
    T = merged_data['pl_orbper'][nosma].values*u.day
    tmpsma = p2sma(GMs,T)
    merged_data.loc[nosma,'pl_orbsmax'] = tmpsma
    merged_data['pl_calc_sma'] = pd.Series(np.zeros(len(merged_data['pl_name'])), index=merged_data.index)
    merged_data.loc[nosma, 'pl_calc_sma'] = 1

    #propagate filled in sma errors
    GMerrs = ((merged_data['st_masserr1'][nosma] - merged_data['st_masserr2'][nosma])/2.).values*u.solMass*const.G
    Terrs = ((merged_data['pl_orbpererr1'][nosma] - merged_data['pl_orbpererr2'][nosma])/2.).values*u.day

    smaerrs = np.sqrt((2.0*T**2.0*GMs)**(2.0/3.0)/(9*np.pi**(4.0/3.0)*T**2.0)*Terrs**2.0 +\
            (2.0*T**2.0*GMs)**(2.0/3.0)/(36*np.pi**(4.0/3.0)*GMs**2.0)*GMerrs**2.0).to('AU')
    merged_data.loc[nosma,'pl_orbsmaxerr1'] = smaerrs
    merged_data.loc[nosma,'pl_orbsmaxerr2'] = -smaerrs


    #update all WAs (and errors) based on sma
    WA = np.arctan((merged_data['pl_orbsmax'].values*u.AU)/(merged_data['sy_dist'].values*u.pc)).to('mas')
    merged_data['pl_angsep'] = WA.value
    sigma_a = ((merged_data['pl_orbsmaxerr1']- merged_data['pl_orbsmaxerr2'])/2.).values*u.AU
    sigma_d = ((merged_data['sy_disterr1']- merged_data['sy_disterr2'])/2.).values*u.pc
    sigma_wa = (np.sqrt(( (merged_data['pl_orbsmax'].values*u.AU)**2.0*sigma_d**2 + (merged_data['sy_dist'].values*u.pc)**2.0*sigma_a**2)/\
            ((merged_data['pl_orbsmax'].values*u.AU)**2.0 + (merged_data['sy_dist'].values*u.pc)**2.0)**2.0).decompose()*u.rad).to(u.mas)
    merged_data['pl_angseperr1'] = sigma_wa.value
    merged_data['pl_angseperr2'] = -sigma_wa.value

    #fill in radius based on mass
    # noR = ((merged_data['pl_rade_reflink'] == '<a refstr="CALCULATED VALUE" href="/docs/composite_calc.html" target=_blank>Calculated Value</a>') |\
           # (merged_data['pl_rade_reflink'] == '<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>') |\
            # merged_data['pl_radj'].isnull()).values
    # merged_data['pl_calc_rad'] = pd.Series(np.zeros(len(merged_data['pl_name'])), index=merged_data.index)
    # merged_data.loc[noR, 'pl_calc_rad'] = 1
    noR = (merged_data['pl_radj'].isnull()).values

    # Initialize the ForecasterMod
    forecaster_mod = ForecasterMod()
    merged_data.loc[3856, 'pl_bmassjerr2'] = -.18
    merged_data.loc[3935, 'pl_orbsmaxerr2'] = -0.6
    # merged_data.loc[3905, 'pl_bmassjerr1'] = 0.1
    m = ((merged_data['pl_bmassj'][noR].values*u.M_jupiter).to(u.M_earth)).value
    merr = (((merged_data['pl_bmassjerr1'][noR].values - merged_data['pl_bmassjerr2'][noR].values)/2.0)*u.M_jupiter).to(u.M_earth).value
    R = forecaster_mod.calc_radius_from_mass(m*u.M_earth)
    # R = [forecaster_m.calc_radius_from_mass(mp*u.M_earth) for mp in m]

    # Turning what was a list comprehension into a loop to handle edge case
    Rerr = np.zeros(len(m))
    for j, m_val in enumerate(m):
        if np.isnan(merr[j]):
            Rerr[j] = np.nan
        elif merr[j] == 0:
            # Happens with truncation error sometimes, check if the error in earth radii is input to IPAC correctly
            merr_earth = (((merged_data.iloc[j]['pl_bmasseerr1'] - merged_data.iloc[j]['pl_bmasseerr2'])/2.0)*u.M_earth).to(u.M_earth).value
            Rerr[j] = forecaster_mod.calc_radius_from_mass(u.M_earth*np.random.normal(loc=m_val, scale=merr_earth, size=int(1e4))).std().value
        else:
            Rerr[j] = forecaster_mod.calc_radius_from_mass(u.M_earth*np.random.normal(loc=m_val, scale=merr[j], size=int(1e4))).std().value
    Rerr = Rerr*u.R_earth

    # Rerr = np.array([forecaster_mod.calc_radius_from_mass(u.M_earth*np.random.normal(loc=m[j], scale=merr[j], size=int(1e4))).std().value if not(np.isnan(merr[j])) else np.nan for j in range(len(m))])*u.R_earth

    #create mod forecaster radius column and error cols
    # breakpoint()
    # merged_data = merged_data.assign(pl_radj_forecastermod=merged_data['pl_radj'].values)
    merged_data['pl_radj_forecastermod'] = merged_data['pl_radj'].values
    merged_data.loc[noR,'pl_radj_forecastermod'] = (R.to(u.R_jupiter)).value

    # merged_data = merged_data.assign(pl_radj_forecastermoderr1=merged_data['pl_radjerr1'].values)
    merged_data['pl_radj_forecastermoderr1'] = merged_data['pl_radjerr1'].values
    merged_data.loc[noR,'pl_radj_forecastermoderr1'] = (Rerr.to(u.R_jupiter)).value

    # merged_data = merged_data.assign(pl_radj_forecastermoderr2=merged_data['pl_radjerr2'].values)
    merged_data['pl_radj_forecastermoderr2'] = merged_data['pl_radjerr2'].values
    merged_data.loc[noR,'pl_radj_forecastermoderr2'] = -(Rerr.to(u.R_jupiter)).value



    # now the Fortney model
    from EXOSIMS.PlanetPhysicalModel.FortneyMarleyCahoyMix1 import \
        FortneyMarleyCahoyMix1
    fortney = FortneyMarleyCahoyMix1()

    ml10 = m <= 17
    Rf = np.zeros(m.shape)
    Rf[ml10] = fortney.R_ri(0.67,m[ml10])

    mg10 = m > 17
    tmpsmas = merged_data['pl_orbsmax'][noR].values
    tmpsmas = tmpsmas[mg10]
    tmpsmas[tmpsmas < fortney.giant_pts2[:,1].min()] = fortney.giant_pts2[:,1].min()
    tmpsmas[tmpsmas > fortney.giant_pts2[:,1].max()] = fortney.giant_pts2[:,1].max()

    tmpmass = m[mg10]
    tmpmass[tmpmass > fortney.giant_pts2[:,2].max()] = fortney.giant_pts2[:,2].max()

    Rf[mg10] = griddata(fortney.giant_pts2, fortney.giant_vals2,( np.array([10.]*np.where(mg10)[0].size), tmpsmas, tmpmass))

    # merged_data = merged_data.assign(pl_radj_fortney=merged_data['pl_radj'].values)
    merged_data['pl_radj_fortney'] = merged_data['pl_radj'].values
    #data['pl_radj_fortney'][noR] = ((Rf*u.R_earth).to(u.R_jupiter)).value
    merged_data.loc[noR,'pl_radj_fortney'] = ((Rf*u.R_earth).to(u.R_jupiter)).value

        # Calculate erros for fortney radius
    Rf_err = []
    tmpsmas = merged_data['pl_orbsmax'][noR].values
    tmpsmaserr = (merged_data['pl_orbsmaxerr1'][noR].values - merged_data['pl_orbsmaxerr2'][noR].values) / 2.0
    adist = np.zeros((len(m), int(1e4)))
    # breakpoint()
    for j, _ in enumerate(tmpsmas):  # Create smax distribution
        if np.isnan(tmpsmaserr[j]) or (tmpsmaserr[j] == 0):
            adist[j, :] = (np.ones(int(1e4))* tmpsmas[j])
        else:
            adist[j, :] = (np.random.normal(loc=tmpsmas[j], scale=tmpsmaserr[j], size=int(1e4)))

    for j, _ in enumerate(m):  # Create m distribution and calculate errors
        if np.isnan(merr[j]) or (merr[j] == 0):
            # merr[j] = 0
            mdist = np.ones(int(1e4)) * m[j]
        else:
            mdist = (np.random.normal(loc=m[j], scale=merr[j], size=int(1e4)))

        for i in range(len(mdist)):
            while mdist[i] < 0:
                mdist[i] = (np.random.normal(loc=m[j], scale=merr[j], size=int(1)))

        ml10_dist = mdist <= 17
        Rf_dist = np.zeros(mdist.shape)
        Rf_dist[ml10_dist] = fortney.R_ri(0.67, mdist[ml10_dist])

        cur_adist = adist[j, :]
        mg10_dist = mdist > 17
        tmpsmas = cur_adist[mg10_dist]
        tmpsmas[tmpsmas < fortney.giant_pts2[:, 1].min()] = fortney.giant_pts2[:, 1].min()
        tmpsmas[tmpsmas > fortney.giant_pts2[:, 1].max()] = fortney.giant_pts2[:, 1].max()

        tmpmass = mdist[mg10_dist]
        tmpmass[tmpmass > fortney.giant_pts2[:, 2].max()] = fortney.giant_pts2[:, 2].max()

        Rf_dist[mg10_dist] = griddata(fortney.giant_pts2, fortney.giant_vals2,
                                      (np.array([10.] * np.where(mg10_dist)[0].size), tmpsmas, tmpmass))
        Rf_err.append(Rf_dist.std())

    # merged_data = merged_data.assign(pl_radj_fortneyerr1=merged_data['pl_radjerr1'].values)
    merged_data['pl_radj_fortneyerr1'] = merged_data['pl_radjerr1'].values
    merged_data.loc[noR, 'pl_radj_fortneyerr1'] = ((Rf_err * u.R_earth).to(u.R_jupiter)).value
    # merged_data = merged_data.assign(pl_radj_forecastermoderr2=merged_data['pl_radjerr2'].values)
    merged_data['pl_radj_fortneyerr2'] = merged_data['pl_radjerr2'].values
    merged_data.loc[noR, 'pl_radj_fortneyerr2'] = -((Rf_err * u.R_earth).to(u.R_jupiter)).value


    #populate max WA based on available eccentricity data (otherwise maxWA = WA)
    hase = ~np.isnan(merged_data['pl_orbeccen'].values)
    maxWA = WA[:]
    maxWA[hase] = np.arctan((merged_data['pl_orbsmax'][hase].values*(1 + merged_data['pl_orbeccen'][hase].values)*u.AU)/(merged_data['sy_dist'][hase].values*u.pc)).to('mas')
    # merged_data = merged_data.assign(pl_maxangsep=maxWA.value)
    merged_data['pl_maxangsep'] = maxWA.value

    #populate min WA based on eccentricity & inclination data (otherwise minWA = WA)
    hasI =  ~np.isnan(merged_data['pl_orbincl'].values)
    s = merged_data['pl_orbsmax'].values*u.AU
    s[hase] *= (1 - merged_data['pl_orbeccen'][hase].values)
    s[hasI] *= np.cos(merged_data['pl_orbincl'][hasI].values*u.deg)
    s[~hasI] = 0
    minWA = np.arctan(s/(merged_data['sy_dist'].values*u.pc)).to('mas')
    # merged_data = merged_data.assign(pl_minangsep=minWA.value)
    merged_data['pl_minangsep'] =minWA.value

    #Fill in missing luminosity from meanstars
    ms = MeanStars()
    nolum_teff = np.isnan(merged_data['st_lum'].values) & ~np.isnan(merged_data['st_teff'].values)
    teffs = merged_data.loc[nolum_teff, 'st_teff']
    lums_1 = ms.TeffOther('logL', teffs) # Calculates Luminosity when teff exists
    merged_data.loc[nolum_teff, 'st_lum'] = lums_1

    nolum_noteff_spect = np.isnan(merged_data['st_lum'].values) & ~merged_data['st_spectype'].isnull().values
    spects = merged_data.loc[nolum_noteff_spect, 'st_spectype']
    lums2 = []
    for str_row in spects.values:
        spec_letter = str_row[0]
        # spec_rest.append(str_row[1:])
        spec_num_match = re.search("^[0-9,]+", str_row[1:])
        if spec_num_match is not None:
            spec_num = str_row[spec_num_match.start() + 1:spec_num_match.end() + 1]
            # Calculates luminosity when teff does not exist but spectral type exists
            lums2.append(ms.SpTOther('logL', spec_letter, spec_num))
        else:
            lums2.append(np.nan)
    merged_data.loc[nolum_noteff_spect, 'st_lum'] = lums2

    return merged_data


def packagePhotometryData(dbfile=None):
    """
    Read photometry data from database of grid models and repackage into single ndarray of data
    """
    if dbfile is None:
        dbfile = os.path.join(os.getenv('HOME'),'Documents','AFTA-Coronagraph','ColorFun','AlbedoModels_2015.db')

    # grab photometry data
    enginel = create_engine('sqlite:///' + dbfile)

    # getting values
    meta_alb = pd.read_sql_table('header',enginel)
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

    tmp = pd.read_sql_table('g25_t150_m0.0_d0.5_NC_phang000',enginel)
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
                        tmp = pd.read_sql_table(name,enginel)
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
    fes = data['st_met'].values
    fes[np.isnan(fes)] = 0.0
    Rps = data['pl_radj_forecastermod'].values
    inc = data['pl_orbincl'].values
    eccen = data['pl_orbeccen'].values
    arg_per = data['pl_orblper'].values
    lum = data['st_lum'].values

    tmpout = {}

    quadinterps = photdict['quadinterps']
    feinterp = photdict['feinterp']
    distinterp = photdict['distinterp']
    lambdas = []

    #iterate over all data rows
    t_bar = trange(len(Rps), leave=False)
    for j, (Rp, fe,a, I, e, w, lm) in enumerate(zip(Rps, fes,smas, inc, eccen, arg_per, lum)):
        t_bar.update(1)
        # print("%d/%d"%(j+1,len(Rps)))
        for c in photdict['clouds']:
            for l,band,bw,ws,wstep in bandzip:
                if j == 0:
                    lambdas.append(l)
                    #allocate output arrays
                    tmpout['quad_pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"] = np.zeros(smas.shape)
                    tmpout['quad_dMag_'+"%03dC_"%(c*100)+str(l)+"NM"] = np.zeros(smas.shape)
                    tmpout['quad_radius_' + "%03dC_" % (c * 100) + str(l) + "NM"] = np.zeros(smas.shape)

                if np.isnan(lm):
                    lum_fix = 1
                else:
                    lum_fix = (10 ** lm) ** .5  # Since lum is log base 10 of solar luminosity


                #Only calculate quadrature distance if known eccentricity and argument of periaps,
                #and not face-on orbit
                if not np.isnan(e) and not np.isnan(w) and I != 0:
                    nu1 = -w
                    nu2 = np.pi - w

                    r1 = a * (1.0 - e ** 2.0) / (1.0 + e * np.cos(nu1)) / lum_fix
                    r2 = a * (1.0 - e ** 2.0) / (1.0 + e * np.cos(nu2)) / lum_fix

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
                    pphi = quadinterps[float(feinterp(fe))][float(distinterp(a/lum_fix))][c](ws).sum()*wstep/bw
                    if np.isinf(pphi):
                        print("Inf value encountered in pphi")
                        pphi = np.nan

                    tmpout['quad_pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"][j] = pphi
                    dMag = deltaMag(1, Rp*u.R_jupiter, a*u.AU, pphi)
                    tmpout['quad_radius_' + "%03dC_" % (c * 100) + str(l) + "NM"][j] = a / lum_fix

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

    data = data.join(pd.DataFrame(tmpout))

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
    lambdas = [l for l,band,bw,ws,wstep in bandzip]


    orbdata = None
    t_bar = trange(len(plannames), leave=False)
    for j in range(len(plannames)):
        t_bar.update(1)
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
        dist = row['sy_dist']
        fe = row['st_met']
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
        beta = np.arccos(-np.sin(I)*np.sin(nu+w))*u.rad

        WA = np.arctan((s*u.AU)/(dist*u.pc)).to('mas').value
        # print(j,plannames[j],WA.min() - minWA[j].value, WA.max() - maxWA[j].value)

        outdict = {'Name': [plannames[j]]*len(M),
                    'M': M,
                    't': t-t0.jd,
                    'r': d,
                    's': s,
                    'WA': WA,
                    'beta': beta.to(u.deg).value}

        lum = row['st_lum']
        if np.isnan(lum):
            lum_fix = 1
        else:
            lum_fix = (10 ** lum) ** .5  # Since lum is log base 10 of solar luminosity


        alldMags = np.zeros((len(photdict['clouds']), len(lambdas), len(beta)))
        allpphis = np.zeros((len(photdict['clouds']), len(lambdas), len(beta)))

        inds = np.argsort(beta)
        for count1,c in enumerate(photdict['clouds']):
            for count2,(l,band,bw,ws,wstep) in enumerate(bandzip):
                pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a/lum_fix))][c](beta.to(u.deg).value[inds],ws).sum(1)*wstep/bw)[np.argsort(inds)]
                pphi[np.isinf(pphi)] = np.nan
                outdict['pPhi_'+"%03dC_"%(c*100)+str(l)+"NM"] = pphi
                allpphis[count1,count2] = pphi
                dMag = deltaMag(1, Rp*u.R_jupiter, d*u.AU, pphi)
                dMag[np.isinf(dMag)] = np.nan
                outdict['dMag_'+"%03dC_"%(c*100)+str(l)+"NM"] = dMag
                alldMags[count1,count2] = dMag


        pphismin = np.nanmin(allpphis,axis=0)
        dMagsmin = np.nanmin(alldMags,axis=0)
        pphismax = np.nanmax(allpphis,axis=0)
        dMagsmax = np.nanmax(alldMags,axis=0)
        pphismed = np.nanmedian(allpphis,axis=0)
        dMagsmed = np.nanmedian(alldMags,axis=0)

        for count3,l in enumerate(lambdas):
            outdict["dMag_min_"+str(l)+"NM"] = dMagsmin[count3]
            outdict["dMag_max_"+str(l)+"NM"] = dMagsmax[count3]
            outdict["dMag_med_"+str(l)+"NM"] = dMagsmed[count3]
            outdict["pPhi_min_"+str(l)+"NM"] = pphismin[count3]
            outdict["pPhi_max_"+str(l)+"NM"] = pphismax[count3]
            outdict["pPhi_med_"+str(l)+"NM"] = pphismed[count3]

        out = pd.DataFrame(outdict)

        if orbdata is None:
            orbdata = out.copy()
        else:
            orbdata = orbdata.append(out)

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
    t_bar = trange(len(plannames), leave=False)
    for j in range(len(plannames)):
        row = data.iloc[j]
        # print("%d/%d  %s"%(j+1,len(plannames),plannames[j]))
        t_bar.set_description(plannames[j].ljust(20))
        t_bar.update(1)

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
        dist = row['sy_dist']
        fe = row['st_met']
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
        if np.isnan(lum):
            lum_fix = 1
        else:
            lum_fix = (10 ** lum) ** .5  # Since lum is log base 10 of solar luminosity

        for k,I in enumerate(Is):
            s = d * np.sqrt(4.0 * np.cos(2 * I) + 4 * np.cos(2 * nu + 2.0 * w) - 2.0 * np.cos(-2 * I + 2.0 * nu + 2 * w) - 2 * np.cos(2 * I + 2 * nu + 2 * w) + 12.0) / 4.0
            beta = np.arccos(-np.sin(I) * np.sin(nu + w)) * u.rad

            WA = np.arctan((s*u.AU)/(dist*u.pc)).to('mas').value

            if I == Icrit:
                Itag = "crit"
            else:
                Itag = "%02d"%(Isglob[k])

            outdict["s_I"+Itag] = s
            outdict["WA_I"+Itag] =  WA
            outdict["beta_I"+Itag] = beta.to(u.deg).value

            inds = np.argsort(beta)
            pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a / lum_fix))][c](beta.to(u.deg).value[inds],ws).sum(1)*wstep/bw)[np.argsort(inds)]
            pphi[np.isinf(pphi)] = np.nan
            outdict['pPhi_'+"%03dC_"%(c*100)+str(l)+"NM_I"+Itag] = pphi
            dMag = deltaMag(1, Rp*u.R_jupiter, d*u.AU, pphi)
            dMag[np.isinf(dMag)] = np.nan
            outdict['dMag_'+"%03dC_"%(c*100)+str(l)+"NM_I"+Itag] = dMag


        out = pd.DataFrame(outdict)

        if altorbdata is None:
            altorbdata = out.copy()
        else:
            altorbdata = altorbdata.append(out)

    return altorbdata

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

    # wfirstcontr = np.genfromtxt(contrfile)
    # contr = wfirstcontr[:,1]
    # angsep = wfirstcontr[:,0] #l/D
    # angsep = (angsep * (575.0*u.nm)/(2.37*u.m)*u.rad).decompose().to(u.mas).value #mas
    wfirstcontr = np.genfromtxt(contrfile, delimiter=',', skip_header=1)
    contr = wfirstcontr[:,3]
    angsep = wfirstcontr[:,0]
    angsep = (angsep * (575.0*u.nm)/(2.37*u.m)*u.rad).decompose().to(u.mas).value #mas
    wfirstc = interp1d(angsep,contr,bounds_error = False, fill_value = 'extrapolate')

    inds = np.where((data['pl_maxangsep'].values > minangsep) &
                    (data['pl_minangsep'].values < maxangsep))[0]

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
        if np.isnan(Imu) or ((row['pl_orbincl'] == 90) and (row['pl_orbinclerr1'] == 0) and (row['pl_orbinclerr2'] == 0)): #Generates full sinusoidal distribution if 90 incl 0 errors
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
        fe = row['st_met']
        if np.isnan(fe): fe = 0.0

        #initialize loops vars
        n = int(1e6) # Number of planets
        c = 0. # Completeness of the current loop
        h = np.zeros((len(WAbins)-3, len(dMagbins)-2))
        k = 0.0 # A counter
        cprev = 0.0 # Previous completeness value
        pdiff = 1.0 # Percent difference

        while (pdiff > 0.0001) | (k <3):
            print(f"Iteration:{k} \t Percent difference between previous completeness value:{pdiff:5.5e} \t Calculated completeness:{c:5.5e}")

            #sample orbital parameters
            a = gena(n)
            e = gene(n)
            I = genI(n)
            w = genw(n)

            #sample cloud vals
            cl = vget_fsed(np.random.rand(n))

            forecaster_mod = ForecasterMod()
            # Rewriting the calculation to not rely on potentially calculated values of mass
            if not np.isnan(row.pl_radj):
                # If we have the radius we can just use it
                Rmu = row.pl_radj
                Rstd = (row['pl_radjerr1'] - row['pl_radjerr2'])/2.
                if np.isnan(Rstd): Rstd = Rmu*0.1
                R = np.random.randn(n)*Rstd + Rmu
            else:
                # If we don't have radius then we need to calculate it based on the mass
                if row.pl_bmassprov == 'Msini':
                    # Sometimes given in Msini which we use the generated inclination values to get the Mp value from
                    Mp = ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value
                    Mp = Mp/np.sin(I)
                else:
                    # Standard calculation
                    Mstd = (((row['pl_bmassjerr1'] - row['pl_bmassjerr2'])*u.M_jupiter).to(u.M_earth)).value
                    if np.isnan(Mstd):
                        Mstd = ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value * 0.1
                    Mp = np.random.randn(n)*Mstd + ((row['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value

                # Now use the Mp value to get the planet radius
                R = forecaster_mod.calc_radius_from_mass(Mp*u.M_earth).to(u.R_jupiter).value
                # R = (ForecasterMod.calc_radius_from_mass(Mp)*u.R_earth).to(u.R_jupiter).value
                R[R > 1.0] = 1.0

            M0 = np.random.uniform(size=n,low=0.0,high=2*np.pi)
            E = eccanom(M0, e)
            nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

            d = a * (1.0 - e ** 2.0) / (1 + e * np.cos(nu))
            s = d * np.sqrt(4.0 * np.cos(2 * I) + 4 * np.cos(2 * nu + 2.0 * w) - 2.0 * np.cos(-2 * I + 2.0 * nu + 2 * w) - 2 * np.cos(2 * I + 2 * nu + 2 * w) + 12.0) / 4.0
            beta = np.arccos(-np.sin(I) * np.sin(nu + w)) * u.rad
            rnorm = d

            lum = row['st_lum']

            if np.isnan(lum):
                lum_fix = 1
            else:
                lum_fix = (10 ** lum) ** .5  # Since lum is log base 10 of solar luminosity

            pphi = np.zeros(n)
            for clevel in np.unique(cl):
                tmpinds = cl == clevel
                betatmp = beta[tmpinds]
                binds = np.argsort(betatmp)
                pphi[tmpinds] = (photinterps2[float(feinterp(fe))][float(distinterp(np.mean(rnorm) / lum_fix))][clevel](betatmp.to(u.deg).value[binds],ws).sum(1)*wstep/bw)[np.argsort(binds)].flatten()

            pphi[np.isinf(pphi)] = np.nan
            pphi[pphi <= 0.0] = 1e-16

            dMag = deltaMag(1, R*u.R_jupiter, rnorm*u.AU, pphi)
            WA = np.arctan((s*u.AU)/(row['sy_dist']*u.pc)).to('mas').value # working angle

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

    out2 = pd.DataFrame({'Name': np.hstack(names),
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
    data['completeness'] = tmp
    # data = data.assign(completeness=tmp)

    tmp = np.full(len(data),np.nan)
    tmp[goodinds] = minCWA
    data['compMinWA'] = tmp
    # data = data.assign(compMinWA=tmp)

    tmp = np.full(len(data),np.nan)
    tmp[goodinds] = maxCWA
    data['compMaxWA'] = tmp
    # data = data.assign(compMaxWA=tmp)

    tmp = np.full(len(data),np.nan)
    tmp[goodinds] = minCdMag
    data['compMindMag'] = tmp
    # data = data.assign(compMindMag=tmp)

    tmp = np.full(len(data),np.nan)
    tmp[goodinds] = maxCdMag
    data['compMaxdMag'] = tmp
    # data = data.assign(compMaxdMag=tmp)

    return out2,outdict,data


def genAliases(data):
    """ Grab all available aliases for list of targets. """

    s = Simbad()
    s.add_votable_fields('ids')
    # baseurl = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    # baseurl = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    baseurl = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=aliastable&objname="
    # https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html

    ids = []
    aliases = []
    noipacalias = []
    nosimbadalias = []
    badstars = []
    priname = []
    starnames = data.pl_name
    max_justification = data.pl_name.str.len().max()
    t_bar = trange(len(starnames), leave=False)
    print('Getting star aliases')
    for j,star in enumerate(starnames):
        t_bar.set_description(star.ljust(max_justification))
        t_bar.update(1)

        #get aliases from IPAC
        # r = requests.get(f"{baseurl}select+*+from+object_aliases+where+resolved_name='{star}'")
        r = requests.get(f"{baseurl}{star}")

        if 'ERROR' in str(r.content):
            noipacalias.append(star)
            tmp = [star]
        else:
            r_content = pd.read_csv(BytesIO(r.content))
            tmp = r_content.aliasdis.tolist()

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


    out3 = pd.DataFrame({'SID': np.hstack(ids),
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
        coldefs = pd.ExcelFile('coldefs.xlsx')
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

