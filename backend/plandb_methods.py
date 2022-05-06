from __future__ import division, print_function
import math
import os
import re
from pathlib import Path
from io import BytesIO

import astropy.constants as const
import astropy.units as u
import EXOSIMS.PlanetPhysicalModel.Forecaster
import json
import numpy as np
import pandas as pd
import pickle
import requests
import sqlalchemy.types
from astropy.time import Time
from astroquery.simbad import Simbad
from EXOSIMS.OpticalSystem.Nemati_2019 import Nemati_2019
from EXOSIMS.PlanetPhysicalModel.ForecasterMod import ForecasterMod
from EXOSIMS.TargetList.KnownRVPlanetsTargetList import KnownRVPlanetsTargetList
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.getExoplanetArchive import (getExoplanetArchivePS,
                                              getExoplanetArchivePSCP)
from MeanStars import MeanStars
from requests.exceptions import ConnectionError
from scipy.interpolate import RectBivariateSpline, griddata, interp1d, interp2d
from scipy.stats import gaussian_kde
from sqlalchemy import create_engine
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from kep_generator import planet

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO

def comp_plot():
    pass

def getIPACdata():
    '''
    grab everything from exoplanet and composite tables, merge,
    add additional column info and return/save to disk
    '''

    print("Querying IPAC for all data.")
    pscp_data = getExoplanetArchivePSCP() # Composite table
    ps_data = getExoplanetArchivePS()

    #only keep stuff related to the star from composite
    composite_cols = [col for col in pscp_data.columns if (col == 'pl_name') or str.startswith(col, 'st_') or col == 'sy_vmag']
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
        t_bar.update()
        pl = row.pl_name

        if pl != last_pl:
            # Get the composite row
            c_row = pscp_data.loc[pscp_data.pl_name == row.pl_name]

        # Now take all the stellar data from the composite table
        for col in composite_cols:
            if col != 'pl_name':
                if pd.isnull(merged_data.loc[i, col]):
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
        t_bar.update()

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
    t_bar = trange(len(pscp_data), leave=False)
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
    has_lum = ~np.isnan(merged_data['st_lum'].values)
    merged_data.loc[has_lum, 'st_lum_correction'] = (10 ** merged_data.loc[has_lum, 'st_lum']) ** .5  # Since lum is log base 10 of solar luminosity


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
    # merged_data.loc[3856, 'pl_bmassjerr2'] = -.18
    # merged_data.loc[3935, 'pl_orbsmaxerr2'] = -0.6
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


def calcQuadratureVals(orbitfits, bandzip, photdict):
    """
    Calculate quadrature photometry values for planets in data (output of getIPACdata)
    for bands in bandzip (output of genBands) assuming photometry info from photdict
    (output from loadPhotometryData)

    """

    smas = orbitfits['pl_orbsmax'].values
    fes = orbitfits['st_met'].values
    fes[np.isnan(fes)] = 0.0
    Rps = orbitfits['pl_radj_forecastermod'].values
    inc = orbitfits['pl_orbincl'].values * np.pi / 180.0
    eccen = orbitfits['pl_orbeccen'].values
    arg_per = orbitfits['pl_orblper'].values * np.pi / 180.0
    lum = orbitfits['st_lum'].values

    tmpout = {}

    quadinterps = photdict['quadinterps']
    feinterp = photdict['feinterp']
    distinterp = photdict['distinterp']
    lambdas = []

    #iterate over all data rows
    t_bar = trange(len(Rps), leave=False)
    for j, (Rp, fe,a, I, e, w, lm) in enumerate(zip(Rps, fes,smas, inc, eccen, arg_per, lum)):
        t_bar.update()
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

    orbitfits = orbitfits.join(pd.DataFrame(tmpout))

    return orbitfits


def genOrbitData(data, bandzip, photdict, t0=None):
    if t0 is None:
        t0 = Time('2026-01-01T00:00:00', format='isot', scale='utc')

    plannames = data['pl_name'].values
    minWA = data['pl_minangsep'].values * u.mas
    maxWA = data['pl_maxangsep'].values * u.mas
    plan_ids = data.index.tolist()

    data['st_lum_ms'] = pd.Series(np.zeros(len(plannames)), index=data.index)

    photinterps2 = photdict['photinterps']
    feinterp = photdict['feinterp']
    distinterp = photdict['distinterp']
    lambdas = [l for l, band, bw, ws, wstep in bandzip]

    orbdata = None
    orbitfits = None
    orbitfit_id = 0
    for j in range(len(plannames)):
        row = data.iloc[j]

        # orbital parameters
        a = row['pl_orbsmax']
        e = row['pl_orbeccen']
        if np.isnan(e): e = 0.0
        w = row['pl_orblper'] * np.pi / 180.0
        if np.isnan(w): w = 0.0
        Rp = row['pl_radj_forecastermod']
        dist = row['sy_dist']
        fe = row['st_met']
        if np.isnan(fe): fe = 0.0

        # time
        tau = row['pl_orbtper']  # jd
        Tp = row['pl_orbper']  # days
        if Tp == 0:
            Tp = np.nan
        if np.isnan(Tp):
            Mstar = row['st_mass']  # solar masses
            if not np.isnan(Mstar):
                if not np.isnan(row['pl_bmassj']):
                    mu = const.G * ((Mstar * u.solMass).decompose() + (row['pl_bmassj'] * u.jupiterMass).decompose())
                else:
                    mu = const.G * (Mstar * u.solMass).decompose()
                Tp = (2 * np.pi * np.sqrt(((a * u.AU) ** 3.0) / mu)).decompose().to(u.d).value

        t = np.zeros(100) * np.nan
        M = np.linspace(0, 2 * np.pi, 100)

        if not np.isnan(Tp):
            if np.isnan(tau):
                tau_temp = 0.0
            else:
                tau_temp = tau
            n = 2 * np.pi / Tp
            num_steps = Tp / 30
            t = np.arange(t0.jd, t0.jd + Tp, 30)
            if num_steps < 100:  # At least 100 steps
                t = np.linspace(t0.jd, t0.jd + Tp, 100)
            if Tp*u.day > 100*u.yr:
                t = np.arange(t0.jd, t0.jd + (100*u.yr).to(u.day).value, 30)

            M = np.mod((t - tau_temp) * n, 2 * np.pi)

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
            tper_next = np.nan
            tper_2026 = np.nan

        I = row['pl_orbincl'] * np.pi / 180.0
        # Generate orbit fits based on if I is null
        if np.isnan(I) and np.isnan(row['pl_orbinclerr1']) and np.isnan(row['pl_orbinclerr2']):
            I = np.pi / 2.0
            if row['pl_bmassprov'] == 'Msini':
                Icrit = np.arcsin(
                    ((row['pl_bmassj'] * u.M_jupiter).to(u.M_earth)).value / ((0.0800 * u.M_sun).to(u.M_earth)).value)
            else:
                Icrit = 10 * np.pi / 180.0
            IList_temp = np.pi / 180 * np.array([90, 60, 30])
            IList = np.hstack((IList_temp, Icrit))
            defs = [1, 0, 0, 0]
            icrit_flag = [0, 0, 0, 1]
            err1 = [0, 0, 0, 0]
            err2 = err1
        elif np.isnan(row['pl_orbinclerr1']) or np.isnan(row['pl_orbinclerr2']):
            # If there are no error bars use just the inclination
            IList = np.array([I])
            defs = [1]
            icrit_flag = [0]
            err1 = [np.nan]
            err2 = err1
        else:
            IList = np.array([I, I+row['pl_orbinclerr1']*np.pi/180, I + row['pl_orbinclerr2']*np.pi/180])
            defs = [1, 0, 0]
            icrit_flag = [0, 0, 0]
            err1 = [row['pl_orbinclerr1'], row['pl_orbinclerr1'], row['pl_orbinclerr1']]
            err2 = [row['pl_orbinclerr2'], row['pl_orbinclerr2'], row['pl_orbinclerr2']]

        for k,i in enumerate(IList):
            I = IList[k]
            # calculate orbital values
            E = eccanom(M, e)
            nu = 2 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E / 2.0));
            d = a * (1.0 - e ** 2.0) / (1 + e * np.cos(nu))
            s = d * np.sqrt(
                4.0 * np.cos(2 * I) + 4 * np.cos(2 * nu + 2.0 * w) - 2.0 * np.cos(-2 * I + 2.0 * nu + 2 * w) - 2 * np.cos(
                    2 * I + 2 * nu + 2 * w) + 12.0) / 4.0
            beta = np.arccos(-np.sin(I) * np.sin(nu + w)) * u.rad
            WA = np.arctan((s * u.AU) / (dist * u.pc)).to('mas').value

            print(j, plannames[j], WA.min() - minWA[j].value, WA.max() - maxWA[j].value)

            # Adjusts planet distance for luminosity
            lum_fix = row['st_lum_correction']
            orbitfits_dict = {'pl_id': [j],
                              'pl_orbincl':  [IList[k] * 180 / np.pi],
                              'pl_orbinclerr1': err1[k],
                              'pl_orbinclerr2': err2[k],
                              'orbtper_next': [tper_next],
                              'orbtper_2026': [tper_2026],
                              'is_Icrit': [icrit_flag[k]],
                              'default_fit': [defs[k]]}

            # Build rest of orbitfits
            for col_name in data.columns:
                if col_name != 'pl_orbincl' and col_name != 'pl_orbinclerr1' and col_name != 'pl_orbinclerr2':
                    orbitfits_dict[col_name] = row[col_name]

            outdict = {'pl_id': [j] * len(M),
                       'pl_name': [plannames[j]] * len(M),
                       'orbitfit_id': [orbitfit_id] * len(M),
                       'M': M,
                       't': t - t0.jd,
                       'r': d,
                       's': s,
                       'WA': WA,
                       'beta': beta.to(u.deg).value,
                       'is_icrit': [icrit_flag[k]] * len(M),
                       'default_orb': [defs[k]] * len(M)}
            orbitfit_id = orbitfit_id + 1

            alldMags = np.zeros((len(photdict['clouds']), len(lambdas), len(beta)))
            allpphis = np.zeros((len(photdict['clouds']), len(lambdas), len(beta)))

            # Photometry calcs
            inds = np.argsort(beta)
            for count1, c in enumerate(photdict['clouds']):
                for count2, (l, band, bw, ws, wstep) in enumerate(bandzip):
                    pphi = (photinterps2[float(feinterp(fe))][float(distinterp(a / lum_fix))][c](
                        beta.to(u.deg).value[inds], ws).sum(1) * wstep / bw)[np.argsort(inds)]
                    # print(f'calced_pphi: [{count1}, {count2}] : {pphi}')
                    pphi[np.isinf(pphi)] = np.nan
                    outdict['pPhi_' + "%03dC_" % (c * 100) + str(l) + "NM"] = pphi
                    allpphis[count1, count2] = pphi
                    dMag = deltaMag(1, Rp * u.R_jupiter, d * u.AU, pphi)
                    # print(f'calced_dMag: [{count1}, {count2}] : {dMag}')
                    dMag[np.isinf(dMag)] = np.nan
                    outdict['dMag_' + "%03dC_" % (c * 100) + str(l) + "NM"] = dMag
                    alldMags[count1, count2] = dMag

            pphismin = np.nanmin(allpphis, axis=0)
            dMagsmin = np.nanmin(alldMags, axis=0)
            pphismax = np.nanmax(allpphis, axis=0)
            dMagsmax = np.nanmax(alldMags, axis=0)
            pphismed = np.nanmedian(allpphis, axis=0)
            dMagsmed = np.nanmedian(alldMags, axis=0)

            for count3, l in enumerate(lambdas):
                outdict["dMag_min_" + str(l) + "NM"] = dMagsmin[count3]
                outdict["dMag_max_" + str(l) + "NM"] = dMagsmax[count3]
                outdict["dMag_med_" + str(l) + "NM"] = dMagsmed[count3]
                outdict["pPhi_min_" + str(l) + "NM"] = pphismin[count3]
                outdict["pPhi_max_" + str(l) + "NM"] = pphismax[count3]
                outdict["pPhi_med_" + str(l) + "NM"] = pphismed[count3]

            out = pd.DataFrame(outdict)

            if orbdata is None:
                orbdata = out.copy()
            else:
                orbdata = orbdata.append(out)

            if orbitfits is None:
                orbitfits = pd.DataFrame(orbitfits_dict).copy()
            else:
                orbitfits = orbitfits.append(pd.DataFrame(orbitfits_dict))

    orbdata = orbdata.sort_values(by=['pl_id', 'orbitfit_id', 't', 'M']).reset_index(drop=True)
    orbitfits = orbitfits.reset_index(drop=True)

    return orbdata, orbitfits

def addEphemeris(data, orbfits, orbdata, bandzip, photdict):
    '''
    Adds the ephemeris information to the orbit fits and orbit data
    '''
    raw_text = np.genfromtxt(
        "cache/planets_updated_param.txt", dtype=str, delimiter="\n"
    )
    planets = []
    for line_num, line_text in enumerate(raw_text):
        if "******" in line_text:
            planet_text = raw_text[line_num : line_num + 9]
            planets.append(planet(planet_text))

    pnames = np.array([p.planet_id.lower() for p in planets])
    tmp = np.char.lower(data['pl_name'].values.astype(str))
    inds = np.array([t in pnames for t in tmp])
    data = data.iloc[inds].reset_index()

    tmp = np.char.lower(data['pl_name'].values.astype(str))
    for p in planets:
        ind = tmp == p.planet_id.lower()
        data.loc[ind,'pl_orbsmax'] = p.a.to(u.AU).value
        data.loc[ind,'pl_orbsmaxerr1'] = p.a_err[0].to(u.AU).value
        data.loc[ind,'pl_orbsmaxerr2'] = -p.a_err[1].to(u.AU).value
        data.loc[ind,'pl_msinij'] = p.Msini.to(u.jupiterMass).value
        data.loc[ind,'pl_msinijerr1'] = p.Msini_err[0].to(u.jupiterMass).value
        data.loc[ind,'pl_msinijerr2'] = -p.Msini_err[0].to(u.jupiterMass).value
        data.loc[ind,'pl_bmassj'] = p.Msini.to(u.jupiterMass).value
        data.loc[ind,'pl_bmassjerr1'] = p.Msini_err[0].to(u.jupiterMass).value
        data.loc[ind,'pl_bmassjerr2'] = -p.Msini_err[0].to(u.jupiterMass).value
        data['pl_bmassprov'] = 'Msini'
        data.loc[ind,'pl_orbper'] = p.T.to(u.d).value
        data.loc[ind,'pl_orbpererr1'] = p.T_err[0].to(u.d).value
        data.loc[ind,'pl_orbpererr2'] = -p.T_err[1].to(u.d).value
        data.loc[ind,'pl_orbtper'] = p.T_p.value
        data.loc[ind,'pl_orbtpererr1'] = p.T_p_err[0].to(u.d).value
        data.loc[ind,'pl_orbtpererr2'] = -p.T_p_err[1].to(u.d).value
        data.loc[ind,'pl_orbeccen'] = p.e
        data.loc[ind,'pl_orbeccenerr1'] = p.e_err[0]
        data.loc[ind,'pl_orbeccenerr2'] = -p.e_err[1]
        data.loc[ind,'pl_orblper'] = p.w.to(u.deg).value
        data.loc[ind,'pl_orblpererr1'] = p.w_err[0].to(u.deg).value
        data.loc[ind,'pl_orblpererr2'] = -p.w_err[1].to(u.deg).value
    has_lum = ~np.isnan(data['st_lum'].values)
    data.loc[has_lum, 'st_lum_correction'] = (10 ** data.loc[has_lum, 'st_lum']) ** .5  # Since lum is log base 10 of solar luminosity


    #fill in missing smas from period & star mass
    nosma = np.isnan(data['pl_orbsmax'].values)
    p2sma = lambda mu,T: ((mu*T**2/(4*np.pi**2))**(1/3.)).to('AU')
    GMs = const.G*(data['st_mass'][nosma].values*u.solMass) # units of solar mass
    T = data['pl_orbper'][nosma].values*u.day
    tmpsma = p2sma(GMs,T)
    data.loc[nosma,'pl_orbsmax'] = tmpsma
    data['pl_calc_sma'] = pd.Series(np.zeros(len(data['pl_name'])), index=data.index)
    data.loc[nosma, 'pl_calc_sma'] = 1

    #propagate filled in sma errors
    GMerrs = ((data['st_masserr1'][nosma] - data['st_masserr2'][nosma])/2.).values*u.solMass*const.G
    Terrs = ((data['pl_orbpererr1'][nosma] - data['pl_orbpererr2'][nosma])/2.).values*u.day

    smaerrs = np.sqrt((2.0*T**2.0*GMs)**(2.0/3.0)/(9*np.pi**(4.0/3.0)*T**2.0)*Terrs**2.0 +\
            (2.0*T**2.0*GMs)**(2.0/3.0)/(36*np.pi**(4.0/3.0)*GMs**2.0)*GMerrs**2.0).to('AU')
    data.loc[nosma,'pl_orbsmaxerr1'] = smaerrs
    data.loc[nosma,'pl_orbsmaxerr2'] = -smaerrs
    #update all WAs (and errors) based on sma
    WA = np.arctan((data['pl_orbsmax'].values*u.AU)/(data['sy_dist'].values*u.pc)).to('mas')
    data['pl_angsep'] = WA.value
    sigma_a = ((data['pl_orbsmaxerr1']- data['pl_orbsmaxerr2'])/2.).values*u.AU
    sigma_d = ((data['sy_disterr1']- data['sy_disterr2'])/2.).values*u.pc
    sigma_wa = (np.sqrt(( (data['pl_orbsmax'].values*u.AU)**2.0*sigma_d**2 + (data['sy_dist'].values*u.pc)**2.0*sigma_a**2)/\
            ((data['pl_orbsmax'].values*u.AU)**2.0 + (data['sy_dist'].values*u.pc)**2.0)**2.0).decompose()*u.rad).to(u.mas)
    data['pl_angseperr1'] = sigma_wa.value
    data['pl_angseperr2'] = -sigma_wa.value
    #propagate filled in sma errors
    GMerrs = ((data['st_masserr1'][nosma] - data['st_masserr2'][nosma])/2.).values*u.solMass*const.G
    Terrs = ((data['pl_orbpererr1'][nosma] - data['pl_orbpererr2'][nosma])/2.).values*u.day

    smaerrs = np.sqrt((2.0*T**2.0*GMs)**(2.0/3.0)/(9*np.pi**(4.0/3.0)*T**2.0)*Terrs**2.0 +\
            (2.0*T**2.0*GMs)**(2.0/3.0)/(36*np.pi**(4.0/3.0)*GMs**2.0)*GMerrs**2.0).to('AU')
    data.loc[nosma,'pl_orbsmaxerr1'] = smaerrs
    data.loc[nosma,'pl_orbsmaxerr2'] = -smaerrs


    #update all WAs (and errors) based on sma
    WA = np.arctan((data['pl_orbsmax'].values*u.AU)/(data['sy_dist'].values*u.pc)).to('mas')
    data['pl_angsep'] = WA.value
    sigma_a = ((data['pl_orbsmaxerr1']- data['pl_orbsmaxerr2'])/2.).values*u.AU
    sigma_d = ((data['sy_disterr1']- data['sy_disterr2'])/2.).values*u.pc
    sigma_wa = (np.sqrt(( (data['pl_orbsmax'].values*u.AU)**2.0*sigma_d**2 + (data['sy_dist'].values*u.pc)**2.0*sigma_a**2)/\
            ((data['pl_orbsmax'].values*u.AU)**2.0 + (data['sy_dist'].values*u.pc)**2.0)**2.0).decompose()*u.rad).to(u.mas)
    data['pl_angseperr1'] = sigma_wa.value
    data['pl_angseperr2'] = -sigma_wa.value

    #fill in radius based on mass
    noR = (data['pl_radj'].isnull()).values

    # Initialize the ForecasterMod
    forecaster_mod = ForecasterMod()
    m = ((data['pl_bmassj'].values*u.M_jupiter).to(u.M_earth)).value
    merr = (((data['pl_bmassjerr1'].values - data['pl_bmassjerr2'].values)/2.0)*u.M_jupiter).to(u.M_earth).value
    R = forecaster_mod.calc_radius_from_mass(m*u.M_earth)

    # Turning what was a list comprehension into a loop to handle edge case
    Rerr = np.zeros(len(m))
    for j, m_val in enumerate(m):
        if np.isnan(merr[j]):
            Rerr[j] = np.nan
        elif merr[j] == 0:
            # Happens with truncation error sometimes, check if the error in earth radii is input to IPAC correctly
            merr_earth = (((data.iloc[j]['pl_bmasseerr1'] - data.iloc[j]['pl_bmasseerr2'])/2.0)*u.M_earth).to(u.M_earth).value
            Rerr[j] = forecaster_mod.calc_radius_from_mass(u.M_earth*np.random.normal(loc=m_val, scale=merr_earth, size=int(1e4))).std().value
        else:
            Rerr[j] = forecaster_mod.calc_radius_from_mass(u.M_earth*np.random.normal(loc=m_val, scale=merr[j], size=int(1e4))).std().value
    Rerr = Rerr*u.R_earth

    data['pl_radj_forecastermod'] = data['pl_radj'].values
    data.loc[noR,'pl_radj_forecastermod'] = (R.to(u.R_jupiter)).value

    data['pl_radj_forecastermoderr1'] = data['pl_radjerr1'].values
    data.loc[noR,'pl_radj_forecastermoderr1'] = (Rerr.to(u.R_jupiter)).value

    data['pl_radj_forecastermoderr2'] = data['pl_radjerr2'].values
    data.loc[noR,'pl_radj_forecastermoderr2'] = -(Rerr.to(u.R_jupiter)).value



    # now the Fortney model
    from EXOSIMS.PlanetPhysicalModel.FortneyMarleyCahoyMix1 import \
        FortneyMarleyCahoyMix1
    fortney = FortneyMarleyCahoyMix1()

    ml10 = m <= 17
    Rf = np.zeros(m.shape)
    Rf[ml10] = fortney.R_ri(0.67,m[ml10])

    mg10 = m > 17
    tmpsmas = data['pl_orbsmax'].values
    tmpsmas = tmpsmas[mg10]
    tmpsmas[tmpsmas < fortney.giant_pts2[:,1].min()] = fortney.giant_pts2[:,1].min()
    tmpsmas[tmpsmas > fortney.giant_pts2[:,1].max()] = fortney.giant_pts2[:,1].max()

    tmpmass = m[mg10]
    tmpmass[tmpmass > fortney.giant_pts2[:,2].max()] = fortney.giant_pts2[:,2].max()

    Rf[mg10] = griddata(fortney.giant_pts2, fortney.giant_vals2,( np.array([10.]*np.where(mg10)[0].size), tmpsmas, tmpmass))

    data['pl_radj_fortney'] = data['pl_radj'].values
    data.loc[noR,'pl_radj_fortney'] = ((Rf*u.R_earth).to(u.R_jupiter)).value

    Rf_err = []
    tmpsmas = data['pl_orbsmax'][noR].values
    tmpsmaserr = (data['pl_orbsmaxerr1'][noR].values - data['pl_orbsmaxerr2'][noR].values) / 2.0
    adist = np.zeros((len(m), int(1e4)))
    for j, _ in enumerate(tmpsmas):  # Create smax distribution
        if np.isnan(tmpsmaserr[j]) or (tmpsmaserr[j] == 0):
            adist[j, :] = (np.ones(int(1e4))* tmpsmas[j])
        else:
            adist[j, :] = (np.random.normal(loc=tmpsmas[j], scale=tmpsmaserr[j], size=int(1e4)))

    for j, _ in enumerate(m):  # Create m distribution and calculate errors
        if np.isnan(merr[j]) or (merr[j] == 0):
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

    # data = data.assign(pl_radj_fortneyerr1=data['pl_radjerr1'].values)
    data['pl_radj_fortneyerr1'] = data['pl_radjerr1'].values
    data.loc[noR, 'pl_radj_fortneyerr1'] = ((Rf_err * u.R_earth).to(u.R_jupiter)).value
    # data = data.assign(pl_radj_forecastermoderr2=data['pl_radjerr2'].values)
    data['pl_radj_fortneyerr2'] = data['pl_radjerr2'].values
    data.loc[noR, 'pl_radj_fortneyerr2'] = -((Rf_err * u.R_earth).to(u.R_jupiter)).value


    #populate max WA based on available eccentricity data (otherwise maxWA = WA)
    hase = ~np.isnan(data['pl_orbeccen'].values)
    maxWA = WA[:]
    maxWA[hase] = np.arctan((data['pl_orbsmax'][hase].values*(1 + data['pl_orbeccen'][hase].values)*u.AU)/(data['sy_dist'][hase].values*u.pc)).to('mas')
    # data = data.assign(pl_maxangsep=maxWA.value)
    data['pl_maxangsep'] = maxWA.value

    #populate min WA based on eccentricity & inclination data (otherwise minWA = WA)
    hasI =  ~np.isnan(data['pl_orbincl'].values)
    s = data['pl_orbsmax'].values*u.AU
    s[hase] *= (1 - data['pl_orbeccen'][hase].values)
    s[hasI] *= np.cos(data['pl_orbincl'][hasI].values*u.deg)
    s[~hasI] = 0
    minWA = np.arctan(s/(data['sy_dist'].values*u.pc)).to('mas')
    # data = data.assign(pl_minangsep=minWA.value)
    data['pl_minangsep'] =minWA.value

    #Fill in missing luminosity from meanstars
    ms = MeanStars()
    nolum_teff = np.isnan(data['st_lum'].values) & ~np.isnan(data['st_teff'].values)
    teffs = data.loc[nolum_teff, 'st_teff']
    lums_1 = ms.TeffOther('logL', teffs) # Calculates Luminosity when teff exists
    data.loc[nolum_teff, 'st_lum'] = lums_1

    nolum_noteff_spect = np.isnan(data['st_lum'].values) & ~data['st_spectype'].isnull().values
    spects = data.loc[nolum_noteff_spect, 'st_spectype']
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
    data.loc[nolum_noteff_spect, 'st_lum'] = lums2
    new_orbdata, new_orbfits = genOrbitData(data, bandzip, photdict)
    new_orbfits = new_orbfits.drop('index', axis=1)
    new_orbfits['default_fit'] = np.zeros(len(new_orbfits))

    # Getting data to make the dataframes combine better
    pl_names, indices = np.unique(new_orbdata.pl_name, return_index=True)
    new_orbdata['orbitfit_id'] += np.max(orbdata.orbitfit_id)+1
    old_orbfits = orbfits.reset_index(drop=True)
    relevant_orbfits = old_orbfits[old_orbfits['pl_name'].isin(pl_names)]
    old_orbfits_id_dict = relevant_orbfits.groupby('pl_name').first()['pl_id'].to_dict()
    new_orbfits_id_dict = new_orbfits.groupby('pl_name').first()['pl_id'].to_dict()

    # Getting a dictionary to map the planet id's to be correct
    replacement_dict = {new_pl_id: old_pl_id for (old_pl_name, old_pl_id) in old_orbfits_id_dict.items() for (new_pl_name, new_pl_id) in new_orbfits_id_dict.items() if new_pl_name==old_pl_name}
    new_orbfits.pl_id = new_orbfits.pl_id.replace(replacement_dict)
    new_orbdata.pl_id = new_orbdata.pl_id.replace(replacement_dict)

    # Combine the orbdata dataframes
    new_orbdata['from_IPAC'] = np.zeros(len(new_orbdata))
    orbdata['from_IPAC'] = np.ones(len(orbdata))
    orbdata = pd.concat([orbdata, new_orbdata])

    # Combine the orbfits dataframes
    new_orbfits['from_IPAC'] = np.zeros(len(new_orbfits))
    orbfits['from_IPAC'] = np.ones(len(orbfits))
    keys_to_take = [key for key in orbfits.keys() if key in new_orbfits.keys()]
    new_orbfits = new_orbfits[keys_to_take]
    orbfits = pd.concat([orbfits, new_orbfits])

    orbdata = orbdata.sort_values(by=['pl_id', 'orbitfit_id', 't', 'M']).reset_index(drop=True)
    orbfits = orbfits.reset_index(drop=True)
    return orbfits, orbdata

# Separates data into pl_data and st_data
def generateTables(data, orbitfits):
    print('Splitting IPAC data into Star and Planet data')
    # Isolate the stellar columns from the planet columns
    st_cols = [col for col in data.columns if ('st_' in col) or ('gaia_' in col) or ('sy_' in col)]
    st_cols.extend(["dec", "decstr", "hd_name", "hip_name", "ra", "rastr", 'elat', 'elon'])

    pl_cols = [col for col in data.columns if ('pl_' in col)]
    pl_cols.extend(["hostname", "disc_year", "disc_refname", "discoverymethod", "disc_locale", "ima_flag",
               "disc_instrument", "disc_telescope", "disc_facility", "rv_flag"])
    # orbitfits_cols = np.setdiff1d(orbitfits.columns, st_cols)
    # orbitfits_cols = np.setdiff1d(orbitfits_cols, pl_cols)
    orbitfits_cols = np.genfromtxt('oft_cols.csv', delimiter=',', dtype=str)
    # orbitfit_extra_cols = ['hostname', 'pl_name', 'pl_id', 'pl_angsep', 'pl_radj_forecastermod', 'pl_bmassj', 'pl_orbsmax']
    # orbitfits_cols.extend(orbitfit_extra_cols)
    # orbitfits_exclude_prefix = ('pl_massj', 'pl_sinij', 'pl_masse', 'pl_msinie', 'pl_bmasse', 'pl_rade', 'pl_rads', 'pl_defrefname', 'disc_', 'sy_', 'st_')
    # orbitfits_exclude = [col for col in orbitfits_cols if col.startswith(orbitfits_exclude_prefix)]
    # orbitfits_cols = np.setdiff1d(orbitfits_cols, orbitfits_exclude)

    st_cols.append("hostname")
    st_data = data[st_cols]
    st_data = st_data.drop_duplicates(subset='hostname', keep='first')
    st_data.reset_index(drop=True)

    pl_data = data[pl_cols]
    orbitfits = orbitfits[orbitfits_cols]
    pl_data = pl_data.rename(columns={'hostname':'st_name'})
    st_data = st_data.rename(columns={'hostname':'st_name'})
    orbitfits = orbitfits.rename(columns={'hostname':'st_name'})

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

    colmap_st = {k: k[3:] if (k.startswith('st_') and (k != 'st_id' and k != 'st_name' and k!= 'st_radv')) else k for k in st_data.keys()}
    st_data = st_data.rename(columns=colmap_st)

    # colmap_fits = {k: k[3:] if (k.startswith('pl_') and (k != 'pl_id' and k != 'pl_name')) else k for k in orbitfits.keys()}
    # orbitfits = orbitfits.rename(columns=colmap_fits)
    # breakpoint()
    return pl_data, st_data, orbitfits


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

def calcContrastCurves(data, exosims_json):
    """
    This function will be used to calculate the contrast curves for every
    star in the IPAC database based on the provided raw contrast file
    Arguments:
        data (pandas DataFrame):
            Contains the best fit data from IPAC
        exosims_json (Path):
            Path to the exosims input json that is used to compute contrast curves for the different modes
    Output:
        data (pandas DataFrame):
            Same dataframe, but with an additional columns that contain the path to each contrast curve
    """

    # Go through the target list and change the Vmag to match IPAC's data
    std_form_re = re.compile('\\s\w$')
    stupid_koi_re = re.compile('\\.\d\d$')

    # This will store every value in the row in order
    dup_star_names = []
    dup_star_Vmags = []
    dup_star_spectypes = []
    dup_star_inds = []
    # These lists don't contain duplicates
    star_names = []
    star_Vmags = []
    star_spectypes = []
    star_dists = []

    # These are used to keep track of where each planet's star's contrast curves
    # are going to be saved
    datestr = Time.now().datetime.strftime("%Y_%m")
    contrast_curve_cache_base = Path(f'cache/cont_curvs_{datestr}/')
    contrast_curve_cache_base.mkdir(parents=True, exist_ok=True)
    star_base_path_list = []

    # Get the stellar information for each star and keep track of them to add to the dataframe
    for _, row in data.iterrows():
        pl_name = row.pl_name
        star_name = row.hostname
        # re_search = std_form_re.search(pl_name)
        # koi_search = stupid_koi_re.search(pl_name)
        # if re_search:
            # star_name = pl_name[:-2]
        # elif koi_search:
            # star_name = pl_name[:-3]
        # else:
            # raise ValueError(f"Unexpected planet name format: {pl_name}")
        star_base_path = Path(f'{contrast_curve_cache_base}/{star_name.replace(" ","_")}.p')
        star_base_path_list.append(star_base_path)
        dup_star_names.append(star_name)
        dup_star_Vmags.append(row.sy_vmag)
        dup_star_spectypes.append(row.st_spectype)
        if star_name in star_names:
            # This catches when a star has already been added due to another planet
            continue
        star_names.append(star_name)
        star_Vmags.append(row.sy_vmag)
        star_spectypes.append(row.st_spectype)
        star_dists.append(row.sy_dist*u.pc)

    dup_sInds = []
    # This finds all the sInds values for the planets, this is needed
    # because the targetlist only stores each star once, but IPAC has multiple
    # planets for the stars. This keeps the sInd for each planet's associated
    # star in order so that they can be assigned to the dataframe easily
    for star_name in dup_star_names:
        #TODO The star '2MASS J01225093-2439505' is somehow getting mixed up with '2MASS J12073346-3932539' in this step and the index seems to be off by one once we hit the contrast curve calculations
        dup_sInds.append(np.where(np.array(star_names) == star_name)[0][0])
        # if star_name == '2MASS J01225093-2439505':
            # breakpoint()
    # data['contr_curve_base_path'] = star_base_path_list
    # sInds = np.arange(len(star_names))

    # Need to create EXOSIMS TargetList, get the observation mode, get fZ, and fEZ, working angles
    with open(exosims_json, 'rb') as ff:
        specs = json.loads(ff.read())
    # TL = EXOSIMS.TargetList.KnownRVPlanetsTargetList.KnownRVPlanetsTargetList(**specs)
    TL = EXOSIMS.Prototypes.TargetList.TargetList(**specs)
    OS = TL.OpticalSystem
    TL.Name = star_names
    TL.Vmag = star_Vmags
    TL.Spec = star_spectypes
    TL.dist = star_dists
    # TL.MV = np.zeros(len(star_names))
    # breakpoint()
    # TL.fillPhotometryVals()
    modes = OS.observingModes

    img_int_times = [25*u.hr, 100*u.hr, 10000*u.hr]
    spe_int_times = [100*u.hr, 400*u.hr, 10000*u.hr]


    for mode in modes:
        mode_name = mode['instName']
        minangsep = mode['IWA']
        maxangsep = mode['OWA']
        # Only care about planets that can be within our geometric constraints
        inds = np.where((data['pl_maxangsep'].values > minangsep.to(u.mas).value) &
                        (data['pl_minangsep'].values < maxangsep.to(u.mas).value) &
                        (data.default_fit == 1) &
                        (~data['sy_vmag'].isnull().values))[0]
        lam_D = mode['lam'].to(u.m)/(OS.pupilDiam*u.mas.to(u.rad))
        # working_angles_lam_D = [3.1, 3.6, 4.2, 5.0, 6.0, 7.0, 7.9]
        # working_angles_as = (working_angles_lam_D*lam_D*u.mas).to(u.arcsec)
        working_angles_as = np.linspace(minangsep.to(u.arcsec).value, maxangsep.to(u.arcsec).value)*u.arcsec
        working_angles_lam_D = (working_angles_as.to(u.mas)/lam_D).value
        # Now need to convert the inds into the corrected ones accounting for duplicates
        viable_inds = []
        for ind in inds:
            viable_inds.append(dup_sInds[ind])

        # Create new F0dict based on these stars
        if 'Spec' in mode_name:
            int_times = spe_int_times
        elif 'Imager' in mode_name:
            int_times = img_int_times
        else:
            raise ValueError(f"Invalid mode name: {mode_name}")

        # Getting the IWA and OWA based on mode
        for int_time in int_times:
            scenario_name = f"{mode_name.replace(' ', '_')}_{int_time.to(u.hr).value:0.0f}hr"
            scenario_path_list = []
            star_F0s = []
            for spectype in star_spectypes:
                # F0 seems incapable of handling sub-dwarves so exclude them
                if isinstance(spectype, str) and not "VI" in spectype:
                    star_F0s.append(TL.F0(mode["BW"], mode["lam"], spec=spectype).value)
                else:
                    star_F0s.append(TL.F0(mode["BW"], mode["lam"]).value)
            TL.F0dict[mode['hex']] = star_F0s * (u.ph / (u.m**2 * u.s * u.nm))

            fZ0 = TL.ZodiacalLight.fZ0
            fEZ = TL.ZodiacalLight.fEZ(5.05, [20]*u.deg, 2.9*u.AU) # Taken from the Nemati spreadsheet
            # for star in tqdm(star_names):
                # contrasts = []
            contrasts = []

            # Create the cache location for the contrast curves
            # int_time = 100*u.hr
            print(f'Calculating contrast curves for {scenario_name}')
            # t_bar = trange(len(viable_inds), leave=False)
            for i, star_name in enumerate(tqdm(dup_star_names)):
                ind = dup_sInds[i]
                star_scenario_path = Path(f'{contrast_curve_cache_base}/{star_name.replace(" ","_")}_{scenario_name}.p')
                scenario_path_list.append(star_scenario_path)
                # if i == 10:
                    # breakpoint()
                if ind not in viable_inds:
                    continue
                # t_bar.update()
                if star_scenario_path.exists():
                    continue
                # if star_Vmags[i] > 15:
                    # continue
                contr_df = pd.DataFrame()
                contrasts = []
                dMags = []
                # test_dMags = np.linspace(0, 21, 1000)
                # test_intTimes = []
                # for test_dMag in test_dMags:
                    # test_intTimes.append(OS.calc_intTime(TL, [ind], fZ0, fEZ, test_dMag, working_angles_as[0], mode))
                # plt.scatter(test_dMags, test_intTimes)
                # plt.xlabel('dMag')
                # plt.ylabel('Integration time (days)')
                # # plt.yscale('log')
                # plt.savefig('dmagvit.png', dpi=200)
                # breakpoint()
                for working_angle in working_angles_as:
                    # dMag = OS.calc_dMag_per_intTime([int_time.to(u.day).value]*u.day, TL, [ind], fZ0, fEZ, working_angles_as, mode)
                    dMag = OS.calc_dMag_per_intTime([int_time.to(u.day).value]*u.day, TL, [ind], fZ0, fEZ, working_angle, mode)[0][0]
                    dMags.append(dMag)
                    contr = 10**(dMag/(-2.5))
                    contrasts.append(contr)
                contr_df['r_lamD'] = working_angles_lam_D
                contr_df['r_as'] = working_angles_as.value
                contr_df['r_mas'] = working_angles_as.value*1000
                contr_df['contrast'] = contrasts
                contr_df['dMag'] = dMags
                contr_df['lam'] = [mode['lam'].value]*len(working_angles_lam_D)
                contr_df['t_int_hr'] = [int_time.value]*len(working_angles_as)
                contr_df['fpp'] = [1/TL.PostProcessing.ppFact(working_angles_as[0])]*len(working_angles_as)
                contr_df.to_pickle(star_scenario_path)
            data[scenario_name] = scenario_path_list
    return data

def calcPlanetCompleteness(data, bandzip, photdict, exosims_json, minangsep=150,maxangsep=450, plotting=False):
    """ For all known planets in data (output from getIPACdata), calculate obscurational
    and photometric completeness for those cases where obscurational completeness is
    non-zero.
    """
    # Get the observing modes
    with open(exosims_json, 'rb') as ff:
        specs = json.loads(ff.read())
    # TL = EXOSIMS.TargetList.KnownRVPlanetsTargetList.KnownRVPlanetsTargetList(**specs)
    TL = EXOSIMS.Prototypes.TargetList.TargetList(**specs)
    OS = TL.OpticalSystem
    modes = OS.observingModes
    img_int_times = [25*u.hr, 100*u.hr, 10000*u.hr]
    spe_int_times = [100*u.hr, 400*u.hr, 10000*u.hr]


    # TODO remove mode override
    # modes = [modes[0]]
    # img_int_times = [25*u.hr]

    # Get scenario specific information
    (l,band,bw,ws,wstep) = bandzip[0]
    photinterps2 = photdict['photinterps']
    feinterp = photdict['feinterp']
    distinterp = photdict['distinterp']
    datestr = Time.now().datetime.strftime("%Y_%m")


    scenario_angles = pd.read_csv('cache/scenario_angles.csv')
    comp_scenarios = []
    compMinWA_scenarios = []
    compMaxWA_scenarios = []
    compMindMag_scenarios = []
    compMaxdMag_scenarios = []
    cols = []
    for scenario_name in scenario_angles.scenario_name:
        cols.append(f'completeness_{scenario_name}')
        cols.append(f'compMinWA_{scenario_name}')
        cols.append(f'compMaxWA_{scenario_name}')
        cols.append(f'compMindMag_{scenario_name}')
        cols.append(f'compMaxdMag_{scenario_name}')
        comp_scenarios.append(f'completeness_{scenario_name}')
        compMinWA_scenarios.append(f'compMinWA_{scenario_name}')
        compMaxWA_scenarios.append(f'compMaxWA_{scenario_name}')
        compMindMag_scenarios.append(f'compMindMag_{scenario_name}')
        compMaxdMag_scenarios.append(f'compMaxdMag_{scenario_name}')
    tmpdf = pd.DataFrame(data=np.full((len(data), len(cols)), np.nan), columns=cols)
    data = pd.concat([data, tmpdf], axis=1)

    # PDF information
    names = []
    WAcs = []
    dMagcs = []
    iinds = []
    jinds = []
    hs = []
    goodinds = []
    #define vectorized f_sed sampler
    vget_fsed = np.vectorize(get_fsed)
    # cs = []
    # inds = np.where((data['pl_maxangsep'].values > minangsep) &
                    # (data['pl_minangsep'].values < maxangsep) &
                    # (~data['sy_vmag'].isnull().values) &
                    # (data.default_fit == 1))[0]
    # Do completeness calculations for each scenario
    # TEMP_data = data.iloc[0:100]
    for ind, planet in tqdm(data.iterrows(), total=data.shape[0]):
        if planet.default_fit == 0:
            continue
        # tqdm.set_description(planet.pl_name)
        minangseps = []
        maxangseps = []
        scenario_names = []
        WAbins_list = []
        dMagbins_list = []
        WAc_list = []
        dMagc_list = []
        WAinds_list = []
        dMaginds_list = []

        for mode in modes:
            minangsep = mode['IWA'].to(u.mas).value
            maxangsep = mode['OWA'].to(u.mas).value
            mode_name = mode['instName']
            # Getting the list of the contrast and star names curves
            wfirstc_list = []
            # WAbins0 = np.arange(minangsep,maxangsep+1,1)
            # WAbins = np.hstack((0, WAbins0, np.inf))
            # dMagbins0 = np.arange(0,26.1,0.1)
            # dMagbins = np.hstack((dMagbins0,np.inf))

            # WAc,dMagc = np.meshgrid(WAbins0[:-1]+np.diff(WAbins0)/2.0,dMagbins0[:-1]+np.diff(dMagbins0)/2.0)
            # WAc = WAc.T
            # dMagc = dMagc.T

            # WAinds = np.arange(WAbins0.size-1)
            # dMaginds = np.arange(dMagbins0.size-1)
            # WAinds,dMaginds = np.meshgrid(WAinds,dMaginds)
            # WAinds = WAinds.T
            # dMaginds = dMaginds.T

            # WAbins_list.append(WAbins)
            # dMagbins_list.append(dMagbins)
            # WAc_list.append(WAc)
            # dMagc_list.append(dMagc)
            # WAinds_list.append(WAinds)
            # dMaginds_list.append(dMaginds)
            # Create new F0dict based on these stars
            if 'Spec' in mode_name:
                int_times = spe_int_times
            elif 'Imager' in mode_name:
                int_times = img_int_times
            else:
                raise ValueError(f"Invalid mode name: {mode_name}")
            for int_time in int_times:
                scenario_name = f"{mode_name.replace(' ', '_')}_{int_time.to(u.hr).value:0.0f}hr"
                scenario_names.append(scenario_name)
                minangseps.append(minangsep)
                maxangseps.append(maxangsep)
        WAbins0 = np.arange(0, 1700,10)
        WAbins = np.hstack((0, WAbins0, np.inf))
        dMagbins0 = np.arange(10,30,0.2)
        dMagbins = np.hstack((dMagbins0,np.inf))

        WAc,dMagc = np.meshgrid(WAbins0[:-1]+np.diff(WAbins0)/2.0,dMagbins0[:-1]+np.diff(dMagbins0)/2.0)
        WAc = WAc.T
        dMagc = dMagc.T

        WAinds = np.arange(WAbins0.size-1)
        dMaginds = np.arange(dMagbins0.size-1)
        WAinds,dMaginds = np.meshgrid(WAinds,dMaginds)
        WAinds = WAinds.T
        dMaginds = dMaginds.T

        planet_wfirstc_list = []
        for scenario in scenario_names:
            scenario_cc = planet[scenario]
            if scenario_cc.exists():
                with open(scenario_cc, 'rb') as f:
                    contr_df = pickle.load(f)
                scenario_wfirstc = interp1d(contr_df.r_mas, contr_df.contrast, bounds_error = False)
            else:
                # scenario_info = scenario_angles.loc[scenario_angles.scenario_name == scenario]
                # scenario_minangsep = u.Quantity(scenario_info.minangsep).to(u.mas)
                # scenario_maxangsep = u.Quantity(scenario_info.maxangsep).to(u.mas)
                scenario_wfirstc = None

            planet_wfirstc_list.append(scenario_wfirstc)
        if np.all(np.array(planet_wfirstc_list) == None):
            continue
        # planet_wfirstc = interp1d(contr_df.r_mas, contr_df.contrast, bounds_error = False)
        # planet_wfirstc = interp1d(contr_df.r_mas, contr_df.contrast, bounds_error = False, fill_value = 'extrapolate')
        # wfirstc_list.append(planet_wfirstc)
        # wfirstc_list = planet_wfirstc
        # data[f"contr_curve_{scenario_name}"] = wfirstc_list



        # Where the old loop started...
        # row = data.iloc[j]

        # wfirstc = row[f'contr_curve_{scenario_name}']

        # dMaglimsc = wfirstc(WAc[:, 0])
        # print("%d/%d  %s"%(i+1,len(inds),row['pl_name']))

        #sma distribution
        amu = planet['pl_orbsmax']
        astd = (planet['pl_orbsmaxerr1'] - planet['pl_orbsmaxerr2'])/2.
        if np.isnan(astd): astd = 0.01*amu
        gena = lambda n: np.clip(np.random.randn(n)*astd + amu,0,np.inf)

        #eccentricity distribution
        emu = planet['pl_orbeccen']
        if np.isnan(emu):
            gene = lambda n: 0.175/np.sqrt(np.pi/2.)*np.sqrt(-2.*np.log(1 - np.random.uniform(size=n)))
        else:
            estd = (planet['pl_orbeccenerr1'] - planet['pl_orbeccenerr2'])/2.
            if np.isnan(estd) or (estd == 0):
                estd = 0.01*emu
            gene = lambda n: np.clip(np.random.randn(n)*estd + emu,0,0.99)

        #inclination distribution
        Imu = planet['pl_orbincl']*np.pi/180.0
        if np.isnan(Imu) or ((planet['pl_orbincl'] == 90) and (planet['pl_orbinclerr1'] == 0) and (planet['pl_orbinclerr2'] == 0)): #Generates full sinusoidal distribution if 90 incl 0 errors
            if planet['pl_bmassprov'] == 'Msini':
                Icrit = np.arcsin( ((planet['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value/((0.0800*u.M_sun).to(u.M_earth)).value )
                Irange = [Icrit, np.pi - Icrit]
                C = 0.5*(np.cos(Irange[0])-np.cos(Irange[1]))
                genI = lambda n: np.arccos(np.cos(Irange[0]) - 2.*C*np.random.uniform(size=n))

            else:
                genI = lambda n: np.arccos(1 - 2.*np.random.uniform(size=n))
        else:
            Istd = (planet['pl_orbinclerr1'] - planet['pl_orbinclerr2'])/2.*np.pi/180.0
            if np.isnan(Istd) or (Istd == 0):
                Istd = Imu*0.01
            genI = lambda n: np.random.randn(n)*Istd + Imu

        #arg. of periastron distribution
        wmu = planet['pl_orblper']*np.pi/180.0
        if np.isnan(wmu):
            genw = lambda n: np.random.uniform(size=n,low=0.0,high=2*np.pi)
        else:
            wstd = (planet['pl_orblpererr1'] - planet['pl_orblpererr2'])/2.*np.pi/180.0
            if np.isnan(wstd) or (wstd == 0):
                wstd = wmu*0.01
            genw = lambda n: np.random.randn(n)*wstd + wmu

        #just a single metallicity
        fe = planet['st_met']
        if np.isnan(fe): fe = 0.0

        #initialize loops vars
        n = int(1e6) # Number of planets
        c = np.zeros(len(scenario_names)) # Completeness of the current loop
        h = np.zeros((len(WAbins)-3, len(dMagbins)-2))
        k = 0.0 # A counter
        cprev = np.zeros(len(scenario_names)) # Previous completeness value
        pdiff = np.ones(len(scenario_names)) # Percent difference

        currc_arr = np.zeros(len(scenario_names))
        while np.any(pdiff > 0.0001) | (k <3):
            # print(f"Iteration:{k}")
            # print(c)
            # print(f"Iteration:{k} \t Percent difference between previous completeness value:{max(pdiff):5.5e} \t Calculated completeness:{c:5.5e}")

            #sample orbital parameters
            a = gena(n)
            e = gene(n)
            I = genI(n)
            w = genw(n)

            #sample cloud vals
            cl = vget_fsed(np.random.rand(n))

            forecaster_mod = ForecasterMod()
            # Rewriting the calculation to not rely on potentially calculated values of mass
            if not np.isnan(planet.pl_radj):
                # If we have the radius we can just use it
                Rmu = planet.pl_radj
                Rstd = (planet['pl_radjerr1'] - planet['pl_radjerr2'])/2.
                if np.isnan(Rstd): Rstd = Rmu*0.1
                R = np.random.randn(n)*Rstd + Rmu
            else:
                # If we don't have radius then we need to calculate it based on the mass
                if planet.pl_bmassprov == 'Msini':
                    # Sometimes given in Msini which we use the generated inclination values to get the Mp value from
                    Mp = ((planet['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value
                    Mp = Mp/np.sin(I)
                else:
                    # Standard calculation
                    Mstd = (((planet['pl_bmassjerr1'] - planet['pl_bmassjerr2'])*u.M_jupiter).to(u.M_earth)).value
                    if np.isnan(Mstd):
                        Mstd = ((planet['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value * 0.1
                    Mp = np.random.randn(n)*Mstd + ((planet['pl_bmassj']*u.M_jupiter).to(u.M_earth)).value

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

            lum = planet['st_lum']

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
            WA = np.arctan((s*u.AU)/(planet['sy_dist']*u.pc)).to('mas').value # working angle

            h += np.histogram2d(WA,dMag,bins=(WAbins,dMagbins))[0][1:-1,0:-1]
            k += 1.0

            for scenario_n, wfirstc in enumerate(planet_wfirstc_list):
                if wfirstc == None:
                    currc = 0
                    currc_arr[scenario_n] = currc
                else:
                    dMaglimtmp = -2.5*np.log10(wfirstc(WA))
                    currc = float(len(np.where((WA >= minangseps[scenario_n]) & (WA <= maxangseps[scenario_n]) & (dMag <= dMaglimtmp))[0]))/n
                    currc_arr[scenario_n] = currc

            if k == 1 and plotting:
                # plt.style.use('dark_background')
                fig_path = Path('figures', f'{mode_name.replace(" ", "_")}_{str(int_time).replace(" ", "_")}')
                fig_path.mkdir(parents=True, exist_ok=True)
                save_path = Path(fig_path, f'{planet["pl_name"].replace(" ", "_")}.png')
                fig, ax = plt.subplots()
                ax.set_title(f'Mode: {mode_name.replace("EB", "Conservative").replace("DRM", "Optimistic").replace("_", " ")}. Int time: {int_time}. Planet: {planet["pl_name"]}')
                ax.set_xlabel(r'$\alpha$ (mas)')
                ax.set_ylabel(r'$\Delta$mag')
                ax.set_xlim([0, maxangsep+100])
                ax.set_ylim([10, 35])
                ax.annotate(f"C={currc:.4f}", xy=(250, 33), ha='center', va='center', zorder=6, size=20)

                IWA_ang = minangsep
                OWA_ang = maxangsep
                wa_range = np.linspace(minangsep, maxangsep)
                dmaglims = -2.5*np.log10(wfirstc(wa_range))
                # detectability_line = mpl.lines.Line2D([IWA_ang, OWA_ang], [dMaglimtmp[0], dMaglimtmp[0]], color='red')
                # ax.add_line(detectability_line)

                my_cmap = plt.get_cmap('viridis')
                WA_edges = np.linspace(0, 1600, 251)
                dMag_edges = np.linspace(10, 35, 251)
                H, WA_edges, dMag_edges = np.histogram2d(WA, dMag, range=[[WA_edges[0], WA_edges[-1]], [dMag_edges[0], dMag_edges[-1]]], bins=(500, 500))
                extent = [WA_edges[0], WA_edges[-1], dMag_edges[0], dMag_edges[-1]]
                levels = np.logspace(-6, -1, num=30)
                H_scaled = H.T/100000
                ax.contourf(H_scaled, levels=levels, cmap=my_cmap, origin='lower', extent=extent, norm=mpl.colors.LogNorm())
                ax.plot(wa_range, dmaglims, c='r')
                FR_norm = mpl.colors.LogNorm()
                sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=FR_norm)
                sm._A=[]
                sm.set_array(np.logspace(-6, -1))
                fig.subplots_adjust(left=0.15, right=0.85)
                cbar_ax = fig.add_axes([0.865, 0.125, 0.02, 0.75])
                cbar = fig.colorbar(sm, cax=cbar_ax, label=r'Normalized Density')
                ax.plot(wa_range, dmaglims, c='r')
                fig.savefig(save_path)
                plt.close()

            #currc = float(len(np.where((WA >= minangsep) & (WA <= maxangsep) & (dMag <= 22.5))[0]))/n
            cprev = c
            if k == 1.0:
                c = currc_arr
            else:
                c = ((k-1)*c + currc_arr)/k
            if np.all(c == 0):
                pdiff = np.ones(len(c))
            else:
                pdiff = np.abs(c - cprev)/c
                # print(f'{planet.pl_name} {k} pdiff: {pdiff}')

            if np.all(c == 0.0) & (k > 2):
                break

            if np.all(c < 1e-5) & (k > 15):
                break

            if k > 30:
                print(len(h.flatten()))
                break


        # if c != 0.0:
        h = h/float(n*k)
        names.append(np.array([planet['pl_name']]*h.size))
        WAcs.append(WAc.flatten())
        dMagcs.append(dMagc.flatten())
        hs.append(h.flatten())
        iinds.append(WAinds.flatten())
        jinds.append(dMaginds.flatten())
        # cs.append(c)
        # data.at]
        data.loc[ind, comp_scenarios] = currc_arr
        if np.any(currc_arr >0):
            print(planet.pl_name)
            print(currc_arr)
        # data.loc[0, compMinWA_scenarios] = currc_arr
        goodinds.append(ind)

        # print("\n\n\n\n")

        # cs = np.array(cs)
        # goodinds = np.array(goodinds)

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
        if np.all(hs[j] == 0):
            minCWA.append(np.nan)
            maxCWA.append(np.nan)
            minCdMag.append(np.nan)
            maxCdMag.append(np.nan)
        else:
            minCWA.append(np.floor(np.min(WAcs[j][hs[j] != 0])))
            maxCWA.append(np.ceil(np.max(WAcs[j][hs[j] != 0])))
            minCdMag.append(np.floor(np.min(dMagcs[j][hs[j] != 0])))
            maxCdMag.append(np.ceil(np.max(dMagcs[j][hs[j] != 0])))
    # breakpoint()
    outdict = {#'cs':cs,
               # 'goodinds':goodinds,
               'minCWA':minCWA,
               'maxCWA':maxCWA,
               'minCdMag':minCdMag,
               'maxCdMag':maxCdMag}

        # tmp = np.full(len(data),np.nan)
        # tmp[goodinds] = cs
        # data[f'completeness_{scenario_name}'] = tmp

        # tmp = np.full(len(data),np.nan)
        # tmp[goodinds] = minCWA
        # data[f'compMinWA_{scenario_name}'] = tmp

        # tmp = np.full(len(data),np.nan)
        # tmp[goodinds] = maxCWA
        # data[f'compMaxWA_{scenario_name}'] = tmp

        # tmp = np.full(len(data),np.nan)
        # tmp[goodinds] = minCdMag
        # data[f'compMindMag_{scenario_name}'] = tmp

        # tmp = np.full(len(data),np.nan)
        # tmp[goodinds] = maxCdMag
        # data[f'compMaxdMag_{scenario_name}'] = tmp

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
        t_bar.update()

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


def writeSQL_old(engine,data=None,orbdata=None,altorbdata=None,comps=None,aliases=None):
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

def writeSQL(engine, plandata=None, stdata=None, orbitfits=None, orbdata=None, pdfs=None, aliases=None,contrastCurves=None,scenarios=None, completeness=None):
    """write outputs to sql database via engine"""
    engine.execute("DROP TABLE IF EXISTS Completeness, ContrastCurves, Scenarios, PDFs, Orbits, OrbitFits, Planets, Stars")

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
        # addSQLcomments(engine, 'Stars')

    if plandata is not None:
        print("Writing Planets")
        namemxchar = np.array([len(n) for n in plandata['pl_name'].values]).max()
        plandata = plandata.rename_axis('pl_id')
        plandata.to_sql('Planets',engine,chunksize=100,if_exists='replace',
                    dtype={'pl_id':sqlalchemy.types.INT,
                            'pl_name':sqlalchemy.types.String(namemxchar),
                            'st_name':sqlalchemy.types.String(namemxchar-2),
                            'pl_letter':sqlalchemy.types.CHAR(1),
                            'st_id': sqlalchemy.types.INT})
        #set indexes
        result = engine.execute("ALTER TABLE Planets ADD INDEX (pl_id)")
        result = engine.execute("ALTER TABLE Planets ADD INDEX (st_id)")
        result = engine.execute("ALTER TABLE Planets ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

        #add comments
        # addSQLcomments(engine,'Planets')

    if orbitfits is not None:
        print("Writing OrbitFits")
        orbitfits = orbitfits.rename_axis('orbitfit_id')
        namemxchar = np.array([len(n) for n in orbitfits['pl_name'].values]).max()
        orbitfits.to_sql('OrbitFits',engine,chunksize=100,if_exists='replace',
                          dtype={'pl_id': sqlalchemy.types.INT,
                                 'orbitfit_id': sqlalchemy.types.INT,
                                 'pl_name': sqlalchemy.types.String(namemxchar)},
                          index=True)
        result = engine.execute("ALTER TABLE OrbitFits ADD INDEX (orbitfit_id)")
        result = engine.execute("ALTER TABLE OrbitFits ADD INDEX (pl_id)")
        result = engine.execute("ALTER TABLE OrbitFits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

        # addSQLcomments(engine,'OrbitFits')

    if orbdata is not None:
        print("Writing Orbits")
        namemxchar = np.array([len(n) for n in orbdata['pl_name'].values]).max()
        orbdata = orbdata.rename_axis('orbit_id')
        orbdata.to_sql('Orbits',engine,chunksize=100,if_exists='replace',
                       dtype={'pl_name':sqlalchemy.types.String(namemxchar),
                              'pl_id': sqlalchemy.types.INT,
                              'orbit_id': sqlalchemy.types.BIGINT,
                              'orbitfit_id': sqlalchemy.types.INT},
                       index=True)
        result = engine.execute("ALTER TABLE Orbits ADD INDEX (orbit_id)")
        result = engine.execute("ALTER TABLE Orbits ADD INDEX (pl_id)")
        result = engine.execute("ALTER TABLE Orbits ADD INDEX (orbitfit_id)")
        result = engine.execute("ALTER TABLE Orbits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION");
        result = engine.execute("ALTER TABLE Orbits ADD FOREIGN KEY (orbitfit_id) REFERENCES OrbitFits(orbitfit_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

        # addSQLcomments(engine,'Orbits')

    if pdfs is not None:
        print("Writing PDFs")
        pdfs = pdfs.reset_index(drop=True)
        namemxchar = np.array([len(n) for n in pdfs['Name'].values]).max()
        pdfs = pdfs.rename_axis('pdf_id')
        pdfs.to_sql('PDFs',engine,chunksize=100,if_exists='replace',
                     dtype={'pl_name':sqlalchemy.types.String(namemxchar),
                            'pl_id': sqlalchemy.types.INT})
        # result = engine.execute("ALTER TABLE PDFs ADD INDEX (orbitfit_id)")
        result = engine.execute("ALTER TABLE PDFs ADD INDEX (pl_id)")
        result = engine.execute("ALTER TABLE PDFs ADD INDEX (pdf_id)")
        # result = engine.execute("ALTER TABLE PDFs ADD FOREIGN KEY (orbitfit_id) REFERENCES OrbitFits(orbitfit_id) ON DELETE NO ACTION ON UPDATE NO ACTION")
        result = engine.execute("ALTER TABLE PDFs ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION")

        # addSQLcomments(engine,'PDFs')

    if aliases is not None:
        print("Writing Alias")
        aliases = aliases.rename_axis('alias_id')
        aliasmxchar = np.array([len(n) for n in aliases['Alias'].values]).max()
        aliases.to_sql('Aliases',engine,chunksize=100,if_exists='replace',dtype={'Alias':sqlalchemy.types.String(aliasmxchar)})
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (alias_id)")
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (Alias)")
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (st_id)")
        result = engine.execute("ALTER TABLE Aliases ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION")

    if scenarios is not None:
        print("Writing Scenarios")

        namemxchar = np.array([len(n) for n in scenarios['scenario_name'].values]).max()
        scenarios.to_sql("Scenarios", engine, chunksize=100, if_exists='replace', dtype={
            'scenario_name': sqlalchemy.types.String(namemxchar),}, index = False)

        result = engine.execute("ALTER TABLE Scenarios ADD INDEX (scenario_name)")

    if contrastCurves is not None:
        print("Writing ContrastCurves")
        contrastCurves = contrastCurves.rename_axis("curve_id")
        namemxchar = np.array([len(n) for n in contrastCurves['scenario_name'].values]).max()
        contrastCurves.to_sql("ContrastCurves", engine, chunksize=100, if_exists='replace', dtype={
            'st_id': sqlalchemy.types.INT,
            'curve_id' : sqlalchemy.types.INT,
            'scenario_name': sqlalchemy.types.String(namemxchar)}, index = True)
        # result = engine.execute("ALTER TABLE ContastCurves ADD INDEX (scenario_name)")

        # result = engine.execute("ALTER TABLE ContastCurves ADD INDEX (st_id)")

        result = engine.execute("ALTER TABLE ContrastCurves ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION")
        result = engine.execute("ALTER TABLE ContrastCurves ADD FOREIGN KEY (scenario_name) REFERENCES Scenarios(scenario_name) ON DELETE NO ACTION ON UPDATE NO ACTION")

    if completeness is not None:
        print("Writing completeness")
        completeness = completeness.rename_axis("completeness_id")
        namemxchar = np.array([len(n) for n in completeness['scenario_name'].values]).max()
        completeness.to_sql("Completeness", engine, chunksize=100, if_exists='replace', dtype={
            # 'scenario_id': sqlalchemy.types.INT,
            'pl_id' : sqlalchemy.types.INT,
            'completeness_id' : sqlalchemy.types.INT,
            'scenario_name': sqlalchemy.types.String(namemxchar)}, index = True)

        result = engine.execute("ALTER TABLE Completeness ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION")
        result = engine.execute("ALTER TABLE ContrastCurves ADD FOREIGN KEY (scenario_name) REFERENCES Scenarios(scenario_name) ON DELETE NO ACTION ON UPDATE NO ACTION")



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
