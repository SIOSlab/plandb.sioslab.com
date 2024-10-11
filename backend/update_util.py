import EXOSIMS.util
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from sqlalchemy import text
from plandb_methods import *
from database_main import *
import sweetviz as sv
from typing import Any, Dict, Tuple
from EXOSIMS.util.get_dirs import *
from EXOSIMS.util.getExoplanetArchive import *


# Util functions for update, also compacts much of the code from gen_plandb and database_main


def customGetExoplanetArchivePS(
    forceNew: bool = False, **specs: Dict[Any, Any]
) -> pd.DataFrame:
    """
    Get the contents of the Exoplanet Archive's Planetary Systems table and cache
    results.  If a previous query has been saved to disk, load that.

    Args:
        forceNew (bool):
            Run a fresh query even if results exist on disk.

    Returns:
        pandas.DataFrame:
            Planetary Systems table
    """

    basestr = "updateExoplanetArchivePS"
    querystring = r"select+*+from+ps"

    return EXOSIMS.util.getExoplanetArchive.cacheExoplanetArchiveQuery(basestr, querystring, forceNew=forceNew, **specs)


def customGetExoplanetArchivePSCP(forceNew: bool = False, **specs: Any) -> pd.DataFrame:
    """
    Get the contents of the Exoplanet Archive's Planetary Systems Composite Parameters
    table and cache results.  If a previous query has been saved to disk, load that.

    Args:
        forceNew (bool):
            Run a fresh query even if results exist on disk.

    Returns:
        pandas.DataFrame:
            Planetary Systems composited parameters table
    """

    basestr = "updateExoplanetArchivePSCP"
    querystring = r"select+*+from+pscomppars"

    return EXOSIMS.util.getExoplanetArchive.cacheExoplanetArchiveQuery(basestr, querystring, forceNew=forceNew, **specs)


def get_store_ipac_helper(pscp_data : pd.DataFrame, ps_data : pd.DataFrame) -> pd.DataFrame:
    """
    
    Majority of getIpacData() from plandb_methods. This helps modulate code between ipac compilation functionality and caching functionality

    Args:
        pscp_data (pd.DataFrame): _description_
        ps_data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
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

def get_store_ipac() -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # Saves PSCP & PS data cache, for current update comparison, and for future update
    pscp_data_dummy = customGetExoplanetArchivePSCP(forceNew=True)
    ps_data_dummy = customGetExoplanetArchivePS(forceNew=True)
        
    downloads_dir = get_downloads_dir()
    print(downloads_dir)
    
    def extract_time(filename):
        time = filename.name[24:]
        return time
    
    
    # TODO: CHANGE THIS PATH, to the system
    ipac_cache_dir = Path(f'/Users/andrewchiu/.EXOSIMS/downloads')

    if ipac_cache_dir.exists():
        filtered_files = [f for f in ipac_cache_dir.iterdir() if f.is_file() and f.name.startswith("updateExoplanetArchivePS_")]
        if (len(filtered_files) > 1):
            sorted_files = sorted(filtered_files, key=extract_time)
            target_file = sorted_files[-2]
            new_file = sorted_files[-1]
            print(target_file)
            print(new_file)
        elif filtered_files:
            print("1 or 0 files, must initial update file.")
        else:
            print("No matching files.")
            
        filtered_files2 = [f for f in ipac_cache_dir.iterdir() if f.is_file() and f.name.startswith("updateExoplanetArchivePSCP_")]
        if (len(filtered_files2) > 1):
            sorted_files2 = sorted(filtered_files2, key=extract_time)
            target_file2 = sorted_files2[-2]
            new_file2 = sorted_files2[-1]
            print(target_file2)
            print(new_file2)
        elif filtered_files2:
            print("1 or 0 files, must initial update file.")
        else:
            print("No matching files.") 
            
        with open(target_file, 'rb') as file:
            ps_data = pickle.load(file)
        
        with open(target_file2, 'rb') as file2:
            pscp_data = pickle.load(file2)
            
        with open(new_file, 'rb') as file3:
            new_ps_data = pickle.load(file3)
        
        with open(new_file2, 'rb') as file4:
            new_pscp_data = pickle.load(file4)
            
    old_ipac_data = get_store_ipac_helper(pscp_data=pscp_data, ps_data = ps_data)
    
    new_ipac_data = get_store_ipac_helper(pscp_data=new_pscp_data, ps_data= new_ps_data)
    
    return old_ipac_data, new_ipac_data
            
def get_current_database(connection: sqlalchemy.Connection) -> pd.DataFrame:
  results = connection.execute(text("SELECT * FROM PLANETS"))
  df = pd.DataFrame(results.fetchall(), columns = results.keys())
  return df
  
def get_ipac_database(data_path: Path, cache: bool) -> pd.DataFrame:
  # data_path = Path(f'cache/data_cache_{datestr}.p')
  if cache:
      Path('cache/').mkdir(parents=True, exist_ok=True)
      if data_path.exists():
          with open(data_path, 'rb') as f:
              ipac_data = pickle.load(f)
              return ipac_data
      else:
          ipac_data = getIPACdata()
          with open(data_path, 'wb') as f:
              pickle.dump(ipac_data, f)
          return ipac_data
  else:
      ipac_data = getIPACdata()
      return ipac_data
  
  
  #Could i have just done comparison = old_df == updated_df
def get_ipac_differences(old_df: pd.DataFrame, updated_df: pd.DataFrame, tolerance: float = 1e-1):
    col_names1 = old_df.columns.tolist()
    col_names2 = updated_df.columns.tolist()
    change_log = []
    
    if col_names1 != col_names2:
        return pd.DataFrame(), change_log  
    
    diff_df = pd.DataFrame()
    
    for index, row in old_df.iterrows():
        
        pl_name_ind = row['pl_name']
        corresponding_updated_row = updated_df.loc[updated_df['pl_name'] == pl_name_ind]
        
        if corresponding_updated_row.empty:
            diff_df = pd.concat([diff_df, pd.DataFrame([row])], ignore_index=True)
        else:
            corresponding_updated_row = corresponding_updated_row.iloc[0]
            differences = {}
            
            for col in old_df.columns:
                old_value = row[col]
                new_value = corresponding_updated_row[col] 
                
                # Nan's
                if pd.isna(old_value) and pd.isna(new_value):
                    continue  # Nan's are equal 
                
                # Tolerance
                if isinstance(old_value, (int, float, np.number)) and isinstance(new_value, (int, float, np.number)):
                    if abs(old_value - new_value) <= tolerance:
                        continue  # Tolerance considered equal 
                
                if old_value != new_value:
                    differences[col] = {"old": old_value, "new": new_value}   
             
            if differences: 
                diff_df = pd.concat([diff_df, pd.DataFrame([corresponding_updated_row])], ignore_index=True)
         
        updated_df = updated_df.drop(corresponding_updated_row.name)
    
    if not updated_df.empty:
        diff_df = pd.concat([diff_df, updated_df], ignore_index=True)
    
    return diff_df, change_log  



def upsert_general(old_df: pd.DataFrame, new_df: pd.DataFrame, col : str) -> pd.DataFrame:
    updated_old_df = old_df[~old_df[col].isin(new_df[col])]
    
    return pd.concat([updated_old_df, new_df], ignore_index=True)
        
    
def get_photo_data(photdict_path: Path, infile: str, cache: bool):
  # photdict_path = Path(f'cache/update_photdict_2022-05.p')
  if cache:
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if photdict_path.exists():
      with open(photdict_path, 'rb') as f:
        photdict = pickle.load(f)
        return photdict
    else:
      # photdict = loadPhotometryData(infile="plandb.sioslab.com/allphotdata_2015.npz")
      photdict = loadPhotometryData(infile=infile)
      with open(photdict_path, 'wb') as f:
        pickle.dump(photdict, f)
      return photdict
  else:
    # photdict = loadPhotometryData(infile="plandb.sioslab.com/allphotdata_2015.npz")
    photdict = loadPhotometryData(infile=infile)
    return photdict
  
def get_bandzip(bandzip_path: Path, cache: bool):
    # bandzip_path = Path(f'cache/update_bandzip_{datestr}.p')
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if bandzip_path.exists():
            with open(bandzip_path, 'rb') as f:
                bandzip = pickle.load(f)
                return bandzip
        else:
            bandzip = list(genBands())
            with open(bandzip_path, 'wb') as f:
                pickle.dump(bandzip, f)
            return bandzip
    else:
        bandzip = list(genBands())
        return bandzip
      
def get_orbdata(orbdata_path: Path, orbfits_path: Path, data: pd.DataFrame, bandzip, photdict, cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # orbdata_path = Path(f'cache/update_orbdata_{datestr}.p')
  # orbfits_path = Path(f'cache/update_orbfits_{datestr}.p')
  if cache:
      orbdata = pd.DataFrame()
      orbitfits = pd.DataFrame()
      Path('cache/').mkdir(parents=True, exist_ok=True)
      if orbdata_path.exists():
            with open(orbdata_path, 'rb') as f:
                orbdata = pickle.load(f)
      if orbfits_path.exists():
            with open(orbfits_path, 'rb') as f:
                orbitfits = pickle.load(f)
      else:
            orbdata, orbitfits = genOrbitData(data, bandzip, photdict)
            with open(orbdata_path, 'wb') as f:
                pickle.dump(orbdata, f)
            with open(orbfits_path, 'wb') as f:
                pickle.dump(orbitfits, f)
  else:
        orbdata, orbitfits = genOrbitData(data, bandzip, photdict)
  return orbdata, orbitfits
        
def get_ephemerisdata(ephemeris_orbdata_path: Path, ephemeris_orbfits_path: Path, data: pd.DataFrame, orbitfits: pd.DataFrame, orbdata: pd.DataFrame, bandzip, photdict, cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ephemeris_orbdata_path = Path(f'cache/update_ephemeris_orbdata_{datestr}.p')
    # ephemeris_orbfits_path = Path(f'cache/update_ephemeris_orbfits_{datestr}.p')
    ephemeris_orbitfits = pd.DataFrame()
    ephemeris_orbdata = pd.DataFrame()
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if ephemeris_orbdata_path.exists():
            with open(ephemeris_orbdata_path, 'rb') as f:
                ephemeris_orbdata = pickle.load(f)
        if ephemeris_orbfits_path.exists():
            with open(ephemeris_orbfits_path, 'rb') as f:
                ephemeris_orbitfits = pickle.load(f)
        else:
            ephemeris_orbitfits, ephemeris_orbdata = addEphemeris(data, orbitfits, orbdata, bandzip, photdict)
            with open(ephemeris_orbdata_path, 'wb') as f:
                pickle.dump(ephemeris_orbdata, f)
            with open(ephemeris_orbfits_path, 'wb') as f:
                pickle.dump(ephemeris_orbitfits, f)
    else:
        ephemeris_orbitfits, ephemeris_orbdata = addEphemeris(data, orbitfits, orbdata, bandzip, photdict)
    return ephemeris_orbitfits, ephemeris_orbdata
  
def get_quadrature(quadrature_data_path: Path, ephemeris_orbitfits: pd.DataFrame, bandzip, photdict, cache: bool) -> pd.DataFrame:
  # quadrature_data_path = Path(f'cache/update_quadrature_data_{datestr}.p')
  quadrature_data = pd.DataFrame()
  if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if quadrature_data_path.exists():
            with open(quadrature_data_path, 'rb') as f:
                quadrature_data = pickle.load(f)
        else:
            quadrature_data = calcQuadratureVals(ephemeris_orbitfits, bandzip, photdict)
            with open(quadrature_data_path, 'wb') as f:
                pickle.dump(quadrature_data, f)
  else:
        quadrature_data = calcQuadratureVals(ephemeris_orbitfits, bandzip, photdict)
        
  return quadrature_data

def get_contrastness(contr_data_path: Path, exosims_json, quadrature_data: pd.DataFrame, cache: bool) -> pd.DataFrame:
    # contr_data_path = Path(f'cache/update_contr_data_{datestr}.p')
    # exosims_json = 'plandb.sioslab.com/ci_perf_exosims.json'
    contr_data = pd.DataFrame()
    if cache:
        if contr_data_path.exists():
            contr_data = pd.read_pickle(contr_data_path)
        else:
            contr_data = calcContrastCurves(quadrature_data, exosims_json=exosims_json)
            with open(contr_data_path, 'wb') as f:
                pickle.dump(contr_data, f)
    else:
        contr_data = calcContrastCurves(quadrature_data, exosims_json=exosims_json)
    return contr_data
  
def get_completeness(comps_path: Path, compdict_path: Path, comps_data_path: Path, contr_data: pd.DataFrame, bandzip, photdict, exosims_json, cache: bool) -> Tuple[pd.DataFrame, dict, pd.DataFrame]: 
    # comps_path = Path(f'cache/update_comps_{datestr}.p')
    # compdict_path = Path(f'cache/update_compdict_{datestr}.p')
    # comps_data_path = Path(f'cache/update_comps_data_{datestr}.p')
    comps = pd.DataFrame()
    compdict = pd.DataFrame()
    comps_data = pd.DataFrame()
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if comps_path.exists():
            comps = pd.read_pickle(comps_path)
            with open(compdict_path, 'rb') as f:
                compdict = pickle.load(f)
            comps_data = pd.read_pickle(comps_data_path)
        else:
            comps, compdict, comps_data = calcPlanetCompleteness(contr_data, bandzip, photdict, exosims_json=exosims_json)
            comps.to_pickle(comps_path)
            with open(compdict_path, 'wb') as f:
                pickle.dump(compdict, f)
            comps_data.to_pickle(comps_data_path)
    else:
        comps, compdict, comps_data = calcPlanetCompleteness(contr_data, bandzip, photdict, exosims_json=exosims_json)
    return comps, compdict, comps_data
  
def get_generated_tables(plandata_path: Path, stdata_path: Path, table_orbitfits_path: Path, data: pd.DataFrame, quadrature_data: pd.DataFrame , comps_data : pd.DataFrame, cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # plandata_path = Path(f'cache/update_plandata_{datestr}.p')
    # stdata_path = Path(f'cache/update_stdata_{datestr}.p')
    # table_orbitfits_path = Path(f'cache/update_table_orbitfits_{datestr}.p')
    plandata = pd.DataFrame()
    stdata = pd.DataFrame()
    orbitfits = pd.DataFrame()
    if cache:
        # plandata.to_pickle('plandata_'+datestr+'.pkl')
        # stdata.to_pickle('stdata_' + datestr + '.pkl')
        # orbitfits.to_pickle('orbitfits_'+datestr+'.pkl')
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if plandata_path.exists():
            plandata = pd.read_pickle(plandata_path)
            stdata = pd.read_pickle(stdata_path)
            table_data = pd.read_pickle(table_orbitfits_path)
        else:
            plandata, stdata, orbitfits = generateTables(data, quadrature_data)
            plandata.to_pickle(plandata_path)
            stdata.to_pickle(stdata_path)
            orbitfits.to_pickle(table_orbitfits_path)
    else:
        plandata, stdata, orbitfits = generateTables(data, comps_data)   
    return plandata, stdata, orbitfits
  
# TODO make get_contrast, and compile, within one func
def get_compiled_contrast(compiled_contr_curvs_path: Path, stars: pd.DataFrame, pdfs: pd.DataFrame, planets: pd.DataFrame, contr_curvs_path: Path):
    # compiled_contr_curvs_path = Path(f'plandb.sioslab.com/cache/compiled_cont_curvs_{datestr}.p')
    # contr_curvs_path = Path(f'plandb.sioslab.com/cache/cont_curvs_{datestr.replace("-", "_")}')
    contrast_curves = pd.DataFrame()
    newpdfs = pd.DataFrame()
    if compiled_contr_curvs_path.exists():
        contrast_curves = pd.read_pickle(compiled_contr_curvs_path)
    else:
        contrast_curves = compileContrastCurves(stars, contr_curvs_path)
        contrast_curves.to_pickle(compiled_contr_curvs_path)
    # TODO finish new pdfs
    # def addId(r):
    #     r['pl_id']= list(planets.index[(planets['pl_name'] == r['Name'])])[0]
    #     return r
    # TODO figure out whats happening, why does the original of this from database_main use pdfs = comp_data
    # newpdfs = pdfs.apply(addId, axis = 1)
    
    return contrast_curves, newpdfs    

def get_compiled_completeness(compiled_completeness_path: Path, comple_data: pd.DataFrame) -> pd.DataFrame:
    # scenarios = pd.read_csv("plandb.sioslab.com/cache/scenario_angles.csv")
    # compiled_completeness_path = Path(f"plandb.sioslab.com/cache/compiled_completeness_{datestr}.p")
    
    def compileCompleteness(comp_data: pd.DataFrame) -> pd.DataFrame:
      # datestr = Time.now().datetime.strftime("%Y-%m")
      # comp_data = pd.read_pickle(f"plandb.sioslab.com/cache/comps_data_{datestr}.p")
      completeness = pd.DataFrame()
      col_names = comp_data.columns.values.tolist()
      scenario_names = []
      for x in col_names:
          if x[:8] == 'complete':
              scenario_names.append(x[13:])
      #drop contr_curve col
      completeness = pd.DataFrame([], columns = ['pl_id', 'completeness',  'scenario_name', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag'])
      for i, row in tqdm(comp_data.iterrows()):
          newRows = []
          for scenario_name in scenario_names:
              newRows = []
              if pd.notna(row[('completeness_' + scenario_name)]):
                  newRows.append([row['pl_id'], row['completeness_' + scenario_name], scenario_name,
                      row['compMinWA_' + scenario_name], 
                      row['compMaxWA_' + scenario_name],
                      row['compMindMag_' + scenario_name],
                      row['compMaxdMag_' + scenario_name]])

                  singleRow = pd.DataFrame(newRows, columns =  ['pl_id', 'completeness',  'scenario_name', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag'])
                  completeness = completeness._append(singleRow, ignore_index=True)
      # print(completeness)
      return completeness
  
    completeness = pd.DataFrame()
    if compiled_completeness_path.exists():
        completeness = pd.read_pickle(compiled_completeness_path)
    else:
        completeness = compileCompleteness(comple_data)
        completeness.to_pickle(compiled_completeness_path)
    return completeness
        
# Create a new shorter database of the updated, that is able to merge later 
# TODO change this to only make dataframe, and then add to database
def update_sql(engine, plandata=None, stdata=None, orbitfits=None, orbdata=None, pdfs=None, aliases=None,contrastCurves=None,scenarios=None, completeness=None):
  return
  
def get_all_from_db(connection: sqlalchemy.Connection) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  planets_results = connection.execute(text("SELECT * FROM PLANETS"))
  planets_df = pd.DataFrame(planets_results.fetchall(), columns = planets_results.keys())
  
  contrast_curves_results = connection.execute(text("SELECT * FROM ContrastCurves"))
  contrast_curves_df = pd.DataFrame(contrast_curves_results.fetchall(), columns = contrast_curves_results.keys())
  
  orbitfits_results = connection.execute(text("SELECT * FROM OrbitFits"))
  orbitfits_df = pd.DataFrame(orbitfits_results.fetchall(), columns = orbitfits_results.keys())

  
  orbits_results = connection.execute(text("SELECT * FROM Orbits"))
  orbits_df = pd.DataFrame(orbits_results.fetchall(), columns = orbits_results.keys())
  
  # TODO, fix to pdfs, do completeness for now, as pdfs is currently empty from diff_database
  pdfs_results = connection.execute(text("SELECT * FROM Completeness"))
  pdfs_df = pd.DataFrame(pdfs_results.fetchall(), columns = pdfs_results.keys())
  
  scenarios_results = connection.execute(text("SELECT * FROM Scenarios"))
  scenarios_df = pd.DataFrame(scenarios_results.fetchall(), columns = scenarios_results.keys())
  
  stars_results = connection.execute(text("SELECT * FROM Stars"))
  stars_df = pd.DataFrame(stars_results.fetchall(), columns = stars_results.keys())
  
  completeness_results = connection.execute(text("SELECT * FROM Completeness"))
  completeness_df = pd.DataFrame(completeness_results.fetchall(), columns = completeness_results.keys())
  
  return completeness_df, contrast_curves_df, orbitfits_df, orbits_df, pdfs_df, planets_df, scenarios_df, stars_df

def temp_writeSQL(engine, plandata=None, stdata=None, orbitfits=None, orbdata=None, pdfs=None, aliases=None,contrastCurves=None,scenarios=None, completeness=None):
    """write outputs to sql database via connection"""
    connection = engine.connect()
    connection.execute(text("DROP TABLE IF EXISTS Completeness, ContrastCurves, Scenarios, PDFs, Orbits, OrbitFits, Planets, Stars"))

    if stdata is not None:
        print("Writing Stars")
        namemxchar = np.array([len(n) for n in stdata['st_name'].values]).max()
        stdata = stdata.rename_axis('st_id')
        stdata.to_sql('Stars', connection, chunksize=100, if_exists='replace',
                    dtype={'st_id': sqlalchemy.types.INT,
                           'st_name': sqlalchemy.types.String(namemxchar)})
        # set indexes
        result = connection.execute(text('ALTER TABLE Stars ADD INDEX (st_id)'))

        # add comments
        # addSQLcomments(connection, 'Stars')

    if plandata is not None:
        print("Writing Planets")
        namemxchar = np.array([len(n) for n in plandata['pl_name'].values]).max()
        plandata = plandata.rename_axis('pl_id')
        plandata.to_sql('Planets',connection,chunksize=100,if_exists='replace',
                    dtype={'pl_id':sqlalchemy.types.INT,
                            'pl_name':sqlalchemy.types.String(namemxchar),
                            'st_name':sqlalchemy.types.String(namemxchar-2),
                            'pl_letter':sqlalchemy.types.CHAR(1),
                            'st_id': sqlalchemy.types.INT})
        #set indexes
        result = connection.execute(text("ALTER TABLE Planets ADD INDEX (pl_id)"))
        result = connection.execute(text("ALTER TABLE Planets ADD INDEX (st_id)"))
        # result = connection.execute(text("ALTER TABLE Planets ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));

        #add comments
        # addSQLcomments(connection,'Planets')

    if orbitfits is not None:
        print("Writing OrbitFits")
        orbitfits = orbitfits.rename_axis('orbitfit_id')
        namemxchar = np.array([len(n) for n in orbitfits['pl_name'].values]).max()
        orbitfits.to_sql('OrbitFits',connection,chunksize=100,if_exists='replace',
                          dtype={'pl_id': sqlalchemy.types.INT,
                                 'orbitfit_id': sqlalchemy.types.INT,
                                 'pl_name': sqlalchemy.types.String(namemxchar)},
                          index=True)
        result = connection.execute(text("ALTER TABLE OrbitFits ADD INDEX (orbitfit_id)"))
        result = connection.execute(text("ALTER TABLE OrbitFits ADD INDEX (pl_id)"))
        # TODO Get this commented back in to correctly have foreign keys
        # result = connection.execute(text("ALTER TABLE OrbitFits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));

        # addSQLcomments(connection,'OrbitFits')

    if orbdata is not None:
        print("Writing Orbits")
        namemxchar = np.array([len(n) for n in orbdata['pl_name'].values]).max()
        orbdata = orbdata.rename_axis('orbit_id')
        orbdata.to_sql('Orbits',connection,chunksize=100,if_exists='replace',
                       dtype={'pl_name':sqlalchemy.types.String(namemxchar),
                              'pl_id': sqlalchemy.types.INT,
                              'orbit_id': sqlalchemy.types.BIGINT,
                              'orbitfit_id': sqlalchemy.types.INT},
                       index=True)
        result = connection.execute(text("ALTER TABLE Orbits ADD INDEX (orbit_id)"))
        result = connection.execute(text("ALTER TABLE Orbits ADD INDEX (pl_id)"))
        result = connection.execute(text("ALTER TABLE Orbits ADD INDEX (orbitfit_id)"))
        #TODO Get this commented back in to correctly have foreign keys
        # result = connection.execute(text("ALTER TABLE Orbits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));
        # result = connection.execute(text("ALTER TABLE Orbits ADD FOREIGN KEY (orbitfit_id) REFERENCES OrbitFits(orbitfit_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));

        # addSQLcomments(connection,'Orbits')

    if pdfs is not None:
        print("Writing PDFs")
        pdfs = pdfs.reset_index(drop=True)
        namemxchar = np.array([len(n) for n in pdfs['Name'].values]).max()
        pdfs = pdfs.rename_axis('pdf_id')
        pdfs.to_sql('PDFs',connection,chunksize=100,if_exists='replace',
                     dtype={'pl_name':sqlalchemy.types.String(namemxchar),
                            'pl_id': sqlalchemy.types.INT})
        # result = connection.execute("ALTER TABLE PDFs ADD INDEX (orbitfit_id)")
        result = connection.execute(text("ALTER TABLE PDFs ADD INDEX (pl_id)"))
        result = connection.execute(text("ALTER TABLE PDFs ADD INDEX (pdf_id)"))
        # result = connection.execute("ALTER TABLE PDFs ADD FOREIGN KEY (orbitfit_id) REFERENCES OrbitFits(orbitfit_id) ON DELETE NO ACTION ON UPDATE NO ACTION")
        result = connection.execute(text("ALTER TABLE PDFs ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))

        # addSQLcomments(connection,'PDFs')

    if aliases is not None:
        print("Writing Alias")
        aliases = aliases.rename_axis('alias_id')
        aliasmxchar = np.array([len(n) for n in aliases['Alias'].values]).max()
        aliases.to_sql('Aliases',connection,chunksize=100,if_exists='replace',dtype={'Alias':sqlalchemy.types.String(aliasmxchar)})
        result = connection.execute(text("ALTER TABLE Aliases ADD INDEX (alias_id)"))
        result = connection.execute(text("ALTER TABLE Aliases ADD INDEX (Alias)"))
        result = connection.execute(text("ALTER TABLE Aliases ADD INDEX (st_id)"))
        result = connection.execute(text("ALTER TABLE Aliases ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))

    if scenarios is not None:
        print("Writing Scenarios")

        namemxchar = np.array([len(n) for n in scenarios['scenario_name'].values]).max()
        scenarios.to_sql("Scenarios", connection, chunksize=100, if_exists='replace', dtype={
            'scenario_name': sqlalchemy.types.String(namemxchar),}, index = False)

        result = connection.execute(text("ALTER TABLE Scenarios ADD INDEX (scenario_name)"))

    if contrastCurves is not None:
        print("Writing ContrastCurves")
        contrastCurves = contrastCurves.rename_axis("curve_id")
        #TODO change namemx char to commented out line, temporary value of 1000000 used here as a placeholder
        # namemxchar = np.array([len(n) for n in contrastCurves['scenario_name'].values]).max()
        namemxchar = 1000
        contrastCurves.to_sql("ContrastCurves", connection, chunksize=100, if_exists='replace', dtype={
            'st_id': sqlalchemy.types.INT,
            'curve_id' : sqlalchemy.types.INT,
            'scenario_name': sqlalchemy.types.String(namemxchar)}, index = True)
        # result = connection.execute("ALTER TABLE ContastCurves ADD INDEX (scenario_name)")

        # result = connection.execute("ALTER TABLE ContastCurves ADD INDEX (st_id)")

        # TODO: get this commented back in to have correct foreign keys
        # result = connection.execute(text("ALTER TABLE ContrastCurves ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))
        # result = connection.execute(text("ALTER TABLE ContrastCurves ADD FOREIGN KEY (scenario_name) REFERENCES Scenarios(scenario_name) ON DELETE NO ACTION ON UPDATE NO ACTION"))

    if completeness is not None:
        print("Writing completeness")
        completeness = completeness.rename_axis("completeness_id")
        namemxchar = np.array([len(n) for n in completeness['scenario_name'].values]).max()
        completeness.to_sql("Completeness", connection, chunksize=100, if_exists='replace', dtype={
            # 'scenario_id': sqlalchemy.types.INT,
            'pl_id' : sqlalchemy.types.INT,
            'completeness_id' : sqlalchemy.types.INT,
            'scenario_name': sqlalchemy.types.String(namemxchar)}, index = True)

        # TODO: get this commented back in to have correct foreign keys
        # result = connection.execute(text("ALTER TABLE Completeness ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))
        # result = connection.execute(text("ALTER TABLE ContrastCurves ADD FOREIGN KEY (scenario_name) REFERENCES Scenarios(scenario_name) ON DELETE NO ACTION ON UPDATE NO ACTION"))

  
  
  
  
  
def final_writeSQL(engine, plandata=None, stdata=None, orbitfits=None, orbdata=None, pdfs=None, aliases=None,contrastCurves=None,scenarios=None, completeness=None):
    """write outputs to sql database via connection"""
    connection = engine.connect()
    connection.execute(text("DROP TABLE IF EXISTS Completeness, ContrastCurves, Scenarios, PDFs, Orbits, OrbitFits, Planets, Stars"))

    if stdata is not None:
        # print("Writing Stars")
        # namemxchar = np.array([len(n) for n in stdata['st_name'].values]).max()
        # stdata = stdata.rename_axis('st_id')
        # stdata.to_sql('Stars', connection, chunksize=100, if_exists='replace',
        #             dtype={'st_id': sqlalchemy.types.INT,
        #                    'st_name': sqlalchemy.types.String(namemxchar)})
        # # set indexes
        # result = connection.execute(text('ALTER TABLE Stars ADD INDEX (st_id)'))
        stdata.to_sql('Stars', connection)
        # add comments
        # addSQLcomments(connection, 'Stars')

    if plandata is not None:
        print("Writing Planets")
        # namemxchar = np.array([len(n) for n in plandata['pl_name'].values]).max()
        # plandata = plandata.rename_axis('pl_id')
        # plandata.to_sql('Planets',connection,chunksize=100,if_exists='replace',
        #             dtype={'pl_id':sqlalchemy.types.INT,
        #                     'pl_name':sqlalchemy.types.String(namemxchar),
        #                     'st_name':sqlalchemy.types.String(namemxchar-2),
        #                     'pl_letter':sqlalchemy.types.CHAR(1),
        #                     'st_id': sqlalchemy.types.INT})
        # #set indexes
        # result = connection.execute(text("ALTER TABLE Planets ADD INDEX (pl_id)"))
        # result = connection.execute(text("ALTER TABLE Planets ADD INDEX (st_id)"))
        # result = connection.execute(text("ALTER TABLE Planets ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));
        plandata.to_sql('Planets', connection)
        #add comments
        # addSQLcomments(connection,'Planets')

    if orbitfits is not None:
        print("Writing OrbitFits")
        # TODO: figure out if this line is okay to keep
        # orbitfits = orbitfits.rename_axis('orbitfit_id')
        namemxchar = np.array([len(n) for n in orbitfits['pl_name'].values]).max()
        orbitfits.to_sql('OrbitFits',connection,chunksize=100,if_exists='replace',
                          dtype={'pl_id': sqlalchemy.types.INT,
                                 'orbitfit_id': sqlalchemy.types.INT,
                                 'pl_name': sqlalchemy.types.String(namemxchar)},
                          index=True)
        # result = connection.execute(text("ALTER TABLE OrbitFits ADD INDEX (orbitfit_id)"))
        # result = connection.execute(text("ALTER TABLE OrbitFits ADD INDEX (pl_id)"))
        # TODO Get this commented back in to correctly have foreign keys
        # result = connection.execute(text("ALTER TABLE OrbitFits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));
        # orbitfits.to_sql('OrbitFits', connection)
        result = connection.execute(text("ALTER TABLE OrbitFits ROW_FORMAT=COMPRESSED"))

        # addSQLcomments(connection,'OrbitFits')

    if orbdata is not None:
        print("Writing Orbits")
        namemxchar = np.array([len(n) for n in orbdata['pl_name'].values]).max()
        orbdata = orbdata.rename_axis('orbit_id')
        orbdata.to_sql('Orbits',connection,chunksize=100,if_exists='replace',
                       dtype={'pl_name':sqlalchemy.types.String(namemxchar),
                              'pl_id': sqlalchemy.types.INT,
                              'orbit_id': sqlalchemy.types.BIGINT,
                              'orbitfit_id': sqlalchemy.types.INT},
                       index=True)
        result = connection.execute(text("ALTER TABLE Orbits ADD INDEX (orbit_id)"))
        result = connection.execute(text("ALTER TABLE Orbits ADD INDEX (pl_id)"))
        result = connection.execute(text("ALTER TABLE Orbits ADD INDEX (orbitfit_id)"))
        # TODO Get this commented back in to correctly have foreign keys
        # result = connection.execute(text("ALTER TABLE Orbits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));
        # result = connection.execute(text("ALTER TABLE Orbits ADD FOREIGN KEY (orbitfit_id) REFERENCES OrbitFits(orbitfit_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));
        orbdata.to_sql('Orbits', connection)
        # addSQLcomments(connection,'Orbits')

    if pdfs is not None:
        # print("Writing PDFs")
        # pdfs = pdfs.reset_index(drop=True)
        # namemxchar = np.array([len(n) for n in pdfs['Name'].values]).max()
        # pdfs = pdfs.rename_axis('pdf_id')
        # pdfs.to_sql('PDFs',connection,chunksize=100,if_exists='replace',
        #              dtype={'pl_name':sqlalchemy.types.String(namemxchar),
        #                     'pl_id': sqlalchemy.types.INT})
        # # result = connection.execute("ALTER TABLE PDFs ADD INDEX (orbitfit_id)")
        # result = connection.execute(text("ALTER TABLE PDFs ADD INDEX (pl_id)"))
        # result = connection.execute(text("ALTER TABLE PDFs ADD INDEX (pdf_id)"))
        # # result = connection.execute("ALTER TABLE PDFs ADD FOREIGN KEY (orbitfit_id) REFERENCES OrbitFits(orbitfit_id) ON DELETE NO ACTION ON UPDATE NO ACTION")
        # result = connection.execute(text("ALTER TABLE PDFs ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))
        pdfs.to_sql('PDFs', connection)
        # addSQLcomments(connection,'PDFs')

    if aliases is not None:
        print("Writing Alias")
        aliases.to_sql('Aliases', connection)
        # aliases = aliases.rename_axis('alias_id')
        # aliasmxchar = np.array([len(n) for n in aliases['Alias'].values]).max()
        # aliases.to_sql('Aliases',connection,chunksize=100,if_exists='replace',dtype={'Alias':sqlalchemy.types.String(aliasmxchar)})
        # result = connection.execute(text("ALTER TABLE Aliases ADD INDEX (alias_id)"))
        # result = connection.execute(text("ALTER TABLE Aliases ADD INDEX (Alias)"))
        # result = connection.execute(text("ALTER TABLE Aliases ADD INDEX (st_id)"))
        # result = connection.execute(text("ALTER TABLE Aliases ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))

    if scenarios is not None:
        print("Writing Scenarios")
        scenarios.to_sql("Scenarios", connection)

        # namemxchar = np.array([len(n) for n in scenarios['scenario_name'].values]).max()
        # scenarios.to_sql("Scenarios", connection, chunksize=100, if_exists='replace', dtype={
        #     'scenario_name': sqlalchemy.types.String(namemxchar),}, index = False)

        # result = connection.execute(text("ALTER TABLE Scenarios ADD INDEX (scenario_name)"))

    if contrastCurves is not None:
        print("Writing ContrastCurves")
        
        contrastCurves.to_sql("ContrastCurves", connection)
        # contrastCurves = contrastCurves.rename_axis("curve_id")
        # #TODO change namemx char to commented out line, temporary value of 1000000 used here as a placeholder
        # # namemxchar = np.array([len(n) for n in contrastCurves['scenario_name'].values]).max()
        # namemxchar = 1000
        # contrastCurves.to_sql("ContrastCurves", connection, chunksize=100, if_exists='replace', dtype={
        #     'st_id': sqlalchemy.types.INT,
        #     'curve_id' : sqlalchemy.types.INT,
        #     'scenario_name': sqlalchemy.types.String(namemxchar)}, index = True)
        # # result = connection.execute("ALTER TABLE ContastCurves ADD INDEX (scenario_name)")

        # result = connection.execute("ALTER TABLE ContastCurves ADD INDEX (st_id)")

        # TODO: get this commented back in to have correct foreign keys
        # result = connection.execute(text("ALTER TABLE ContrastCurves ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))
        # result = connection.execute(text("ALTER TABLE ContrastCurves ADD FOREIGN KEY (scenario_name) REFERENCES Scenarios(scenario_name) ON DELETE NO ACTION ON UPDATE NO ACTION"))

    if completeness is not None:
        # print("Writing completeness")
        # completeness = completeness.rename_axis("completeness_id")
        # namemxchar = np.array([len(n) for n in completeness['scenario_name'].values]).max()
        # completeness.to_sql("Completeness", connection, chunksize=100, if_exists='replace', dtype={
        #     # 'scenario_id': sqlalchemy.types.INT,
        #     'pl_id' : sqlalchemy.types.INT,
        #     'completeness_id' : sqlalchemy.types.INT,
        #     'scenario_name': sqlalchemy.types.String(namemxchar)}, index = True)
        completeness.to_sql("Completeness", connection)
        # TODO: get this commented back in to have correct foreign keys
        # result = connection.execute(text("ALTER TABLE Completeness ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))
        # result = connection.execute(text("ALTER TABLE ContrastCurves ADD FOREIGN KEY (scenario_name) REFERENCES Scenarios(scenario_name) ON DELETE NO ACTION ON UPDATE NO ACTION"))

  
  

