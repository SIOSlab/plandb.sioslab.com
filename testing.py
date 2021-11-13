import pandas as pd
from plandb_methods import *

from sqlalchemy import create_engine

import pymysql

def writeSQL2(engine, data=None, stdata=None, orbitfits=None, orbdata=None, comps=None, aliases=None):
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
        addSQLcomments(engine, 'Stars')

    if data is not None:
        print("Writing Planets")
        namemxchar = np.array([len(n) for n in data['pl_name'].values]).max()
        data = data.rename_axis('pl_id')
        data.to_sql('Planets',engine,chunksize=100,if_exists='replace',
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
        addSQLcomments(engine,'Planets')

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

        addSQLcomments(engine,'OrbitFits')

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

        addSQLcomments(engine,'Orbits')

    if comps is not None:
        print("Writing Completeness")
        comps = comps.reset_index(drop=True)
        namemxchar = np.array([len(n) for n in comps['pl_name'].values]).max()
        comps = comps.rename_axis('completeness_id')
        comps.to_sql('Completeness',engine,chunksize=100,if_exists='replace',
                     dtype={'Name':sqlalchemy.types.String(namemxchar),
                            'pl_id': sqlalchemy.types.INT,
                            'orbitfit_id': sqlalchemy.types.INT})
        result = engine.execute("ALTER TABLE Completeness ADD INDEX (orbitfit_id)")
        result = engine.execute("ALTER TABLE Completeness ADD INDEX (pl_id)")
        result = engine.execute("ALTER TABLE Completeness ADD INDEX (completeness_id)")
        result = engine.execute("ALTER TABLE Completeness ADD FOREIGN KEY (orbitfit_id) REFERENCES OrbitFits(orbitfit_id) ON DELETE NO ACTION ON UPDATE NO ACTION");
        result = engine.execute("ALTER TABLE Completeness ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

        addSQLcomments(engine,'Completeness')

    if aliases is not None:
        print("Writing Alias")
        aliases = aliases.rename_axis('alias_id')
        aliasmxchar = np.array([len(n) for n in aliases['Alias'].values]).max()
        aliases.to_sql('Aliases',engine,chunksize=100,if_exists='replace',dtype={'Alias':sqlalchemy.types.String(aliasmxchar)})
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (alias_id)")
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (Alias)")
        result = engine.execute("ALTER TABLE Aliases ADD INDEX (st_id)")
        result = engine.execute("ALTER TABLE Aliases ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION");

def genOrbitData_2(data, bandzip, photdict, t0=None):
    if t0 is None:
        t0 = Time('2026-01-01T00:00:00', format='isot', scale='utc')

    plannames = data['pl_name'].values
    minWA = data['pl_minangsep'].values * u.mas
    maxWA = data['pl_maxangsep'].values * u.mas

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
        if np.isnan(I) or np.isnan(row['pl_orbinclerr1']) or np.isnan(row['pl_orbinclerr2']):
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
            lum = row['st_lum']
            if np.isnan(lum):
                lum_fix = 1
            else:
                lum_fix = (10 ** lum) ** .5  # Since lum is log base 10 of solar luminosity

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
                    pphi[np.isinf(pphi)] = np.nan
                    outdict['pPhi_' + "%03dC_" % (c * 100) + str(l) + "NM"] = pphi
                    allpphis[count1, count2] = pphi
                    dMag = deltaMag(1, Rp * u.R_jupiter, d * u.AU, pphi)
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

def     Gen(data, orbitfits):
    print('Splitting IPAC data into Star and Planet data')
    # Isolate the stellar columns from the planet columns
    st_cols = [col for col in data.columns if ('st_' in col) or ('gaia_' in col)]
    st_cols.extend(["dec", "dec_str", "hd_name", "hip_name", "ra", "ra_str"])
    # print(st_cols)

    pl_cols = ["pl_hostname", "pl_name", "pl_letter", "pl_disc", "pl_disc_refname", "pl_discmethod", "pl_locale", "pl_imgflag",
               "pl_instrument", "pl_k2flag", "pl_kepflag", "pl_telescope", "pl_publ_date", "pl_facility", "pl_edelink",
               "pl_mnum", "pl_rvflag", "pl_status", "pl_pelink"]
    orbitfits_cols = np.setdiff1d(orbitfits.columns, st_cols)
    orbitfits_exclude_prefix = ('pl_massj', 'pl_sinij', 'pl_masse', 'pl_msinie', 'pl_bmasse', 'pl_rade', 'pl_rads', 'pl_defrefname')
    orbitfits_exclude = [col for col in orbitfits_cols if col.startswith(orbitfits_exclude_prefix)]
    # print(orbitfits_cols)
    # print(np.asarray(orbitfits_exclude))
    orbitfits_cols = np.setdiff1d(orbitfits_cols, orbitfits_exclude)
    # np.append(orbitfits_cols, ['orbtper_next', 'orbtper_2026', 'is_Icrit', 'default_fit'])

    st_cols.append("pl_hostname")
    # st_cols.remove("pl_st_npar")
    # st_cols.remove("pl_st_nref")
    st_data = data[st_cols]
    st_data = st_data.drop_duplicates(keep='first')
    st_data.reset_index(drop=True)

    # pl_cols = [col for col in pl_cols if 'pl_' not in col]
    # pl_cols = np.append(pl_cols, ['pl_name', 'pl_letter', 'pl_hostname'])

    # pl_cols = ['pl_name', 'pl_letter', 'pl_hostname']

    # print(pl_cols)5

    # pl_data_dict = dict()
    # for col_name in pl_cols:
    #     pl_data_dict[col_name.replace('pl_', '')] = data[col_name]

    # pl_data = data[pl_cols]
    # pl_data = pd.DataFrame(pl_data_dict)
    pl_data = data[pl_cols]
    orbitfits = orbitfits[orbitfits_cols]
    pl_data = pl_data.rename(columns={'pl_hostname':'st_name'})
    st_data = st_data.rename(columns={'pl_hostname':'st_name'})
    orbitfits = orbitfits.rename(columns={'pl_hostname':'st_name'})

    # print(orbitfits_data['pl_name'].values)
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

    # keep_pl = ['pl_name', 'pl_id']
    colmap_fits = {k: k[3:] if (k.startswith('pl_') and (k != 'pl_id' and k != 'pl_name')) else k for k in orbitfits.keys()}
    orbitfits = orbitfits.rename(columns=colmap_fits)
    return pl_data, st_data, orbitfits

if __name__ == "__main__":
    testing = False
    if not testing:
        # data = pd.read_csv("/data/plandb/plandb.sioslab.com/csv_data.csv")
        # bandzip = genBands()
        # data = calcContrastCurves(data)
        # photdict = loadPhotometryData("/data/plandb/plandb.sioslab.com/allphotdata_2015.npz")
        # completeness, _, data = calcPlanetCompleteness(data, bandzip, photdict)
        # completeness.to_csv("./csvs/completeness")
        # data.to_csv("./csvs/data2")
        completeness = pd.read_csv("./csvs/completeness")
        data = pd.read_csv("./csvs/data2")
        # aliases = genAliases(data)
        # aliases.to_csv("./csvs/aliases")
        # orb_data, orbitfits = genOrbitData_2(data, bandzip, photdict)
        # orb_data.to_csv("./csvs/orb_data")
        # orbitfits.to_csv("./csvs/orbfits")

        orb_data = pd.read_csv("./csvs/orb_data")
        orbitfits = pd.read_csv("./csvs/orbfits")
        print("generating tables")

        pl_data, st_data, orbitfits_data = generateTables(data, orbitfits) 
        print("Writing csvs")
        pl_data.to_csv("./csvs/pl_data")
        st_data.to_csv("./csvs/st_data")
        orbfits_data.to_csv("./csvs/orbfits_data")

    
    username = 'plandb_admin'
    passwd = input("db password: ")
    engine = create_engine('mysql+pymysql://'+username+':'+passwd+'@localhost/plandb_scratch',echo=False)
    print("engine made")
    # engine.execute()
    if testing:
        data = [['tom', 10], ['nick', 15], ['juli', 14]]
        # Create the pandas DataFrame
        df = pd.DataFrame(data, columns = ['Name', 'Age'])
        df.to_sql('test', engine, chunksize=100, if_exists='replace')
    else:
        writeSQL2(engine, data= pl_data, stdata= st_data, orbitfits=orbfits_data, orbdata=orb_data, comps=completeness)
