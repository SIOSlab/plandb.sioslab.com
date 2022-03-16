from __future__ import division, print_function

import getpass

import keyring

import pickle
from plandb_methods import *
import pandas as pd
from pathlib import Path

cache = True
from sqlalchemy.types import String

datestr = Time.now().datetime.strftime("%Y-%m")
# datestr = Time.now().datetime.strftime("%Y-%m-%d")

#initial data dump
data_path = Path(f'cache/data_cache_{datestr}.p')
if cache:
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = getIPACdata()
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
else:
    data = getIPACdata()
#photometric data
photdict_path = Path(f'cache/photdict_{datestr}.p')
if cache:
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if photdict_path.exists():
        with open(photdict_path, 'rb') as f:
            photdict = pickle.load(f)
    else:
        photdict = loadPhotometryData()
        with open(photdict_path, 'wb') as f:
            pickle.dump(photdict, f)
else:
    photdict = loadPhotometryData()

#band info
bandzip_path = Path(f'cache/bandzip_{datestr}.p')
if cache:
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if bandzip_path.exists():
        with open(bandzip_path, 'rb') as f:
            bandzip = pickle.load(f)
    else:
        bandzip = list(genBands())
        with open(bandzip_path, 'wb') as f:
            pickle.dump(bandzip, f)
else:
    bandzip = list(genBands())
#get orbital data
print("Getting orbit data")
orbdata_path = Path(f'cache/orbdata_{datestr}.p')
orbfits_path = Path(f'cache/orbfits_{datestr}.p')
if cache:
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

print("Adding ephemeris")
ephemeris_orbdata_path = Path(f'cache/ephemeris_orbdata_{datestr}.p')
ephemeris_orbfits_path = Path(f'cache/ephemeris_orbfits_{datestr}.p')
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

#calculate quadrature columns:
print('Computing quadrature values')
quadrature_data_path = Path(f'cache/quadrature_data_{datestr}.p')
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


# Calculating contrast curves for stars
print('Contrast curve calculations')
contr_data_path = Path(f'cache/contr_data_{datestr}.p')
exosims_json = 'ci_perf_exosims.json'
if cache:
    if contr_data_path.exists():
        contr_data = pd.read_pickle(contr_data_path)
    else:
        contr_data = calcContrastCurves(quadrature_data, exosims_json=exosims_json)
        with open(contr_data_path, 'wb') as f:
            pickle.dump(contr_data, f)
else:
    contr_data = calcContrastCurves(quadrature_data, exosims_json=exosims_json)

#completeness
print('Doing completeness calculations')
comps_path = Path(f'cache/comps_{datestr}.p')
compdict_path = Path(f'cache/compdict_{datestr}.p')
comps_data_path = Path(f'cache/comps_data_{datestr}.p')
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

# Split data into stars, planets, and orbitfits
plandata_path = Path(f'cache/plandata_{datestr}.p')
stdata_path = Path(f'cache/stdata_{datestr}.p')
table_orbitfits_path = Path(f'cache/table_orbitfits_{datestr}.p')
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
#aliases
# aliases_path = Path(f'cache/aliases_{datestr}.p')
# aliases_path = Path(f'cache/aliases_2021-09-22.p') # Saved version
# if cache:
    # Path('cache/').mkdir(parents=True, exist_ok=True)
    # if aliases_path.exists():
        # with open(aliases_path, 'rb') as f:
            # aliases = pickle.load(f)
    # else:
        # aliases = genAliases(comps_data)
        # with open(aliases_path, 'wb') as f:
            # pickle.dump(aliases_path, f)
# else:
    # aliases = genAliases(comps_data)
#testdb
# engine = create_engine('mysql+pymysql://cas584@127.0.0.1/plandb_newscratch', echo=False, encoding='utf8', pool_pre_ping=True)
# engine = create_engine(f'mysql+pymysql://corey:password@localhost/test', echo=False)

#proddb#################################################################################################
# username = 'dsavrans_admin'
# passwd = keyring.get_password('plandb_sql_login', username)
# if passwd is None:
    # passwd = getpass.getpass("Password for mysql user %s:\n"%username)
    # keyring.set_password('plandb_sql_login', username, passwd)

# engine = create_engine('mysql+pymysql://'+username+':'+passwd+'@sioslab.com/dsavrans_plandb',echo=False)
#proddb#################################################################################################
# df = pd.DataFrame({'name' : ['User1', "user2", "User 3"]})
# df.to_sql('users', con=engine)
# df.to_sql('test', con=engine, chunksize=100, if_exists='replace', index=False, dtype={"A": String()})
# writeSQL(engine,data=data)
# writeSQL(engine,orbdata=orbdata,altorbdata=altorbdata,comps=comps)
