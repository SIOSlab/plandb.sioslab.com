from __future__ import division, print_function

import getpass

import keyring

import pickle
from plandb_methods import *
import pandas as pd
from pathlib import Path

cache = True
from sqlalchemy.types import String

datestr = Time.now().datetime.strftime("%Y-%m-%d")

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

#calculate quadrature columns:
print('Computing quadrature values')
quadrature_data_path = Path(f'cache/quadrature_data_{datestr}.p')
if cache:
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if quadrature_data_path.exists():
        with open(quadrature_data_path, 'rb') as f:
            quadrature_data = pickle.load(f)
    else:
        quadrature_data = calcQuadratureVals(data, bandzip, photdict)
        with open(quadrature_data_path, 'wb') as f:
            pickle.dump(quadrature_data, f)
else:
    quadrature_data = calcQuadratureVals(data, bandzip, photdict)

#get orbital data
print("Getting orbit data")
orbdata_path = Path(f'cache/orbdata_{datestr}.p')
if cache:
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if orbdata_path.exists():
        with open(orbdata_path, 'rb') as f:
            orbdata = pickle.load(f)
    else:
        orbdata = genOrbitData(quadrature_data, bandzip, photdict)
        with open(orbdata_path, 'wb') as f:
            pickle.dump(orbdata, f)
else:
    orbdata = genOrbitData(quadrature_data, bandzip, photdict)

# print('Getting alt orbit data')
altorbdata_path = Path(f'cache/altorbdata_{datestr}.p')
if cache:
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if altorbdata_path.exists():
        with open(altorbdata_path, 'rb') as f:
            altorbdata = pickle.load(f)
    else:
        altorbdata = genAltOrbitData(quadrature_data, bandzip, photdict)
        with open(altorbdata_path, 'wb') as f:
            pickle.dump(altorbdata, f)
else:
    altorbdata = genAltOrbitData(quadrature_data, bandzip, photdict)

# Calculating contrast curves for stars
print('Contrast curve calculations')
contr_data_path = Path(f'cache/contr_data_{datestr}.p')
if cache:
    if contr_data_path.exists():
        contr_data = pd.read_pickle(contr_data_path)
    else:
        contr_data = calcContrastCurves(quadrature_data, 'CGPERF_HLC_20190210b')
        with open(contr_data_path, 'wb') as f:
            pickle.dump(contr_data, f)
else:
    contr_data = calcContrastCurves(quadrature_data, 'CGPERF_HLC_20190210b')

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
        comps, compdict, comps_data = calcPlanetCompleteness(quadrature_data, bandzip, photdict, contrfile="CGPERF_HLC_20190210b.csv")
        comps.to_pickle(comps_path)
        with open(compdict_path, 'wb') as f:
            pickle.dump(compdict, f)
        comps_data.to_pickle(comps_data_path)
else:
    comps, compdict, comps_data = calcPlanetCompleteness(quadrature_data, bandzip, photdict, contrfile="CGPERF_HLC_20190210b.csv")

#aliases
aliases_path = Path(f'cache/aliases_{datestr}.p')
if cache:
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if aliases_path.exists():
        with open(aliases_path, 'rb') as f:
            aliases = pickle.load(f)
    else:
        aliases = genAliases(comps_data)
        with open(aliases_path, 'wb') as f:
            pickle.dump(aliases_path, f)
else:
    aliases = genAliases(comps_data)


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
