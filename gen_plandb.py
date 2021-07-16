from __future__ import division, print_function

import getpass

import keyring

from plandb_methods import *
import pandas as pd

cache = True
from sqlalchemy.types import String

datestr = Time.now().datetime.strftime("%Y-%m-%d")

#initial data dump
data = getIPACdata()

# #photometric data
photdict = loadPhotometryData()

# #band info
bandzip = list(genBands())

# #calculate quadrature columns:
# print('Computing quadrature values')
# data = calcQuadratureVals(data, bandzip, photdict)
# if cache: data.to_pickle('data_'+datestr+'.pkl')

# # #get orbital data
# print("Getting orbit data")
# orbdata = genOrbitData(data,bandzip,photdict)
# if cache: orbdata.to_pickle('orbdata_'+datestr+'.pkl')

# print('Getting alt orbit data')
# altorbdata =genAltOrbitData(data, bandzip, photdict)
# if cache: altorbdata.to_pickle('altorbdata_'+datestr+'.pkl')


#completeness
print('Doing completeness calculations')
comps,compdict,data = calcPlanetCompleteness(data, bandzip, photdict)
if cache:
    comps.to_pickle('completeness_'+datestr+'.pkl')
    np.savez('completeness_'+datestr,**compdict)
    data.to_pickle('data_'+datestr+'.pkl')


#aliases
aliases = genAliases(data)
if cache: aliases.to_pickle('aliases_'+datestr+'.pkl')


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
