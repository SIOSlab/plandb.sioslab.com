from __future__ import print_function
from __future__ import division
from plandb_methods import *
import getpass,keyring


datestr = Time.now().datetime.strftime("%Y-%m-%d")
cache = False

#initial data dump
data = getIPACdata()

#photometric data
photdict = loadPhotometryData()

#band info
bandzip = genBands()

#calculate quadrature columns:
data = calcQuadratureVals(data, bandzip, photdict)
if cache: data.to_pickle('data_'+datestr+'.pkl')

#get orbital data
orbdata = genOrbitData(data,bandzip,photdict)
if cache: orbdata.to_pickle('orbdata_'+datestr+'.pkl')

altorbdata =genAltOrbitData(data, bandzip, photdict)
if cache: altorbdata.to_pickle('altorbdata_'+datestr+'.pkl')


#completeness
comps,compdict,data = calcPlanetCompleteness(data, bandzip, photdict)
if cache: 
    comps.to_pickle('completeness_'+datestr+'.pkl')
    np.savez('completeness_'+datestr,**compdict)
    data.to_pickle('data_'+datestr+'.pkl')


#aliases
aliases = genAliases(data)
if cache: aliases.to_pickle('aliases_'+datestr+'.pkl')


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

writeSQL(engine,data=data)
writeSQL(engine,orbdata=orbdata,altorbdata=altorbdata,comps=comps)
