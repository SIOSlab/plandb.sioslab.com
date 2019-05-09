from __future__ import print_function
from __future__ import division
from plandb_methods import *
###### Integration times
# from intTime import calcIntTimes
######
import getpass,keyring
import sqlalchemy

datestr = Time.now().datetime.strftime("%Y-%m-%d")
cache = False

#initial data dump
data = getIPACdata()

#photometric data
photdict = loadPhotometryData()

#band info
bandzip = genBands()

# calculate OrbitFits and orbit data
orbdata, orbitfits = genOrbitData_2(data, bandzip, photdict)
if cache:
    orbdata.to_pickle('orbdata_'+datestr+'.pkl')
    orbitfits.to_pickle('orbitfits_' + datestr + '.pkl')

#calculate quadrature columns:
orbitfits = calcQuadratureVals(orbitfits, bandzip, photdict)
if cache: orbitfits.to_pickle('orbitfits_'+datestr+'.pkl')

# Calculate completeness
comps,compdict,orbitfits = calcPlanetCompleteness(orbitfits, bandzip, photdict)
if cache:
    comps.to_pickle('comps_'+datestr+'.pkl')
    np.savez('completeness_' + datestr, **compdict)
    orbitfits.to_pickle('orbitfits_'+datestr+'.pkl')

#### integration times (UNTESTED)
# orbdata = calcIntTimes(orbitfits, orbdata)
# if cache: orbdata.to_pickles('orbdata_'+datestr+'.pkl')
####

# Split data into stars, planets, and orbitfits
plandata, stdata, orbitfits = generateTables(data, orbitfits)
if cache:
    plandata.to_pickle('plandata_'+datestr+'.pkl')
    stdata.to_pickle('stdata_' + datestr + '.pkl')
    orbitfits.to_pickle('orbitfits_'+datestr+'.pkl')

aliases = genAllAliases(stdata)
if cache: aliases.to_pickle('aliases_'+datestr+'.pkl')

# Adds pl_id and orbitfit_id to comps
comps = add_pl_idx(comps, 'Name', pl_data=plandata,orbitfits=orbitfits)
if cache: comps.to_pickle('aliases_'+datestr+'.pkl')

# Test db scratch
engine = sqlalchemy.create_engine('mysql+pymysql://nrk42@127.0.0.1/plandb_scratch')

#testdb
# engine = create_engine('mysql+pymysql://ds264@127.0.0.1/dsavrans_plandb',echo=False)
# #proddb#################################################################################################
# username = 'dsavrans_admin'
# passwd = keyring.get_password('plandb_sql_login', username)
# if passwd is None:
#     passwd = getpass.getpass("Password for mysql user %s:\n"%username)
#     keyring.set_password('plandb_sql_login', username, passwd)
#
# engine = create_engine('mysql+pymysql://'+username+':'+passwd+'@sioslab.com/dsavrans_plandb',echo=False)
# #proddb#################################################################################################

writeSQL2(engine, stdata=stdata)
writeSQL2(engine,data=plandata)
writeSQL2(engine, orbitfits=orbitfits, orbdata=orbdata, comps=comps, aliases=aliases)