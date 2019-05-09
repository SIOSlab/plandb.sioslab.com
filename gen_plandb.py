from __future__ import print_function
from __future__ import division
from plandb_methods import *
import getpass,keyring
import sqlalchemy

datestr = Time.now().datetime.strftime("%Y-%m-%d")
cache = False

#initial data dump
data = getIPACdata()
# data.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\data_save.csv")
# quit()

#photometric data
photdict = loadPhotometryData()

#band info
bandzip = genBands()

orbdata, orbitfits = genOrbitData_2(data, bandzip, photdict)

#calculate quadrature columns:
orbitfits = calcQuadratureVals(orbitfits, bandzip, photdict)
# if cache: data.to_pickle('data_'+datestr+'.pkl')

# data.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\data_save.csv")
#calculate quadrature columns:
# data = calcQuadratureVals(data, bandzip, photdict)
# data.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\data.csv")


comps,compdict,orbitfits = calcPlanetCompleteness(orbitfits, bandzip, photdict)

# orbdata.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\orbdata_new.csv")


#get orbital data
# orbdata = genOrbitData(data,bandzip,photdict)
# if cache: orbdata.to_pickle('orbdata_'+datestr+'.pkl')
#
# altorbdata =genAltOrbitData(data, bandzip, photdict)
# if cache: altorbdata.to_pickle('altorbdata_'+datestr+'.pkl')

#
#completeness
# comps,compdict,data = calcPlanetCompleteness(data, bandzip, photdict)
# if cache:
#     comps.to_pickle('completeness_'+datestr+'.pkl')
#     np.savez('completeness_'+datestr,**compdict)
#     data.to_pickle('data_'+datestr+'.pkl')
# comps.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\comps.csv")
# data.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\plan_orbdata_comps.csv")
#
#
# #aliases
# aliases = genAliases(data)
# if cache: aliases.to_pickle('aliases_'+datestr+'.pkl')
#
#
# #testdb
# engine = create_engine('mysql+pymysql://ds264@127.0.0.1/dsavrans_plandb',echo=False)

# orbitfits.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\orbitfits_presplit.csv")

####
# Integration times go here
####
plandata, stdata, orbitfits = generateTables(data, orbitfits)
# aliases = genAllAliases(stdata)
# orbitfits.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\orbitfits.csv")
# plandata.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\plandata.csv")
# stdata.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\stdata.csv")

comps = add_pl_idx(comps, 'Name', pl_data=plandata,orbitfits=orbitfits)

# orbdata.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\orbdata.csv")
# orbitfits.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\orbitfits.csv")
# plandata.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\plandata.csv")
# stdata.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\stdata.csv")
# comps.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\data\\comps.csv")

engine = sqlalchemy.create_engine('mysql+pymysql://nrk42@127.0.0.1/plandb_scratch')

# #proddb#################################################################################################
# username = 'dsavrans_admin'
# passwd = keyring.get_password('plandb_sql_login', username)
# if passwd is None:
#     passwd = getpass.getpass("Password for mysql user %s:\n"%username)
#     keyring.set_password('plandb_sql_login', username, passwd)
#
# engine = create_engine('mysql+pymysql://'+username+':'+passwd+'@sioslab.com/dsavrans_plandb',echo=False)
# #proddb#################################################################################################
#
writeSQL2(engine, stdata=stdata)
writeSQL2(engine,data=plandata)
writeSQL2(engine, orbitfits=orbitfits, orbdata=orbdata, comps=comps) #, comps=comps, aliases=aliases

# writeSQL(engine,data=data)
# writeSQL(engine,orbdata=orbdata,altorbdata=altorbdata,comps=comps)