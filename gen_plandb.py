from plandb_methods import *

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
comps,compdict = calcPlanetCompleteness(data, bandzip, photdict)
if cache: 
    comps.to_pickle('completeness_'+datestr+'.pkl')
    np.savez('completeness_'+datestr,**compdict)
    data.to_pickle('data_'+datestr+'.pkl')


#aliases
aliases = genAliases(data)
if cache: aliases.to_pickle('aliases_'+datestr+'.pkl')

