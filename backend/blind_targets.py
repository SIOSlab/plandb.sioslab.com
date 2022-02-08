import cPickle as pickle
import pandas
import numpy as np
import astropy.units as u
from sqlalchemy import create_engine
import getpass,keyring
import sqlalchemy.types 
import astropy.io
import re
import scipy.interpolate
import astropy.constants as const

tmp = pandas.read_csv('blindsortedStars.txt')
names = tmp['Name'].values 


aliases = pandas.read_pickle('aliases_080718.pkl')

s1 = set(aliases['Alias'].values)
s2 = set(names) 

knownplan = list(s1.intersection(s2))
goodtogo = list(s2 - s1)

with open('/Users/ds264/Downloads/forDaniel.txt', 'w') as f:
    for item in goodtogo:
        f.write("%s\n" % item)


knownplan2 = [ aliases['Alias'][aliases['SID'] == aliases['SID'][aliases['Alias'] == n].values[0]].values[np.where(aliases['NEAName'][aliases['SID'] == aliases['SID'][aliases['Alias'] == n].values[0]].values)[0]][0] for n in knownplan]


for p in knownplan:
    tmp = tmp[tmp['Name'] != p]
tmp = tmp.reset_index(drop=True)
tmp = tmp.rename(columns={'Count':'Priority'})
tmp['Priority'] = tmp['Priority'].values/149000.


#grab the catalog and identify the star indices
from EXOSIMS.StarCatalog.EXOCAT1 import EXOCAT1
cat = EXOCAT1()

names = tmp['Name'].values
priorities = tmp['Priority'].values
catinds = np.hstack([np.where(cat.Name == n)[0] for n in names])

#grab the Mamajek table to fill in radii
mamajektable = astropy.io.ascii.read('EEM_dwarf_UBVIJHK_colors_Teff.txt',fill_values=[('...',np.nan),('....',np.nan),('.....',np.nan)])

specregex = re.compile('([OBAFGKMLTY])(\d*\.\d+|\d+)V')
specregex2 = re.compile('([OBAFGKMLTY])(\d*\.\d+|\d+).*')

#grab spectral types from table
MK = []
MKn = []
for s in mamajektable['SpT'].data:
    m = specregex.match(s)
    MK.append(m.groups()[0])
    MKn.append(m.groups()[1])
MK = np.array(MK)
MKn = np.array(MKn)

#create interpolants for Radius
Ri = {}
for l in 'OBAFGKM':
    Ri[l] = scipy.interpolate.interp1d(MKn[MK==l].astype(float),mamajektable['R_Rsun'][MK==l].data.astype(float),bounds_error=False,fill_value='extrapolate')

#compute stellar radii
specs = cat.Spec[catinds]
specs[specs == 'APSREU(CR)'] = 'A7VpSrCrEu'
radii = []
for s in specs:
    m = specregex2.match(s)
    if m:
        radii.append(Ri[m.groups()[0]](m.groups()[1]))
    else:
        radii.append(np.nan)
        print(s)
radii = np.array(radii)

dists = cat.dist[catinds]
angrad = np.arctan((radii*const.R_sun)/dists).to('mas')


blind_targs = pandas.DataFrame({'Name':names,
                                'Priority':priorities,
                                'dist':dists.to(u.pc).value,
                                'spec':specs,
                                'Vmag':cat.Vmag[catinds],
                                'L':cat.L[catinds],
                                'RA':cat.coords[catinds].ra.to(u.deg).value,
                                'DEC':cat.coords[catinds].dec.to(u.deg).value,
                                'radius':radii,
                                'angrad':angrad.value})


##############
with open('DoS_10_2018_v2/DoS.res', 'rb') as f:
    dosdat = pickle.load(f)


abins = dosdat['aedges']
Rbins = dosdat['Rpedges']


ac,Rc = np.meshgrid(abins[:-1]+np.diff(abins)/2.0,Rbins[:-1]+np.diff(Rbins)/2.0)

ainds = np.arange(abins.size-1)
Rinds = np.arange(Rbins.size-1)
ainds,Rinds = np.meshgrid(ainds,Rinds)

adiffs = np.array(np.diff(abins),ndmin=2)
Rdiffs = np.array(np.diff(Rbins),ndmin=2)
binareas = Rdiffs.transpose()*adiffs

Rnepind = np.where(Rc[:,0] < 3.883)[0].max()

tnames = []
acs = []
Rcs = []
iinds = []
jinds = []
DoS = []
summedDoS = []

for name in names:
    vals = dosdat['DoS'][name]
    tnames.append(np.array([name]*vals.size))
    acs.append(ac.flatten())
    Rcs.append(Rc.flatten())
    DoS.append(vals.flatten())
    iinds.append(ainds.flatten())
    jinds.append(Rinds.flatten())
    summedDoS.append(np.sum(vals[:Rnepind+1,:]))

DoStable = pandas.DataFrame({'Name': np.hstack(tnames),
                            'sma': np.hstack(acs),
                            'Rp': np.hstack(Rcs),
                            'vals': np.hstack(DoS),
                            'iind': np.hstack(iinds),
                            'jind': np.hstack(jinds)
                            })
DoStable = DoStable[DoStable['vals'].values != 0.]
DoStable['vals'] = np.log10(DoStable['vals'].values)

blind_targs['sumDoS'] = np.array(summedDoS)

#------write to db------------
namemxchar = np.array([len(n) for n in names]).max()

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


##write blind targets
blind_targs.to_sql('BlindTargs',engine,chunksize=100,if_exists='replace',
        dtype={'Name':sqlalchemy.types.String(namemxchar)})
result = engine.execute("ALTER TABLE BlindTargs ENGINE=InnoDB")
result = engine.execute("ALTER TABLE BlindTargs ADD INDEX (Name)")
  

#write DoS
DoStable.to_sql('DoS',engine,chunksize=100,if_exists='replace',dtype={'Name':sqlalchemy.types.String(namemxchar)})
result = engine.execute("ALTER TABLE DoS ENGINE=InnoDB")
result = engine.execute("ALTER TABLE DoS ADD INDEX (Name)")
#result = engine.execute("ALTER TABLE Completeness ADD FOREIGN KEY (Name) REFERENCES KnownPlanets(pl_name) ON DELETE NO ACTION ON UPDATE NO ACTION");

#####expand aliases#########
from astroquery.simbad import Simbad
s = Simbad()
s.add_votable_fields('ids')
stind = engine.execute("select max(SID) from Aliases").fetchone()[0] + 1



ids = []
aliases = []
nosimbadalias = []
priname = []
for j,star in enumerate(names):
    print(j,star)
    #get aliases from SIMBAD
    r = s.query_object(star)
    if r:
        tmp = r['IDS'][0].split('|')
    else:
        nosimbadalias.append(star)
        continue

    if star not in tmp: tmp.append(star)
    if 'aliasdis' in tmp: tmp.remove('aliasdis')
    tmp = list(np.unique(tmp))
        
    ids.append([j+stind]*len(tmp))
    aliases.append(tmp)
    priname.append(list((np.array(tmp) == star).astype(int)))


newaliases = pandas.DataFrame({'SID': np.hstack(ids),
                         'Alias': np.hstack(aliases),
                         'NEAName':np.hstack(priname)
                         })


newaliases.to_sql('Aliases',engine,chunksize=100,if_exists='append')

