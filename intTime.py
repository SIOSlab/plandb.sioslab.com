import os
os.chdir('git/EXOSIMS/')
import EXOSIMS.MissionSim
import EXOSIMS.OpticalSystem.Nemati_2019 as Nemati_2019
os.chdir('../..')
import json
from astropy import units as u
import pandas as pd
import numpy as np
import time


# Sdata is orbitfits presplit, data is orbdata
def calcIntTimes(sdata, data):
    np.seterr(divide='ignore', invalid='ignore')

    scriptfile = 'intTime_script.json'
    script = open(scriptfile).read()

    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

    specs = json.loads(script)

    print('Calculating Integration Times:')
    print('-----------------------------------------------------------------')
    print('Progress  Target   Name        Inclination  Calc_time  Total_time')
    print('-----------------------------------------------------------------')

    TL = sim.TargetList
    sInds = [0]
    mode = sim.OpticalSystem.observingModes[0]
    fZ = sim.ZodiacalLight.fZ0
    fEZ = sim.ZodiacalLight.fEZ0
    TK = sim.TimeKeeping

    N = Nemati_2019.Nemati_2019(**specs)

    TL.nStars = 1

    # sdata = pd.read_csv('orbitfits_presplit.csv')
    # data = pd.read_csv('orbdata.csv')

    ids = list(data.orbitfit_id)
    iddict = {}
    for i in range(len(ids)):
        iid = ids[i]
        if iid not in iddict.keys():
            iddict[iid] = [i]
        else:
            iddict[iid].append(i)

    dMags = [i for i in list(data.columns.values) if i[:4] == 'dMag']

    inttimes = {}
    timer = 0.

    for sid, sind in iddict.items():

        t1 = time.time()

        WA = np.array(data.WA[sind])/1e3*u.arcsec

        sSpec = sdata.st_spstr[sid].replace(' ', '')
        sName = sdata.hd_name[sid]
        sMag = sdata.st_vj[sid]
        pInc = sdata.pl_orbincl[sid]

        TL.Spec = [sSpec]
        TL.Name = [sName]
        TL.Vmag = np.array([sMag])

        for i in dMags:

            dMag = np.array(data[i][sind])

            mode['lam'] = float(i[10:13])*u.nm
            intName = 'intTime' + i[4:]

            times = Nemati_2019.Nemati_2019.calc_intTime(N, TL, sInds, fZ, fEZ, dMag, WA, mode, TK)

            if sid == 0:
                inttimes[intName] = times
            else:
                inttimes[intName] = np.append(inttimes[intName], times)

        t2 = time.time()
        timer += t2-t1

        print('{a:7.2f}%  {b:7s}  {c:9s}  {d:12.2f}  {e:9.2f}  {f:10.2f}'.format(a=(sid+1)/len(iddict)*100., b=str((sid+1))+'/'+str(len(iddict)), c=str(sName)[:9], d=pInc, e=t2-t1, f=timer))

    for tid, times in inttimes.items():

        data[tid] = times

    return data
    # data.to_csv('orbdata_test.csv', index=False)