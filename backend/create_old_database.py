import pandas as pd
from sqlalchemy import create_engine
import pymysql
import glob
from sqlalchemy import text
from plandb_methods import *
from database_main import *
from update_util import *
import os
import sys 
import logging

from database_main import compileContrastCurves
from database_main import compileCompleteness

from update_plandb_method import *


"""
Create old database, 2022 information, to be updated.
For benchmarking purposes.
Currently uses MySQL database: (database2022before)
"""


logging.basicConfig(filename="timer_log.txt", level=logging.INFO, format = "%(asctime)s - %(message)s")

cache = False
datestr = Time.now().datetime.strftime("%Y-%m")


start_time_complete = time.time()
logging.info("Complete database build start")

data = getIPACdata()



#photometric data
photdict = loadPhotometryData(infile="plandb.sioslab.com/allphotdata_2015.npz")

#band info
bandzip = list(genBands())

# TODO: look at genOrbitData method signature, with t0 default, and consider adding t0
orbdata, orbitfits = genOrbitData(data, bandzip, photdict)

ephemeris_orbitfits, ephemeris_orbdata = addEphemeris(data, orbitfits, orbdata, bandzip, photdict)

quadrature_data = calcQuadratureVals(ephemeris_orbitfits, bandzip, photdict)

exosims_json = 'plandb.sioslab.com/ci_perf_exosims.json'
contr_data = calcContrastCurves(quadrature_data, exosims_json=exosims_json)

comps, compdict, comps_data = calcPlanetCompleteness(contr_data, bandzip, photdict, exosims_json=exosims_json)

plandata, stdata, orbitfits = generateTables(data, comps_data)

# TODO: Prob replace this path stuff, prob HAVE to plandb.com/cache contr curvecalculations and then compile, or maybe write new compile function
# TODO: calculate contrast curves and test it here
contr_curvs_path = Path(f'plandb.sioslab.com/cache/cont_curvs_{datestr.replace("-", "_")}')
contrast_curves = compileContrastCurves(stdata, contr_curvs_path)

    

# newpdfs = pdfs.apply(addId, axis = 1)
scenarios = pd.read_csv("plandb.sioslab.com/cache/scenario_angles.csv")

# compiled_completeness_path = Path(f"plandb.com/cache/compiled_completeness_{datestr}.p")
# if compiled_completeness_path.exists():
#     completeness = pd.read_pickle(compiled_completeness_path)
# else:
#     completeness = compileCompleteness()
#     completeness.to_pickle(compiled_completeness_path)


# passwd = input("db password: ")
# username = 'plandb_admin'
engineToday = create_engine('mysql+pymysql://'+"andrewchiu"+':'+"Password123!"+'@localhost/TestToday',echo=True)

#TODO: completeness later

writeSQL(engineToday, plandata=plandata, stdata=stdata, orbitfits=orbitfits, orbdata=orbdata, pdfs=None, aliases=None,contrastCurves=None,scenarios=None, completeness=None)

# TODO end time
end_time_complete = time.time()
elapsed_time = end_time_complete - start_time_complete
logging.info(f"complete database build end. Elapsed time: {elapsed_time:.2f} seconds")

print(f"elapsed time: {elapsed_time}" )

