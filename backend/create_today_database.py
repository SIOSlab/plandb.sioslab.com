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
Original compile of today's database using all current information.
Currently uses MySQL database: (TestToday)
"""


logging.basicConfig(filename="timer_log.txt", level=logging.INFO, format = "%(asctime)s - %(message)s")

cache = False
datestr = Time.now().datetime.strftime("%Y-%m")




start_time_2022 = time.time()
logging.info("2022 database build start")

#This is prob faster because it's pickled

datestr = "2022-05"
plandata_path = Path(f'plandb.sioslab.com/cache/plandata_{datestr}.p')
planetsOld = pd.read_pickle(plandata_path)
stdata_path = Path(f'plandb.sioslab.com/cache/stdata_{datestr}.p')
starsOld = pd.read_pickle(stdata_path)
orbfits_path = Path(f'plandb.sioslab.com/cache/table_orbitfits_{datestr}.p')
orbitfitsOld = pd.read_pickle(orbfits_path)
orbdata_path = Path(f'plandb.sioslab.com/cache/ephemeris_orbdata_{datestr}.p')
orbitsOld = pd.read_pickle(orbdata_path)
comps_path = Path(f'plandb.sioslab.com/cache/comps_{datestr}.p')
pdfs = pd.read_pickle(comps_path)
print(pdfs)
compiled_contr_curvs_path = Path(f'plandb.sioslab.com/cache/compiled_cont_curvs_{datestr}.p')
contr_curvs_path = Path(f'plandb.sioslab.com/cache/cont_curvs_{datestr.replace("-", "_")}')
if compiled_contr_curvs_path.exists():
    contrast_curvesOld = pd.read_pickle(compiled_contr_curvs_path)
else:
    contrast_curvesOld = compileContrastCurves(starsOld, contr_curvs_path)
    contrast_curvesOld.to_pickle(compiled_contr_curvs_path)
def addId(r):
    r['pl_id']= list(planetsOld.index[(planetsOld['pl_name'] == r['Name'])])[0]
    return r
  
#TODO Pdfs
newpdfs = pdfs.apply(addId, axis = 1)
scenarios = pd.read_csv("plandb.sioslab.com/cache/scenario_angles.csv")

before_engine = create_engine('mysql+pymysql://'+"andrewchiu"+':'+"Password123!"+'@localhost/database2022before',echo=True)

writeSQL(before_engine, plandata=planetsOld, stdata=starsOld, orbitfits=orbitfitsOld, orbdata=orbitsOld, pdfs=None, aliases=None,contrastCurves=None,scenarios=None, completeness=None)

print("Done")

end_time_2022 = time.time()
elapsed_time_2022 = end_time_2022 - start_time_2022
logging.info(f"2022 database build complete, elapsed time: {elapsed_time_2022}")

#TODO; compile later

# compiled_completeness_path = Path(f"plandb.com/cache/compiled_completeness_{datestr}.p")
# if compiled_completeness_path.exists():
#     completeness = pd.read_pickle(compiled_completeness_path)
# else:
#     completeness = compileCompleteness()
#     completeness.to_pickle(compiled_completeness_path)
