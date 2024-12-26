import pandas as pd
import time
from sqlalchemy import create_engine
from tqdm import tqdm
from plandb_methods import *
from update_util import *
import logging

from database_main import compileContrastCurves
from database_main import compileCompleteness

from update_plandb_method import *



# build database from new values
# database_main/build db

"""

Entire database benchmarking and building script.
Broken down into the following steps.
a) Complete database compile, using today's information.
b) Build old database using 2022 information.
c) update old database to 2024.

"""

logging.basicConfig(filename="timer_log.txt", level=logging.INFO, format = "%(asctime)s - %(message)s")

cache = False
datestr = Time.now().datetime.strftime("%Y-%m")


######### Complete Database build ############################################################################################################################

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


# build database from old values and update

# build old (use plandb.com/cache files from 2022)

######### Old Database build (2022) ############################################################################################################################


start_time_2022 = time.time()
logging.info("2022 database build start")

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

# TODO start time

######### Update Database ############################################################################################################################

start_time_update = time.time()
logging.info("update database start")

updateDatabaseTest("andrewchiu", "Password123!", "database2022before", 'databaseDiffUpdate', "databaseAfterUpdate")


# update

#TODO end time
end_time_update = time.time()
elapsed_time_update = end_time_update - start_time_update
logging.info(f"end update database elapsed time: {elapsed_time_update}")

# Compare
elapsed_time_difference = elapsed_time_update - elapsed_time
elapsed_time_percent_change = elapsed_time_difference / elapsed_time
logging.info(f"time difference between update and build {elapsed_time_difference} difference of {elapsed_time_percent_change}")

######### Compare Database Test #################################################################


def get_all_tables_as_dataframes(connection):
    cursor = connection.cursor() 
    
    cursor.execute("SHOW TABLES;")
    tables = cursor.fetchall()  
    
    table_names = [table[0] for table in tables]
    
    dataframes = {}
    for table_name in table_names:
        query = f"SELECT * FROM {table_name};" 
        df = pd.read_sql(query, connection) 
        dataframes[table_name] = df
    
    cursor.close()
    return dataframes
  
def compare_all_tables(dataframes1, dataframes2):
    comparison_results = {}

    tables1 = set(dataframes1.keys())
    tables2 = set(dataframes2.keys())

    if tables1 != tables2:
        comparison_results["missing_or_extra_tables"] = {
            "in_dataframes1_only": tables1 - tables2,
            "in_dataframes2_only": tables2 - tables1,
        }
        return comparison_results

    for table_name in tables1:
        df1 = dataframes1[table_name]
        df2 = dataframes2[table_name]

        if set(df1.columns) != set(df2.columns):
            comparison_results[table_name] = {
                "status": "mismatch",
                "reason": "Column names don't match",
                "df1_columns": list(df1.columns),
                "df2_columns": list(df2.columns),
            }
            continue

        df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
        df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

        if df1_sorted.equals(df2_sorted):
            comparison_results[table_name] = {"status": "match"}
        else:
            comparison_results[table_name] = {
                "status": "mismatch",
                "reason": "Row contents different",
            }

    return comparison_results
  
with engineToday.connect() as connection2022:
  
  today_dataframes = get_all_tables_as_dataframes(connection2022)
  
  engine_after_update = create_engine('mysql+pymysql://'+"andrewchiu"+':'+"Password123!"+'@localhost/' + "databaseAfterUpdate",echo=True)


  with engine_after_update.connect() as connectionAfter:
    
    after_update_dataframes = get_all_tables_as_dataframes(connectionAfter)
    
    comparison_result = compare_all_tables(today_dataframes, after_update_dataframes)
    
    for table, result in comparison_result.items():
      print(f"Table: {table}, Result: {result}")