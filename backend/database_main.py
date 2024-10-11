
#move to new file
import pandas as pd

from sqlalchemy import create_engine
from tqdm import tqdm
from plandb_methods import *
from update_util import *

def compileContrastCurves(stars, cont_path):
    scenarios = pd.read_csv('plandb.sioslab.com/cache/scenario_angles.csv')['scenario_name']

    cols = ['scenario_name', 'st_id', 'r_lamD', 'r_as', 'r_mas', 'contrast', 'dMag', 'lam', 't_int_hr', 'fpp']
    contrastCurves = pd.DataFrame([], columns = cols)
    sts_missed = []
    cont_path.mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(cont_path):
        st_sen = filename[:-2]
        st_name = ""
        scen = ""
        for sen_name in scenarios:
            if sen_name in st_sen:
                st_name = st_sen[:-(1+len(sen_name))].replace("_", " ")
                scen = st_sen[-(len(sen_name)):]

        indCC = pd.read_pickle(Path(cont_path, filename))

        if len( list(stars.index[(stars['st_name'] == st_name)])) == 0:
            if st_name not in sts_missed:
                print(st_name)
                sts_missed.append(st_name)
        else:
            indCC["scenario_name"] = scen
            indCC["st_id"] = list(stars.index[(stars['st_name'] == st_name)])[0]
            contrastCurves = contrastCurves._append(indCC, ignore_index=True)
    return contrastCurves

def compileCompleteness():
    datestr = Time.now().datetime.strftime("%Y-%m")
    comp_data = pd.read_pickle(f"plandb.sioslab.com/cache/comps_data_{datestr}.p")
    col_names = comp_data.columns.values.tolist()
    scenario_names = []
    for x in col_names:
        if x[:8] == 'complete':
            scenario_names.append(x[13:])
    #drop contr_curve col
    completeness = pd.DataFrame([], columns = ['pl_id', 'completeness',  'scenario_name', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag'])
    for i, row in tqdm(comp_data.iterrows()):
        newRows = []
        for scenario_name in scenario_names:
            newRows = []
            if pd.notna(row[('completeness_' + scenario_name)]):
                newRows.append([row['pl_id'], row['completeness_' + scenario_name], scenario_name,
                    row['compMinWA_' + scenario_name], 
                    row['compMaxWA_' + scenario_name],
                    row['compMindMag_' + scenario_name],
                    row['compMaxdMag_' + scenario_name]])

                singleRow = pd.DataFrame(newRows, columns =  ['pl_id', 'completeness',  'scenario_name', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag'])
                completeness = completeness._append(singleRow, ignore_index=True)
    # print(completeness)
    return completeness

if __name__ == "__main__":
    datestr = Time.now().datetime.strftime("2022-05")
    plandata_path = Path(f'plandb.sioslab.com/cache/plandata_{datestr}.p')
    planets = pd.read_pickle(plandata_path)

    stdata_path = Path(f'plandb.sioslab.com/cache/stdata_{datestr}.p')
    stars = pd.read_pickle(stdata_path)

    orbfits_path = Path(f'plandb.sioslab.com/cache/table_orbitfits_{datestr}.p')
    orbitfits = pd.read_pickle(orbfits_path)

    orbdata_path = Path(f'plandb.sioslab.com/cache/ephemeris_orbdata_{datestr}.p')
    orbits = pd.read_pickle(orbdata_path)

    comps_path = Path(f'plandb.sioslab.com/cache/comps_{datestr}.p')
    pdfs = pd.read_pickle(comps_path)
    print(pdfs)

    compiled_contr_curvs_path = Path(f'plandb.sioslab.com/cache/compiled_cont_curvs_{datestr}.p')
    contr_curvs_path = Path(f'plandb.sioslab.com/cache/cont_curvs_{datestr.replace("-", "_")}')

    if compiled_contr_curvs_path.exists():
        contrast_curves = pd.read_pickle(compiled_contr_curvs_path)
    else:
        contrast_curves = compileContrastCurves(stars, contr_curvs_path)
        contrast_curves.to_pickle(compiled_contr_curvs_path)
    def addId(r):
        r['pl_id']= list(planets.index[(planets['pl_name'] == r['Name'])])[0]
        return r
    newpdfs = pdfs.apply(addId, axis = 1)

    scenarios = pd.read_csv("plandb.sioslab.com/cache/scenario_angles.csv")
    
    compiled_completeness_path = Path(f"plandb.sioslab.com/cache/compiled_completeness_{datestr}.p")
    if compiled_completeness_path.exists():
        completeness = pd.read_pickle(compiled_completeness_path)
    else:
        completeness = compileCompleteness()
        completeness.to_pickle(compiled_completeness_path)

    # passwd = input("db password: ")
    # username = 'plandb_admin'

    engine = create_engine('mysql+pymysql://'+"andrewchiu"+':'+"Password123!"+'@localhost/testSios',echo=True)
        
    #pool_pre_ping=True for remote
    planets.to_excel("main_planets.xlsx")
    stars.to_excel("main_stars.xlsx")
    # orbitfits.to_excel("main_orbitfits")
    # orbits.to_excel('main_orbits.xlsx')
    contrast_curves.to_excel('main_contrast.xlsx')
    completeness.to_excel('main_completeness.xlsx')
    
    
    final_writeSQL(engine, plandata=planets, stdata=stars, orbitfits=orbitfits, orbdata=orbits, pdfs=newpdfs, aliases=None,contrastCurves=contrast_curves,scenarios=scenarios, completeness=completeness)
