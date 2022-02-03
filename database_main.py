
#move to new file
import pandas as pd

from sqlalchemy import create_engine
from tqdm import tqdm
from plandb_methods import *

def compileContrastCurves(stars, cont_path):
    # stars = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/stdata_2021-11-29.p")
    scenarios = [
        'Conservative_NF_Imager_25hr',
        'Conservative_NF_Imager_100hr',
        'Conservative_NF_Imager_10000hr',
        'Optimistic_NF_Imager_25hr',
        'Optimistic_NF_Imager_100hr',
        'Optimistic_NF_Imager_10000hr',
        'Conservative_Amici_Spec_100hr',
        'Conservative_Amici_Spec_400hr',
        'Conservative_Amici_Spec_10000hr',
        'Optimistic_Amici_Spec_100hr',
        'Optimistic_Amici_Spec_400hr',
        'Optimistic_Amici_Spec_10000hr',
        'Conservative_WF_Imager_25hr',
        'Conservative_WF_Imager_100hr',
        'Conservative_WF_Imager_10000hr',
        'Optimistic_WF_Imager_25hr',
        'Optimistic_WF_Imager_100hr',
        'Optimistic_WF_Imager_10000hr']

    cols = ['scenario_name', 'st_id', 'r_lamD', 'r_as', 'r_mas', 'contrast', 'dMag', 'lam', 't_int_hr', 'fpp']
    contrastCurves = pd.DataFrame([], columns = cols)
    sts_missed = []
    cont_path.mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(cont_path):
        st_sen = filename[:-2]
        # print(st_sen)
        st_name = ""
        scen = ""
        for sen_name in scenarios:
            if sen_name in st_sen:
                st_name = st_sen[:-(1+len(sen_name))].replace("_", " ")
                scen = st_sen[-(len(sen_name)):]

        # print("Star: ", st_name)
        # print("Scenario: ", scen)
        indCC = pd.read_pickle(Path(cont_path, filename))
        indCC['dMag'] = np.log10([x[0] for x in indCC['contrast']])*-2.5

        if len( list(stars.index[(stars['st_name'] == st_name)])) == 0:
            if st_name not in sts_missed:
                print(st_name)
                sts_missed.append(st_name)
        else:
            indCC["scenario_name"] = scen
            indCC["st_id"] = list(stars.index[(stars['st_name'] == st_name)])[0]
            # print(indCC)
            contrastCurves = contrastCurves.append(indCC, ignore_index=True)
    return contrastCurves

def compileCompleteness():
    datestr = Time.now().datetime.strftime("%Y-%m")
    comp_data = pd.read_pickle(f"cache/comps_data_{datestr}.p")
    col_names = comp_data.columns.values.tolist()
    # print(str(list(comp_data)).replace(", ", "\n"))
    scenario_names = []
    for x in col_names:
        if x[:8] == 'complete':
            scenario_names.append(x[13:])
    #drop contr_curve col
    completeness = pd.DataFrame([], columns = ['pl_id', 'completeness',  'scenario_name', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag'])
    for i, row in tqdm(comp_data.iterrows()):
        # print(type(row))
        newRows = []
        for scenario_name in scenario_names:
            newRows = []
            if pd.notna(row[('completeness_' + scenario_name)]):
                newRows.append([row['pl_id'], row['completeness_' + scenario_name], scenario_name,
                    row['compMinWA_' + scenario_name], 
                    row['compMaxWA_' + scenario_name],
                    row['compMindMag_' + scenario_name],
                    row['compMaxdMag_' + scenario_name]])

                    # row['compMinWA_{scenario_name}'],
                    # row['compMaxWA_{scenario_name}'],
                    # row['compMindMag_{scenario_name}'],
                    # row['compMaxdMag_{scenario_name}']])

                singleRow = pd.DataFrame(newRows, columns =  ['pl_id', 'completeness',  'scenario_name', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag'])
                completeness = completeness.append(singleRow, ignore_index=True)
                print(row['pl_name'])
    # print(completeness)
    return completeness

if __name__ == "__main__":
    # completeness = compileCompleteness()
    datestr = Time.now().datetime.strftime("%Y-%m")
    plandata_path = Path(f'cache/plandata_{datestr}.p')
    planets = pd.read_pickle(plandata_path)

    stdata_path = Path(f'cache/stdata_{datestr}.p')
    stars = pd.read_pickle(stdata_path)

    orbfits_path = Path(f'cache/orbfits_{datestr}.p')
    orbitfits = pd.read_pickle(orbfits_path)

    orbdata_path = Path(f'cache/orbdata_{datestr}.p')
    orbits = pd.read_pickle(orbdata_path)

    comps_path = Path(f'cache/comps_{datestr}.p')
    pdfs = pd.read_pickle(comps_path)

    compiled_contr_curvs_path = Path(f'cache/compiled_cont_curvs_{datestr}.p')
    contr_curvs_path = Path(f'cache/cont_curvs_{datestr.replace("-", "_")}')

    if compiled_contr_curvs_path.exists():
        contrast_curves = pd.read_pickle(compiled_contr_curvs_path)
    else:
        contrast_curves = compileContrastCurves(stars, contr_curvs_path)
        contrast_curves.to_pickle(compiled_contr_curvs_path)
    # print(list(pdfs))
    # # # df.loc[df['column_name'] == some_value]
    # # print(list(planets.index[(planets['pl_name'] == "47 UMa d")])[0])
    def addId(r):
        r['pl_id']= list(planets.index[(planets['pl_name'] == r['Name'])])[0]
        return r
    newpdfs = pdfs.apply(addId, axis = 1)
    # # pdfs = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/indexed_pdfs_2021-12-10.p")
    # # print(pdfs)

    scenarios = pd.read_csv("cache/scenario_angles.csv")
    # # print(scenarios)
    # # contrastCurves = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/cont_curvs_2021_11_compiled.p")
    # # print(contrastCurves)
    compiled_completeness_path = Path(f"cache/compiled_completeness_{datestr}.p")
    if compiled_completeness_path.exists():
        completeness = pd.read_pickle(compiled_completeness_path)
    else:
        completeness = compileCompleteness()
        completeness.to_pickle(compiled_completeness_path)

    passwd = input("db password: ")
    username = 'plandb_admin'

    engine = create_engine('mysql+pymysql://'+username+':'+passwd+'@localhost/plandb_scratch2',echo=False)
    writeSQL(engine, plandata=planets, stdata=stars, orbitfits=orbitfits, orbdata=orbits, pdfs=newpdfs, aliases=None,contrastCurves=contrast_curves,scenarios=scenarios, completeness=completeness)
