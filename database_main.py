
#move to new file
import pandas as pd

from sqlalchemy import create_engine
from tqdm import tqdm
from plandb_methods import *

def compileContrastCurves():
    stars = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/stdata_2021-11-29.p") 
    scenarios = [
        'EB_NF_Imager_25hr',
        'EB_NF_Imager_100hr',
        'EB_NF_Imager_10000hr',
        'DRM_NF_Imager_25hr',
        'DRM_NF_Imager_100hr',
        'DRM_NF_Imager_10000hr',
        'EB_Amici_Spec_100hr',
        'EB_Amici_Spec_400hr',
        'EB_Amici_Spec_10000hr',
        'DRM_Amici_Spec_100hr',
        'DRM_Amici_Spec_400hr',
        'DRM_Amici_Spec_10000hr',
        'EB_WF_Imager_25hr',
        'EB_WF_Imager_100hr',
        'EB_WF_Imager_10000hr',
        'DRM_WF_Imager_25hr',
        'DRM_WF_Imager_100hr',
        'DRM_WF_Imager_10000hr']
    
    cols = ['scenario_name', 'st_id', 'r_lamD', 'r_as', 'r_mas', 'contrast', 'lam', 't_int_hr', 'fpp']
    contrastCurves = pd.DataFrame([], columns = cols)
    sts_missed = []
    for filename in os.listdir("/data/plandb/plandb.sioslab.com/cache/cont_curvs_2021_11/"):
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
        indCC = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/cont_curvs_2021_11/" + filename)

        if len( list(stars.index[(stars['st_name'] == st_name)])) == 0:
            if st_name not in sts_missed:
                print(st_name)
                sts_missed.append(st_name)
        else: 
            indCC["scenario_name"] = scen
            indCC["st_id"] = list(stars.index[(stars['st_name'] == st_name)])[0]
            # print(indCC)
            contrastCurves = contrastCurves.append(indCC, ignore_index=True)
        return constrastCurves


if __name__ == "__main__":
    stars = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/stdata_2021-11-29.p")
    # print(stars)
    # # print(stars
    # planets = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/plandata_2021-11-29.p")
    # print('st_id' in list(planets))
    # print((planets))
    orbitfits = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/orbfits_2021-11-29.p")
    print(orbitfits['pl_radj'])

    # # orbits = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/orbdata_2021-11-29.p")
    # # pdfs = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/comps_2021-11-29.p")
    # # print((pdfs))
    # # # df.loc[df['column_name'] == some_value]
    # # print(list(planets.index[(planets['pl_name'] == "47 UMa d")])[0])
    # # def addId(r):
    # #     r['pl_id']= list(planets.index[(planets['pl_name'] == r['Name'])])[0]
    # #     return r
    # # newpdfs = pdfs.apply(addId, axis = 1)
    # # pdfs = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/indexed_pdfs_2021-12-10.p")
    # # print(pdfs)
    # comp_data = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/comps_data_2021-11-29.p")
    # col_names = comp_data.columns.values.tolist()
    # print(str(list(comp_data)).replace(", ", "\n"))
    # scenario_names = []
    # for x in col_names:
    #     if x[:8] == 'complete':
    #         scenario_names.append(x[13:])

    # # print(scenario_names)
    # # a = 0
    # # b = 1/a
    # #drop contr_curve col
    # completeness = pd.DataFrame([], columns = ['pl_id', 'completeness',  'scenario_name', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag'])
    # for i, row in tqdm(comp_data.iterrows()):
    #     # print(type(row))
    #     newRows = []
        
    #     # if pd.notna(row[('compMinWA_{scenario_name}')]):
    #     #     print(row['compMinWA_{scenario_name}'])
    #     #     print(row['compMaxWA_{scenario_name}'])
    #     #     print(row['compMindMag_{scenario_name}'])
    #     #     print(row['compMaxdMag_{scenario_name}'])
    #     #     print(row['compMaxdMag_{scenario_name}'])
    #     #     for scenario_name in scenario_names:
    #     #         print(row['completeness_' + scenario_name])
    #     for scenario_name in scenario_names:
    #         newRows = []
    #         if pd.notna(row[('completeness_' + scenario_name)]):
    #             newRows.append([row['pl_id'], row['completeness_' + scenario_name], scenario_name, 
    #                 # row['compMinWA_' + scenario_name], 
    #                 # row['compMaxWA_' + scenario_name],
    #                 # row['compMindMag_' + scenario_name],
    #                 # row['compMaxdMag_' + scenario_name]])
                    
    #                 row['compMinWA_{scenario_name}'], 
    #                 row['compMaxWA_{scenario_name}'],
    #                 row['compMindMag_{scenario_name}'],
    #                 row['compMaxdMag_{scenario_name}']])
                    
    #             singleRow = pd.DataFrame(newRows, columns =  ['pl_id', 'completeness',  'scenario_name', 'compMinWA', 'compMaxWA', 'compMindMag', 'compMaxdMag'])
    #             completeness = completeness.append(singleRow, ignore_index=True)   
    # print(completeness)
    # # scenarios = pd.read_csv("scenarios.csv")    
    # # print(scenarios)
    # # contrastCurves = pd.read_pickle("/data/plandb/plandb.sioslab.com/cache/cont_curvs_2021_11_compiled.p")
    # # print(contrastCurves)


    # passwd = input("db password: ")  
    # username = 'plandb_admin'

    # engine = create_engine('mysql+pymysql://'+username+':'+passwd+'@localhost/plandb_scratch',echo=False)
    # writeSQL(engine, data=None, stdata=None, orbitfits=None, orbdata=None, pdfs=None, aliases=None,contrastCurves=None,scenarios=None, completeness=completeness)