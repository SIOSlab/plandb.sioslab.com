import pandas as pd
from sqlalchemy import create_engine
import pymysql
from sqlalchemy import text
from plandb_methods import *
from database_main import *

# Not working, don't use this as this merges the unmade tables with the database

columns = ['pl_name', 'pl_letter', 'pl_refname', 'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2', 'pl_orbperlim', 'pl_orbperstr', 'pl_orblpererr1', 'pl_orblper', 'pl_orblpererr2', 'pl_orblperlim', 'pl_orblperstr', 'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_orbsmaxlim', 'pl_orbsmaxstr', 'pl_orbincl', 'pl_orbinclerr1', 'pl_orbinclerr2', 'pl_orbincllim', 'pl_orbinclstr', 'pl_orbtper', 'pl_orbtpererr1', 'pl_orbtpererr2', 'pl_orbtperlim', 'pl_orbtperstr', 'pl_orbeccen', 'pl_orbeccenerr1', 'pl_orbeccenerr2', 'pl_orbeccenlim', 'pl_orbeccenstr', 'pl_eqt', 'pl_eqterr1', 'pl_eqterr2', 'pl_eqtlim', 'pl_eqtstr', 'pl_occdep', 'pl_occdeperr1', 'pl_occdeperr2', 'pl_occdeplim', 'pl_occdepstr', 'pl_insol', 'pl_insolerr1', 'pl_insolerr2', 'pl_insollim', 'pl_insolstr', 'pl_dens', 'pl_denserr1', 'pl_denserr2', 'pl_denslim', 'pl_densstr', 'pl_trandep', 'pl_trandeperr1', 'pl_trandeperr2', 'pl_trandeplim', 'pl_trandepstr', 'pl_tranmid', 'pl_tranmiderr1', 'pl_tranmiderr2', 'pl_tranmidlim', 'pl_tranmidstr', 'pl_trandur', 'pl_trandurerr1', 'pl_trandurerr2', 'pl_trandurlim', 'pl_trandurstr', 'pl_controv_flag', 'pl_tsystemref', 'pl_projobliq', 'pl_projobliqerr1', 'pl_projobliqerr2', 'pl_projobliqlim', 'pl_projobliqstr', 'pl_rvamp', 'pl_rvamperr1', 'pl_rvamperr2', 'pl_rvamplim', 'pl_rvampstr', 'pl_radj', 'pl_radjerr1', 'pl_radjerr2', 'pl_radjlim', 'pl_radjstr', 'pl_radestr', 'pl_ratror', 'pl_ratrorerr1', 'pl_ratrorerr2', 'pl_ratrorlim', 'pl_ratrorstr', 'pl_ratdor', 'pl_trueobliq', 'pl_trueobliqerr1', 'pl_trueobliqerr2', 'pl_trueobliqlim', 'pl_trueobliqstr', 'pl_pubdate', 'pl_ratdorerr1', 'pl_ratdorerr2', 'pl_ratdorlim', 'pl_ratdorstr', 'pl_imppar', 'pl_impparerr1', 'pl_impparerr2', 'pl_impparlim', 'pl_impparstr', 'pl_cmassj', 'pl_cmassjerr1', 'pl_cmassjerr2', 'pl_cmassjlim', 'pl_cmassjstr', 'pl_cmasse', 'pl_cmasseerr1', 'pl_cmasseerr2', 'pl_cmasselim', 'pl_cmassestr', 'pl_massj', 'pl_massjerr1', 'pl_massjerr2', 'pl_massjlim', 'pl_massjstr', 'pl_massestr', 'pl_bmassj', 'pl_bmassjerr1', 'pl_bmassjerr2', 'pl_bmassjlim', 'pl_bmassjstr', 'pl_bmasse', 'pl_bmasseerr1', 'pl_bmasseerr2', 'pl_bmasselim', 'pl_bmassestr', 'pl_bmassprov', 'pl_msinij', 'pl_msinijerr1', 'pl_msinijerr2', 'pl_msinijlim', 'pl_msinijstr', 'pl_msiniestr', 'pl_nespec', 'pl_ntranspec', 'pl_nnotes', 'pl_def_override', 'pl_calc_sma', 'pl_angsep', 'pl_angseperr1', 'pl_angseperr2', 'pl_radj_forecastermod', 'pl_radj_forecastermoderr1', 'pl_radj_forecastermoderr2', 'pl_radj_fortney', 'pl_radj_fortneyerr1', 'pl_radj_fortneyerr2', 'pl_maxangsep', 'pl_minangsep', 'disc_year', 'disc_refname', 'discoverymethod', 'disc_locale', 'ima_flag', 'disc_instrument', 'disc_telescope', 'disc_facility', 'rv_flag']
cache = True
datestr = Time.now().datetime.strftime("%Y-%m")

  
  
# TODO fix different rows, so that diff rows is base on actual columns to compare, compare the switched values
#look into comparing values more in depth
def get_different_rows(df1, df2):
  df_combined = pd.merge(df1, df2, indicator=True, how='outer')
  different_rows = df_combined[df_combined['_merge'] != 'both']
  different_rows = different_rows.drop(columns=['_merge'])

  return different_rows

password = input("Password:")
engine = create_engine('mysql+pymysql://'+"andrewchiu"+':'+password+'@localhost/testSios',echo=True)

with engine.connect() as connection:
  
  sql = text("SELECT * FROM Planets")
  results = connection.execute(sql)
  df = pd.DataFrame(results.fetchall(), columns = results.keys())
  
  
  
  data_path = Path(f'cache/data_cache_{datestr}.p')
  if cache:
      Path('cache/').mkdir(parents=True, exist_ok=True)
      if data_path.exists():
          with open(data_path, 'rb') as f:
              ipac_data = pickle.load(f)
      else:
          ipac_data = getIPACdata()
          with open(data_path, 'wb') as f:
              pickle.dump(data, f)
  else:
      ipac_data = getIPACdata()
      
    
  changed_rows = []
  
  for index, row in df.iterrows():
    name = row['pl_name']
    print(name)
    
    filter = ipac_data.query(f'pl_name == "{name}"')
    filtered_planet = filter.loc(1)
    
    for col_name in columns:
    
      if filtered_planet[col_name].count() > 0:
        ipac_col = filtered_planet[col_name].values[0]
      sios_col = row[col_name]
      
      print(f"{name} ipac maxangsep   {ipac_col}")
      print(f"sios maxangsep {sios_col}")

      if  ipac_col != None:
        if  (ipac_col != sios_col):
          print(f"different")
          changed_rows.append(row["pl_name"])
      else:
        changed_rows.append(row['pl_name'])
        
  set_changed_row = list(set(changed_rows))
  print(f"Changed rows: {set_changed_row}")
  
  print(set_changed_row)
  
  filtered_ipac = ipac_data[ipac_data['pl_name'].isin(set_changed_row)]
  
  filtered_ipac.to_csv('output.csv', index=True)
  
      
  
  
  #successfully have all planets (set of all different planets)
  # now update recalc 
  # start with 
  # photo data good
  # band info good
  # orb data do now
  
  #orb data

  
  with open('debug.txt', 'w') as file:
    
    file.write("loading photometry data\n")

    photdict_path = Path(f'cache/update_photdict_2022-05.p')
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if photdict_path.exists():
            with open(photdict_path, 'rb') as f:
                photdict = pickle.load(f)
        else:
            photdict = loadPhotometryData(infile="plandb.sioslab.com/allphotdata_2015.npz")
            with open(photdict_path, 'wb') as f:
                pickle.dump(photdict, f)
    else:
      photdict = loadPhotometryData(infile="plandb.sioslab.com/allphotdata_2015.npz")
        
    file.write("Bandzip\n")
    bandzip_path = Path(f'cache/update_bandzip_{datestr}.p')
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if bandzip_path.exists():
            with open(bandzip_path, 'rb') as f:
                bandzip = pickle.load(f)
        else:
            bandzip = list(genBands())
            with open(bandzip_path, 'wb') as f:
                pickle.dump(bandzip, f)
    else:
        bandzip = list(genBands())
    

    file.write("orbdata/orbfits\n")
    orbdata_path = Path(f'cache/update_orbdata_{datestr}.p')
    orbfits_path = Path(f'cache/update_orbfits_{datestr}.p')
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if orbdata_path.exists():
            with open(orbdata_path, 'rb') as f:
                orbdata = pickle.load(f)
        if orbfits_path.exists():
            with open(orbfits_path, 'rb') as f:
                orbitfits = pickle.load(f)
        else:
            orbdata, orbitfits = genOrbitData(filtered_ipac, bandzip, photdict)
            with open(orbdata_path, 'wb') as f:
                pickle.dump(orbdata, f)
            with open(orbfits_path, 'wb') as f:
                pickle.dump(orbitfits, f)
    else:
        orbdata, orbitfits = genOrbitData(filtered_ipac, bandzip, photdict)
    
    file.write(f"orbdata: {orbdata}\n")
    file.write(f"orbfits: {orbitfits}\n")
    


    file.write("ephemeris orbitfits/orbdata")
    ephemeris_orbdata_path = Path(f'cache/update_ephemeris_orbdata_{datestr}.p')
    ephemeris_orbfits_path = Path(f'cache/update_ephemeris_orbfits_{datestr}.p')
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if ephemeris_orbdata_path.exists():
            with open(ephemeris_orbdata_path, 'rb') as f:
                ephemeris_orbdata = pickle.load(f)
        if ephemeris_orbfits_path.exists():
            with open(ephemeris_orbfits_path, 'rb') as f:
                ephemeris_orbitfits = pickle.load(f)
        else:
            ephemeris_orbitfits, ephemeris_orbdata = addEphemeris(filtered_ipac, orbitfits, orbdata, bandzip, photdict)
            with open(ephemeris_orbdata_path, 'wb') as f:
                pickle.dump(ephemeris_orbdata, f)
            with open(ephemeris_orbfits_path, 'wb') as f:
                pickle.dump(ephemeris_orbitfits, f)
    else:
        ephemeris_orbitfits, ephemeris_orbdata = addEphemeris(filtered_ipac, orbitfits, orbdata, bandzip, photdict)
    file.write(f"ephemeris orbitfits: {ephemeris_orbitfits}\n")
    file.write(f"ephemeris orbfits: {ephemeris_orbdata}\n")
    

    
    file.write("quadrature data")
    quadrature_data_path = Path(f'cache/update_quadrature_data_{datestr}.p')
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if quadrature_data_path.exists():
            with open(quadrature_data_path, 'rb') as f:
                quadrature_data = pickle.load(f)
        else:
            quadrature_data = calcQuadratureVals(ephemeris_orbitfits, bandzip, photdict)
            with open(quadrature_data_path, 'wb') as f:
                pickle.dump(quadrature_data, f)
    else:
        quadrature_data = calcQuadratureVals(ephemeris_orbitfits, bandzip, photdict)
    file.write(f"quadrature data: {quadrature_data}\n")
    
    
    file.write("contr data")
    contr_data_path = Path(f'cache/update_contr_data_{datestr}.p')
    exosims_json = 'plandb.sioslab.com/ci_perf_exosims.json'
    if cache:
        if contr_data_path.exists():
            contr_data = pd.read_pickle(contr_data_path)
        else:
            contr_data = calcContrastCurves(quadrature_data, exosims_json=exosims_json)
            with open(contr_data_path, 'wb') as f:
                pickle.dump(contr_data, f)
    else:
        contr_data = calcContrastCurves(quadrature_data, exosims_json=exosims_json)
    file.write(f"contr data: {contr_data}\n")

    
    
    print('Doing completeness calculations')
    comps_path = Path(f'cache/update_comps_{datestr}.p')
    compdict_path = Path(f'cache/update_compdict_{datestr}.p')
    comps_data_path = Path(f'cache/update_comps_data_{datestr}.p')
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if comps_path.exists():
            comps = pd.read_pickle(comps_path)
            with open(compdict_path, 'rb') as f:
                compdict = pickle.load(f)
            comps_data = pd.read_pickle(comps_data_path)
        else:
            comps, compdict, comps_data = calcPlanetCompleteness(contr_data, bandzip, photdict, exosims_json=exosims_json)
            comps.to_pickle(comps_path)
            with open(compdict_path, 'wb') as f:
                pickle.dump(compdict, f)
            comps_data.to_pickle(comps_data_path)
    else:
        comps, compdict, comps_data = calcPlanetCompleteness(contr_data, bandzip, photdict, exosims_json=exosims_json)
    file.write(f"comps: {comps}\n")
    file.write(f"compdict: {compdict}\n")
    file.write(f"comps_data: {comps_data}\n")


    file.write("generateTables")

    plandata_path = Path(f'cache/update_plandata_{datestr}.p')
    stdata_path = Path(f'cache/update_stdata_{datestr}.p')
    table_orbitfits_path = Path(f'cache/update_table_orbitfits_{datestr}.p')
    if cache:
        # plandata.to_pickle('plandata_'+datestr+'.pkl')
        # stdata.to_pickle('stdata_' + datestr + '.pkl')
        # orbitfits.to_pickle('orbitfits_'+datestr+'.pkl')
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if plandata_path.exists():
            plandata = pd.read_pickle(plandata_path)
            stdata = pd.read_pickle(stdata_path)
            table_data = pd.read_pickle(table_orbitfits_path)
        else:
            plandata, stdata, orbitfits = generateTables(filtered_ipac, quadrature_data)
            plandata.to_pickle(plandata_path)
            stdata.to_pickle(stdata_path)
            orbitfits.to_pickle(table_orbitfits_path)
    else:
        plandata, stdata, orbitfits = generateTables(filtered_ipac, comps_data)    

    file.write(f"plandata: {plandata}\n")
    file.write(f"stdata: {stdata}\n")
    file.write(f"orbitfits: {orbitfits}\n")
    
    
    
    def resolve(r):
      if pd.notna(r['value_right']):
        return r['value_right']
      return r['value_left']
    # Necessary to merge/upsert dataframes, because since indices are recalculated everytime, and multiple keys/indices, must be filtered by scenerio_name, upserting table2 to table1, taking new updated values from table2 and add new records, might have to reindex
    def mergeTables( table1, table2):
      merged_df = pd.merge(table1, table2, on='pl_name',how='outer',suffixes=('_left', '_right'))
        


        
      merged_df['value'] = merged_df.apply(resolve, axis=1)
        
      merged_df.drop(columns=['value_left', 'value_right'], inplace=True)
        
      return merged_df
        
        # filtered_planet = filter.loc(1)
        
        # for col_name in columns:
        
        #   if filtered_planet[col_name].count() > 0:
        #     ipac_col = filtered_planet[col_name].values[0]
        #   sios_col = row[col_name]
          
        #   print(f"{name} ipac maxangsep   {ipac_col}")
        #   print(f"sios maxangsep {sios_col}")

        #   if  ipac_col != None:
        #     if  (ipac_col != sios_col):
        #       print(f"different")
        #       changed_rows.append(row["pl_name"])
        #   else:
        #     changed_rows.append(row['pl_name'])
        
    def compileCompleteness():
      datestr = Time.now().datetime.strftime("%Y-%m")
      comp_data = pd.read_pickle(f"cache/comps_data_{datestr}.p")
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

        
      
    compiled_contr_curvs_path = Path(f'plandb.sioslab.com/cache/update_compiled_cont_curvs_{datestr}.p')
    contr_curvs_path = Path(f'plandb.sioslab.com/cache/update_cont_curvs_{datestr.replace("-", "_")}')


    if compiled_contr_curvs_path.exists():
      contrast_curves = pd.read_pickle(compiled_contr_curvs_path)
    else:
      contrast_curves = compileContrastCurves(stdata, contr_curvs_path)
      contrast_curves.to_pickle(compiled_contr_curvs_path)
    def addId(r):
      r['pl_id']= list(planets.index[(planets['pl_name'] == r['Name'])])[0]
      return r
    

    scenarios = pd.read_csv("plandb.sioslab.com/cache/scenario_angles.csv")
    compiled_completeness_path = Path(f"cache/update_compiled_completeness_{datestr}.p")
    # if compiled_completeness_path.exists():
    #   completeness = pd.read_pickle(compiled_completeness_path)
    # else:
    completeness = compileCompleteness()
    completeness.to_pickle(compiled_completeness_path)
                        
    sql = text("SELECT * FROM ContrastCurves")
    results = connection.execute(sql)
    contrast_df = pd.DataFrame(results.fetchall(), columns = results.keys())
    
    new_contrast = compileContrastCurves(stdata, contr_curvs_path)
    
    merged_contrast = mergeTables(contrast_df, contrast_curves)
    print(merged_contrast)
    
    
    
    # compare scenerio name with what we have here calculated from the newly calculated, and create a new dataframe that appends the old ones, the merged duplciates, and new ones, then just df.to_sql .if_exists == replace, with new 
    # not able to j do a query, doesnt allow for 1/3 indices, no other query to just upsert, especially since id's are reclaculated everytime
    
    
    
      
    
    
    
    
    
    
    
    # if(filtered_planet.loc[0, "{col_name}"] "{col_name}")
    # changed_rows.append
    
    
    
    
    
    
    
    #What if new planets get added?
  
  
  
  # sios_columns = results.keys()
  
  
  
  
  
  
  
  
  # print(df)
  
  # data = getIPACdata()
  
  # ipac_columns = data.keys()
  
  # print(data)
  
  # diff1 = list(sios_columns - ipac_columns)
  # print("sios columns" + sios_columns + "\n\n\n\n\n")
  # print(diff1)
  # print("ipac columns" +ipac_columns + "\n\n\n\n\n")
  # diff2 = list(ipac_columns - sios_columns)
  # print(diff2)
  
  # combined_diff = diff1+ diff2
  # print(combined_diff)
  
  #WOrks, 3800 rows here as oppose to 4800
  # NOTES branch/changed absolute path to relative path, did this (got changed rows that need to be changed, now just need to run claculations for only these rows, talk about the bad code/syntax errors (.F0, _append, and the engine connection, unequal pass in param, and sql text(), too big to be minor error, but finished database on my system :) )