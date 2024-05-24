import pandas as pd
from sqlalchemy import create_engine
import pymysql
from sqlalchemy import text
from plandb_methods import *
from database_main import *
import sweetviz as sv
from typing import Tuple


# Util functions for update, also compacts much of the code from gen_plandb and database_main

def get_current_database(connection: sqlalchemy.Connection) -> pd.DataFrame:
  results = connection.execute(text("SELECT * FROM PLANETS"))
  df = pd.DataFrame(results.fetchall(), columns = results.keys())
  return df
  
def get_ipac_database(data_path: Path, cache: bool) -> pd.DataFrame:
  # data_path = Path(f'cache/data_cache_{datestr}.p')
  if cache:
      Path('cache/').mkdir(parents=True, exist_ok=True)
      if data_path.exists():
          with open(data_path, 'rb') as f:
              ipac_data = pickle.load(f)
              return ipac_data
      else:
          ipac_data = getIPACdata()
          with open(data_path, 'wb') as f:
              pickle.dump(ipac_data, f)
          return ipac_data
  else:
      ipac_data = getIPACdata()
      return ipac_data
    
# returns updated, and added
# TODO change to find more accurate differences, using generateTables in plandb_methods, 
def find_row_differences(current_database: pd.DataFrame, new_database: pd.DataFrame) -> pd.DataFrame:
  changed_rows = []
  columns = ['pl_name', 'pl_letter', 'pl_refname', 'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2', 'pl_orbperlim', 'pl_orbperstr', 'pl_orblpererr1', 'pl_orblper', 'pl_orblpererr2', 'pl_orblperlim', 'pl_orblperstr', 'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_orbsmaxlim', 'pl_orbsmaxstr', 'pl_orbincl', 'pl_orbinclerr1', 'pl_orbinclerr2', 'pl_orbincllim', 'pl_orbinclstr', 'pl_orbtper', 'pl_orbtpererr1', 'pl_orbtpererr2', 'pl_orbtperlim', 'pl_orbtperstr', 'pl_orbeccen', 'pl_orbeccenerr1', 'pl_orbeccenerr2', 'pl_orbeccenlim', 'pl_orbeccenstr', 'pl_eqt', 'pl_eqterr1', 'pl_eqterr2', 'pl_eqtlim', 'pl_eqtstr', 'pl_occdep', 'pl_occdeperr1', 'pl_occdeperr2', 'pl_occdeplim', 'pl_occdepstr', 'pl_insol', 'pl_insolerr1', 'pl_insolerr2', 'pl_insollim', 'pl_insolstr', 'pl_dens', 'pl_denserr1', 'pl_denserr2', 'pl_denslim', 'pl_densstr', 'pl_trandep', 'pl_trandeperr1', 'pl_trandeperr2', 'pl_trandeplim', 'pl_trandepstr', 'pl_tranmid', 'pl_tranmiderr1', 'pl_tranmiderr2', 'pl_tranmidlim', 'pl_tranmidstr', 'pl_trandur', 'pl_trandurerr1', 'pl_trandurerr2', 'pl_trandurlim', 'pl_trandurstr', 'pl_controv_flag', 'pl_tsystemref', 'pl_projobliq', 'pl_projobliqerr1', 'pl_projobliqerr2', 'pl_projobliqlim', 'pl_projobliqstr', 'pl_rvamp', 'pl_rvamperr1', 'pl_rvamperr2', 'pl_rvamplim', 'pl_rvampstr', 'pl_radj', 'pl_radjerr1', 'pl_radjerr2', 'pl_radjlim', 'pl_radjstr', 'pl_radestr', 'pl_ratror', 'pl_ratrorerr1', 'pl_ratrorerr2', 'pl_ratrorlim', 'pl_ratrorstr', 'pl_ratdor', 'pl_trueobliq', 'pl_trueobliqerr1', 'pl_trueobliqerr2', 'pl_trueobliqlim', 'pl_trueobliqstr', 'pl_pubdate', 'pl_ratdorerr1', 'pl_ratdorerr2', 'pl_ratdorlim', 'pl_ratdorstr', 'pl_imppar', 'pl_impparerr1', 'pl_impparerr2', 'pl_impparlim', 'pl_impparstr', 'pl_cmassj', 'pl_cmassjerr1', 'pl_cmassjerr2', 'pl_cmassjlim', 'pl_cmassjstr', 'pl_cmasse', 'pl_cmasseerr1', 'pl_cmasseerr2', 'pl_cmasselim', 'pl_cmassestr', 'pl_massj', 'pl_massjerr1', 'pl_massjerr2', 'pl_massjlim', 'pl_massjstr', 'pl_massestr', 'pl_bmassj', 'pl_bmassjerr1', 'pl_bmassjerr2', 'pl_bmassjlim', 'pl_bmassjstr', 'pl_bmasse', 'pl_bmasseerr1', 'pl_bmasseerr2', 'pl_bmasselim', 'pl_bmassestr', 'pl_bmassprov', 'pl_msinij', 'pl_msinijerr1', 'pl_msinijerr2', 'pl_msinijlim', 'pl_msinijstr', 'pl_msiniestr', 'pl_nespec', 'pl_ntranspec', 'pl_nnotes', 'pl_def_override', 'pl_calc_sma', 'pl_angsep', 'pl_angseperr1', 'pl_angseperr2', 'pl_radj_forecastermod', 'pl_radj_forecastermoderr1', 'pl_radj_forecastermoderr2', 'pl_radj_fortney', 'pl_radj_fortneyerr1', 'pl_radj_fortneyerr2', 'pl_maxangsep', 'pl_minangsep', 'disc_year', 'disc_refname', 'discoverymethod', 'disc_locale', 'ima_flag', 'disc_instrument', 'disc_telescope', 'disc_facility', 'rv_flag']
  

  for index, row in current_database.iterrows():
    name = row['pl_name']
    print(name)
    
    filter = new_database.query(f'pl_name == "{name}"')
    filtered_planet = filter.loc(1)
    
    for col_name in columns:
    
      if filtered_planet[col_name].count() > 0:
        ipac_col = filtered_planet[col_name].values[0]
      sios_col = row[col_name]
      
      if  ipac_col != None:
        if  (ipac_col != sios_col):
          changed_rows.append(row["pl_name"])
      else:
        changed_rows.append(row['pl_name'])
        
  set_changed_row = list(set(changed_rows))
    
  return new_database[new_database['pl_name'].isin(set_changed_row)]

def get_photo_data(photdict_path: Path, infile: str, cache: bool):
  # photdict_path = Path(f'cache/update_photdict_2022-05.p')
  if cache:
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if photdict_path.exists():
      with open(photdict_path, 'rb') as f:
        photdict = pickle.load(f)
        return photdict
    else:
      # photdict = loadPhotometryData(infile="plandb.sioslab.com/allphotdata_2015.npz")
      photdict = loadPhotometryData(infile=infile)
      with open(photdict_path, 'wb') as f:
        pickle.dump(photdict, f)
      return photdict
  else:
    # photdict = loadPhotometryData(infile="plandb.sioslab.com/allphotdata_2015.npz")
    photdict = loadPhotometryData(infile=infile)
    return photdict
  
def get_bandzip(bandzip_path: Path, cache: bool):
    # bandzip_path = Path(f'cache/update_bandzip_{datestr}.p')
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if bandzip_path.exists():
            with open(bandzip_path, 'rb') as f:
                bandzip = pickle.load(f)
                return bandzip
        else:
            bandzip = list(genBands())
            with open(bandzip_path, 'wb') as f:
                pickle.dump(bandzip, f)
            return bandzip
    else:
        bandzip = list(genBands())
        return bandzip
      
def get_orbdata(orbdata_path: Path, orbfits_path: Path, data: pd.DataFrame, bandzip, photdict, cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # orbdata_path = Path(f'cache/update_orbdata_{datestr}.p')
  # orbfits_path = Path(f'cache/update_orbfits_{datestr}.p')
  if cache:
      orbdata = pd.DataFrame()
      orbitfits = pd.DataFrame()
      Path('cache/').mkdir(parents=True, exist_ok=True)
      if orbdata_path.exists():
            with open(orbdata_path, 'rb') as f:
                orbdata = pickle.load(f)
      if orbfits_path.exists():
            with open(orbfits_path, 'rb') as f:
                orbitfits = pickle.load(f)
      else:
            orbdata, orbitfits = genOrbitData(data, bandzip, photdict)
            with open(orbdata_path, 'wb') as f:
                pickle.dump(orbdata, f)
            with open(orbfits_path, 'wb') as f:
                pickle.dump(orbitfits, f)
  else:
        orbdata, orbitfits = genOrbitData(data, bandzip, photdict)
  return orbdata, orbitfits
        
def get_ephemerisdata(ephemeris_orbdata_path: Path, ephemeris_orbfits_path: Path, data: pd.DataFrame, orbitfits: pd.DataFrame, orbdata: pd.DataFrame, bandzip, photdict, cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ephemeris_orbdata_path = Path(f'cache/update_ephemeris_orbdata_{datestr}.p')
    # ephemeris_orbfits_path = Path(f'cache/update_ephemeris_orbfits_{datestr}.p')
    ephemeris_orbitfits = pd.DataFrame()
    ephemeris_orbdata = pd.DataFrame()
    if cache:
        Path('cache/').mkdir(parents=True, exist_ok=True)
        if ephemeris_orbdata_path.exists():
            with open(ephemeris_orbdata_path, 'rb') as f:
                ephemeris_orbdata = pickle.load(f)
        if ephemeris_orbfits_path.exists():
            with open(ephemeris_orbfits_path, 'rb') as f:
                ephemeris_orbitfits = pickle.load(f)
        else:
            ephemeris_orbitfits, ephemeris_orbdata = addEphemeris(data, orbitfits, orbdata, bandzip, photdict)
            with open(ephemeris_orbdata_path, 'wb') as f:
                pickle.dump(ephemeris_orbdata, f)
            with open(ephemeris_orbfits_path, 'wb') as f:
                pickle.dump(ephemeris_orbitfits, f)
    else:
        ephemeris_orbitfits, ephemeris_orbdata = addEphemeris(data, orbitfits, orbdata, bandzip, photdict)
    return ephemeris_orbitfits, ephemeris_orbdata
  
def get_quadrature(quadrature_data_path: Path, ephemeris_orbitfits: pd.DataFrame, bandzip, photdict, cache: bool) -> pd.DataFrame:
  # quadrature_data_path = Path(f'cache/update_quadrature_data_{datestr}.p')
  quadrature_data = pd.DataFrame()
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
        
  return quadrature_data

def get_contrastness(contr_data_path: Path, exosims_json, quadrature_data: pd.DataFrame, cache: bool) -> pd.DataFrame:
    # contr_data_path = Path(f'cache/update_contr_data_{datestr}.p')
    # exosims_json = 'plandb.sioslab.com/ci_perf_exosims.json'
    contr_data = pd.DataFrame()
    if cache:
        if contr_data_path.exists():
            contr_data = pd.read_pickle(contr_data_path)
        else:
            contr_data = calcContrastCurves(quadrature_data, exosims_json=exosims_json)
            with open(contr_data_path, 'wb') as f:
                pickle.dump(contr_data, f)
    else:
        contr_data = calcContrastCurves(quadrature_data, exosims_json=exosims_json)
    return contr_data
  
def get_completeness(comps_path: Path, compdict_path: Path, comps_data_path: Path, contr_data: pd.DataFrame, bandzip, photdict, exosims_json, cache: bool) -> Tuple[pd.DataFrame, dict, pd.DataFrame]: 
    # comps_path = Path(f'cache/update_comps_{datestr}.p')
    # compdict_path = Path(f'cache/update_compdict_{datestr}.p')
    # comps_data_path = Path(f'cache/update_comps_data_{datestr}.p')
    comps = pd.DataFrame()
    compdict = pd.DataFrame()
    comps_data = pd.DataFrame()
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
    return comps, compdict, comps_data
  
def get_generated_tables(plandata_path: Path, stdata_path: Path, table_orbitfits_path: Path, data: pd.DataFrame, quadrature_data: pd.DataFrame , comps_data : pd.DataFrame, cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # plandata_path = Path(f'cache/update_plandata_{datestr}.p')
    # stdata_path = Path(f'cache/update_stdata_{datestr}.p')
    # table_orbitfits_path = Path(f'cache/update_table_orbitfits_{datestr}.p')
    plandata = pd.DataFrame()
    stdata = pd.DataFrame()
    orbitfits = pd.DataFrame()
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
            plandata, stdata, orbitfits = generateTables(data, quadrature_data)
            plandata.to_pickle(plandata_path)
            stdata.to_pickle(stdata_path)
            orbitfits.to_pickle(table_orbitfits_path)
    else:
        plandata, stdata, orbitfits = generateTables(data, comps_data)   
    return plandata, stdata, orbitfits
  
# TODO make get_contrast, and compile, within one func
def get_compiled_contrast(compiled_contr_curvs_path: Path, stars: pd.DataFrame, pdfs: pd.DataFrame, planets: pd.DataFrame, contr_curvs_path: Path):
    # compiled_contr_curvs_path = Path(f'plandb.sioslab.com/cache/compiled_cont_curvs_{datestr}.p')
    # contr_curvs_path = Path(f'plandb.sioslab.com/cache/cont_curvs_{datestr.replace("-", "_")}')
    contrast_curves = pd.DataFrame()
    newpdfs = pd.DataFrame()
    if compiled_contr_curvs_path.exists():
        contrast_curves = pd.read_pickle(compiled_contr_curvs_path)
    else:
        contrast_curves = compileContrastCurves(stars, contr_curvs_path)
        contrast_curves.to_pickle(compiled_contr_curvs_path)
    # TODO finish new pdfs
    # def addId(r):
    #     r['pl_id']= list(planets.index[(planets['pl_name'] == r['Name'])])[0]
    #     return r
    # TODO figure out whats happening, why does the original of this from database_main use pdfs = comp_data
    # newpdfs = pdfs.apply(addId, axis = 1)
    
    return contrast_curves, newpdfs
    
def get_compiled_completeness(compiled_completeness_path: Path, comple_data: pd.DataFrame) -> pd.DataFrame:
    # scenarios = pd.read_csv("plandb.sioslab.com/cache/scenario_angles.csv")
    # compiled_completeness_path = Path(f"plandb.sioslab.com/cache/compiled_completeness_{datestr}.p")
    
    def compileCompleteness(comp_data: pd.DataFrame) -> pd.DataFrame:
      # datestr = Time.now().datetime.strftime("%Y-%m")
      # comp_data = pd.read_pickle(f"plandb.sioslab.com/cache/comps_data_{datestr}.p")
      completeness = pd.DataFrame()
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
  
    completeness = pd.DataFrame()
    if compiled_completeness_path.exists():
        completeness = pd.read_pickle(compiled_completeness_path)
    else:
        completeness = compileCompleteness(comple_data)
        completeness.to_pickle(compiled_completeness_path)
    return completeness
        
# create a new shorter database of the updated, that is able to merge later 
# TODO change this to only make dataframe, and then add to database
def update_sql(engine, plandata=None, stdata=None, orbitfits=None, orbdata=None, pdfs=None, aliases=None,contrastCurves=None,scenarios=None, completeness=None):
  return
  
def get_all_from_db(connection: sqlalchemy.Connection) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  planets_results = connection.execute(text("SELECT * FROM PLANETS"))
  planets_df = pd.DataFrame(planets_results.fetchall(), columns = planets_results.keys())
  
  contrast_curves_results = connection.execute(text("SELECT * FROM ContrastCurves"))
  contrast_curves_df = pd.DataFrame(contrast_curves_results.fetchall(), columns = contrast_curves_results.keys())
  
  orbitfits_results = connection.execute(text("SELECT * FROM OrbitFits"))
  orbitfits_df = pd.DataFrame(orbitfits_results.fetchall(), columns = orbitfits_results.keys())

  
  orbits_results = connection.execute(text("SELECT * FROM Orbits"))
  orbits_df = pd.DataFrame(orbits_results.fetchall(), columns = orbits_results.keys())
  
  # TODO, fix to pdfs, do completeness for now, as pdfs is currently empty from diff_database
  pdfs_results = connection.execute(text("SELECT * FROM Completeness"))
  pdfs_df = pd.DataFrame(pdfs_results.fetchall(), columns = pdfs_results.keys())
  
  scenarios_results = connection.execute(text("SELECT * FROM Scenarios"))
  scenarios_df = pd.DataFrame(scenarios_results.fetchall(), columns = scenarios_results.keys())
  
  stars_results = connection.execute(text("SELECT * FROM Stars"))
  stars_df = pd.DataFrame(stars_results.fetchall(), columns = stars_results.keys())
  
  completeness_results = connection.execute(text("SELECT * FROM Completeness"))
  completeness_df = pd.DataFrame(completeness_results.fetchall(), columns = completeness_results.keys())
  
  return completeness_df, contrast_curves_df, orbitfits_df, orbits_df, pdfs_df, planets_df, scenarios_df, stars_df

def upsert_dataframe(current_df : pd.DataFrame, new_df : pd.DataFrame , key_column) -> pd.DataFrame:
  current_df.set_index(key_column, inplace=True)
  new_df.set_index(key_column, inplace=True)

  df_combined = current_df.combine_first(new_df)

  df_combined.reset_index(inplace=True)

  return df_combined
def temp_writeSQL(engine, plandata=None, stdata=None, orbitfits=None, orbdata=None, pdfs=None, aliases=None,contrastCurves=None,scenarios=None, completeness=None):
    """write outputs to sql database via connection"""
    connection = engine.connect()
    connection.execute(text("DROP TABLE IF EXISTS Completeness, ContrastCurves, Scenarios, PDFs, Orbits, OrbitFits, Planets, Stars"))

    if stdata is not None:
        print("Writing Stars")
        namemxchar = np.array([len(n) for n in stdata['st_name'].values]).max()
        stdata = stdata.rename_axis('st_id')
        stdata.to_sql('Stars', connection, chunksize=100, if_exists='replace',
                    dtype={'st_id': sqlalchemy.types.INT,
                           'st_name': sqlalchemy.types.String(namemxchar)})
        # set indexes
        result = connection.execute(text('ALTER TABLE Stars ADD INDEX (st_id)'))

        # add comments
        # addSQLcomments(connection, 'Stars')

    if plandata is not None:
        print("Writing Planets")
        namemxchar = np.array([len(n) for n in plandata['pl_name'].values]).max()
        plandata = plandata.rename_axis('pl_id')
        plandata.to_sql('Planets',connection,chunksize=100,if_exists='replace',
                    dtype={'pl_id':sqlalchemy.types.INT,
                            'pl_name':sqlalchemy.types.String(namemxchar),
                            'st_name':sqlalchemy.types.String(namemxchar-2),
                            'pl_letter':sqlalchemy.types.CHAR(1),
                            'st_id': sqlalchemy.types.INT})
        #set indexes
        result = connection.execute(text("ALTER TABLE Planets ADD INDEX (pl_id)"))
        result = connection.execute(text("ALTER TABLE Planets ADD INDEX (st_id)"))
        result = connection.execute(text("ALTER TABLE Planets ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));

        #add comments
        # addSQLcomments(connection,'Planets')

    if orbitfits is not None:
        print("Writing OrbitFits")
        orbitfits = orbitfits.rename_axis('orbitfit_id')
        namemxchar = np.array([len(n) for n in orbitfits['pl_name'].values]).max()
        orbitfits.to_sql('OrbitFits',connection,chunksize=100,if_exists='replace',
                          dtype={'pl_id': sqlalchemy.types.INT,
                                 'orbitfit_id': sqlalchemy.types.INT,
                                 'pl_name': sqlalchemy.types.String(namemxchar)},
                          index=True)
        result = connection.execute(text("ALTER TABLE OrbitFits ADD INDEX (orbitfit_id)"))
        result = connection.execute(text("ALTER TABLE OrbitFits ADD INDEX (pl_id)"))
        # TODO Get this commented back in to correctly have foreign keys
        # result = connection.execute(text("ALTER TABLE OrbitFits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));

        # addSQLcomments(connection,'OrbitFits')

    if orbdata is not None:
        print("Writing Orbits")
        namemxchar = np.array([len(n) for n in orbdata['pl_name'].values]).max()
        orbdata = orbdata.rename_axis('orbit_id')
        orbdata.to_sql('Orbits',connection,chunksize=100,if_exists='replace',
                       dtype={'pl_name':sqlalchemy.types.String(namemxchar),
                              'pl_id': sqlalchemy.types.INT,
                              'orbit_id': sqlalchemy.types.BIGINT,
                              'orbitfit_id': sqlalchemy.types.INT},
                       index=True)
        result = connection.execute(text("ALTER TABLE Orbits ADD INDEX (orbit_id)"))
        result = connection.execute(text("ALTER TABLE Orbits ADD INDEX (pl_id)"))
        result = connection.execute(text("ALTER TABLE Orbits ADD INDEX (orbitfit_id)"))
        #TODO Get this commented back in to correctly have foreign keys
        # result = connection.execute(text("ALTER TABLE Orbits ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));
        # result = connection.execute(text("ALTER TABLE Orbits ADD FOREIGN KEY (orbitfit_id) REFERENCES OrbitFits(orbitfit_id) ON DELETE NO ACTION ON UPDATE NO ACTION"));

        # addSQLcomments(connection,'Orbits')

    if pdfs is not None:
        print("Writing PDFs")
        pdfs = pdfs.reset_index(drop=True)
        namemxchar = np.array([len(n) for n in pdfs['Name'].values]).max()
        pdfs = pdfs.rename_axis('pdf_id')
        pdfs.to_sql('PDFs',connection,chunksize=100,if_exists='replace',
                     dtype={'pl_name':sqlalchemy.types.String(namemxchar),
                            'pl_id': sqlalchemy.types.INT})
        # result = connection.execute("ALTER TABLE PDFs ADD INDEX (orbitfit_id)")
        result = connection.execute(text("ALTER TABLE PDFs ADD INDEX (pl_id)"))
        result = connection.execute(text("ALTER TABLE PDFs ADD INDEX (pdf_id)"))
        # result = connection.execute("ALTER TABLE PDFs ADD FOREIGN KEY (orbitfit_id) REFERENCES OrbitFits(orbitfit_id) ON DELETE NO ACTION ON UPDATE NO ACTION")
        result = connection.execute(text("ALTER TABLE PDFs ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))

        # addSQLcomments(connection,'PDFs')

    if aliases is not None:
        print("Writing Alias")
        aliases = aliases.rename_axis('alias_id')
        aliasmxchar = np.array([len(n) for n in aliases['Alias'].values]).max()
        aliases.to_sql('Aliases',connection,chunksize=100,if_exists='replace',dtype={'Alias':sqlalchemy.types.String(aliasmxchar)})
        result = connection.execute(text("ALTER TABLE Aliases ADD INDEX (alias_id)"))
        result = connection.execute(text("ALTER TABLE Aliases ADD INDEX (Alias)"))
        result = connection.execute(text("ALTER TABLE Aliases ADD INDEX (st_id)"))
        result = connection.execute(text("ALTER TABLE Aliases ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))

    if scenarios is not None:
        print("Writing Scenarios")

        namemxchar = np.array([len(n) for n in scenarios['scenario_name'].values]).max()
        scenarios.to_sql("Scenarios", connection, chunksize=100, if_exists='replace', dtype={
            'scenario_name': sqlalchemy.types.String(namemxchar),}, index = False)

        result = connection.execute(text("ALTER TABLE Scenarios ADD INDEX (scenario_name)"))

    if contrastCurves is not None:
        print("Writing ContrastCurves")
        contrastCurves = contrastCurves.rename_axis("curve_id")
        #TODO change namemx char to commented out line, temporary value of 1000000 used here as a placeholder
        # namemxchar = np.array([len(n) for n in contrastCurves['scenario_name'].values]).max()
        namemxchar = 1000
        contrastCurves.to_sql("ContrastCurves", connection, chunksize=100, if_exists='replace', dtype={
            'st_id': sqlalchemy.types.INT,
            'curve_id' : sqlalchemy.types.INT,
            'scenario_name': sqlalchemy.types.String(namemxchar)}, index = True)
        # result = connection.execute("ALTER TABLE ContastCurves ADD INDEX (scenario_name)")

        # result = connection.execute("ALTER TABLE ContastCurves ADD INDEX (st_id)")

        # TODO: get this commented back in to have correct foreign keys
        # result = connection.execute(text("ALTER TABLE ContrastCurves ADD FOREIGN KEY (st_id) REFERENCES Stars(st_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))
        # result = connection.execute(text("ALTER TABLE ContrastCurves ADD FOREIGN KEY (scenario_name) REFERENCES Scenarios(scenario_name) ON DELETE NO ACTION ON UPDATE NO ACTION"))

    if completeness is not None:
        print("Writing completeness")
        completeness = completeness.rename_axis("completeness_id")
        namemxchar = np.array([len(n) for n in completeness['scenario_name'].values]).max()
        completeness.to_sql("Completeness", connection, chunksize=100, if_exists='replace', dtype={
            # 'scenario_id': sqlalchemy.types.INT,
            'pl_id' : sqlalchemy.types.INT,
            'completeness_id' : sqlalchemy.types.INT,
            'scenario_name': sqlalchemy.types.String(namemxchar)}, index = True)

        # TODO: get this commented back in to have correct foreign keys
        # result = connection.execute(text("ALTER TABLE Completeness ADD FOREIGN KEY (pl_id) REFERENCES Planets(pl_id) ON DELETE NO ACTION ON UPDATE NO ACTION"))
        # result = connection.execute(text("ALTER TABLE ContrastCurves ADD FOREIGN KEY (scenario_name) REFERENCES Scenarios(scenario_name) ON DELETE NO ACTION ON UPDATE NO ACTION"))

  
  
  
  
  

  

