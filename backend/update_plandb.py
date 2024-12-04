import pandas as pd
from sqlalchemy import create_engine
import pymysql
import glob
from sqlalchemy import text
from plandb_methods import *
from database_main import *
from update_util import *
import os


#Flags
cache = True
datestr = Time.now().datetime.strftime("%Y-%m")

# sign in originally before running, to solve cryptography error, sign in using mysql -u username (andrewchiu) -p, then cryptography is solved
# workflow: get current database -> get ipac database -> compare current database to updated values in ipac database, to create a difference dataframe -> create a database from the difference dataframe (updated database) -> merge/upsert the difference database with the current database -> replace current database, with the merged, keep old values and adding updated values
# create connection with current database
password = input("SQL password: ")
engine = create_engine('mysql+pymysql://'+"andrewchiu"+':'+password+'@localhost/testSios',echo=True)
diff_engine = create_engine('mysql+pymysql://'+"andrewchiu"+':'+password+'@localhost/testSiosDiffEngine',echo=True)
new_engine = create_engine('mysql+pymysql://'+"andrewchiu"+':'+password+'@localhost/newEngine',echo=True)


with engine.connect() as connection:      
      
      

    # with diff_engine.connect() as diff_engine_connection:
      
      # get current database dataframe
      print("getting current database")
      current_database_df = get_current_database(connection)

      current_database_df.to_excel("plandb.sioslab.com/backend/sheets/current_database.xlsx", index=False)
      
      # get ipac database dataframe
      print("getting ipac database")
      
      ipac_archive_dir = "plandb.sioslab.com/backend/ipac_archive"
      
      files = os.listdir(ipac_archive_dir)
      
      most_recent_file = files[0]
      most_recent_ind = 0
      
      for file in files:

        ind_str = file.rsplit('-', 1)[-1]
        current_ind = int(ind_str)
        
        if(current_ind > most_recent_ind):
          most_recent_ind = current_ind
          most_recent_file = file
      
      #load archive, load first because it needs be the most recent excluding month,
      print("loading archived ipac")
          
          
      data_path = Path(f'cache/data_cache_{datestr}.p')
      ipac_data_df = get_ipac_database(data_path, cache)
      # Archive everytime, replace if same month, archive for future use
      print("archiving present time ipac")
      with open(Path(f"plandb.sioslab.com/backend/ipac_archive/ipac_archive_{datestr}_{most_recent_ind + 1}.p"), 'wb') as f:
          pickle.dump(ipac_data_df, f)
              
              
      most_recent_ipac_data = pickle.load(Path(f"plandb.sioslab.com/backend/ipac_archive/{most_recent_file}"))
      
      change_ipac_df = find_row_differences(most_recent_ipac_data, ipac_data_df)
      
      
      #find updates from current to ipac
      changed_df = find_row_differences(current_database_df, ipac_data_df)
      changed_df.to_excel("plandb.sioslab.com/backend/sheets/changed.xlsx")      
      changed_df.to_pickle(Path(f"plandb.sioslab.com/changed_df"))
      
      changed_df = pd.read_pickle(Path(f"plandb.sioslab.com/changed_df"))
      
      
      # get photodict
      photdict_path = Path(f'cache/update_photdict_2022-05.p')
      infile="plandb.sioslab.com/backend/allphotdata_2015.npz"
      photdict = get_photo_data(photdict_path, infile, cache)
      

      # get bandzip
      bandzip_path = Path(f'cache/update_bandzip_{datestr}.p')
      bandzip = get_bandzip(bandzip_path, cache)
      
      # get orbdata, orbfits
      orbdata_path = Path(f'cache/update_orbdata_{datestr}.p')
      orbfits_path = Path(f'cache/update_orbfits_{datestr}.p')
      orbdata, orbfits = get_orbdata(orbdata_path, orbfits_path, changed_df, bandzip, photdict, cache)
      
      # Line below, contains too main records to write xlsx, check manually
      # orbdata.to_excel("orbdata.xlsx")
      orbfits.to_excel("orbfits.xlsx")
      
      # get ephemeris 
      ephemeris_orbdata_path = Path(f'cache/update_ephemeris_orbdata_{datestr}.p')
      ephemeris_orbfits_path = Path(f'cache/update_ephemeris_orbfits_{datestr}.p')
      ephemeris_orbitfits, ephemeris_orbdata = get_ephemerisdata(ephemeris_orbdata_path, ephemeris_orbfits_path, changed_df, orbfits, orbdata, bandzip, photdict, cache)
      ephemeris_orbitfits.to_excel("ephemeris_orbfits.xlsx")
      
      # Line below, contains too main records to write xlsx, check manually
      # ephemeris_orbdata.to_excel("ephemeris_orbdata.xlsx")
      
      quadrature_data_path = Path(f'cache/update_quadrature_data_{datestr}.p')
      quadrature_data = get_quadrature(quadrature_data_path, ephemeris_orbitfits, bandzip, photdict, cache)
      quadrature_data.to_excel("plandb.sioslab.com/backend/sheets/quadrature_data.xlsx")
      
      contr_data_path = Path(f'cache/update_contr_data_{datestr}.p')
      exosims_json = 'plandb.sioslab.com/ci_perf_exosims.json'
      
      contr_data = get_contrastness(contr_data_path, exosims_json, quadrature_data, cache)
      contr_data.to_excel("plandb.sioslab.com/backend/sheets/contr_data.xlsx")
      
      comps_path = Path(f'cache/update_comps_{datestr}.p')
      compdict_path = Path(f'cache/update_compdict_{datestr}.p')
      comps_data_path = Path(f'cache/update_comps_data_{datestr}.p')
      
      comps, compdict, comps_data = get_completeness(comps_path, compdict_path, comps_data_path, contr_data, bandzip, photdict, exosims_json, cache)
      comps.to_excel("plandb.sioslab.com/backend/sheets/comps.xlsx")
      comps_data.to_excel("plandb.sioslab.com/backend/sheets/comps_data.xlsx")
      #None for compdict, as its dictionary
      
      plandata_path = Path(f'cache/update_plandata_{datestr}.p')
      stdata_path = Path(f'cache/update_stdata_{datestr}.p')
      table_orbitfits_path = Path(f'cache/update_table_orbitfits_{datestr}.p')
      
      # orbitfits got updated, maybe change to new var
      plan_data, stdata, orbitfits = get_generated_tables(plandata_path, stdata_path, table_orbitfits_path, changed_df, quadrature_data, comps_data, cache)
      plan_data.to_excel("plandb.sioslab.com/backend/sheets/plandata.xlsx")
      stdata.to_excel("plandb.sioslab.com/backend/sheets/stdata.xlsx")
      orbitfits.to_excel('plandb.sioslab.com/backend/sheets/later_orbitfits.xlsx')
      
      #do compileContrastness and compile contrastness
      # Look into and possibly remove the - to _
      contr_curvs2_path = Path(f'plandb.sioslab.com/cache/cont_curvs2_{datestr.replace("-", "_")}')
      compiled_contr_curvs_path = Path(f'plandb.sioslab.com/cache/cont_curvs_{datestr.replace("-", "_")}')
      compiled_contrast_curves, newpdfs = get_compiled_contrast(compiled_contr_curvs_path, stdata, comps_data, changed_df, contr_curvs2_path)
      # compiled_contrast_curves = compileContrastCurves(stdata, compiled_contr_curvs_path)
      compiled_contrast_curves.to_excel("plandb.sioslab.com/backend/sheets/compiled_contrast_curves.xlsx")
      # With current code, since get_compiled_contrast isnt fully working with new pdfs, new pdfs should be empty
      newpdfs.to_excel("plandb.sioslab.com/backend/sheets/newpdfs.xlsx")
      
      # compile completeness
      compiled_completeness_path = Path(f"plandb.sioslab.com/cache/compiled_completeness_{datestr}.p")

      compiled_completeness = get_compiled_completeness(compiled_completeness_path, comps_data)
      compiled_completeness.to_excel("plandb.sioslab.com/backend/sheets/compiled_completeness.xlsx")
      
      
      
      # remember do makesql with this and then get those tables and upsert those shorter new ones with the current
      # writeSQL(engine, plandata=planets, stdata=stars, orbitfits=orbitfits, orbdata=orbits, pdfs=newpdfs, aliases=None,contrastCurves=contrast_curves,scenarios=scenarios, completeness=completeness)
      scenarios = pd.read_csv("plandb.sioslab.com/cache/scenario_angles.csv")


      temp_writeSQL(diff_engine, plandata=plan_data, stdata=stdata, orbitfits=orbfits, orbdata=orbdata, pdfs=None, aliases=None, contrastCurves=compiled_contrast_curves, scenarios=scenarios, completeness=compiled_completeness)
      
      #get from diff_database
      diff_engine_connection = diff_engine.connect()
      diff_completeness_df, diff_contrast_curves_df, diff_orbitfits_df, diff_orbits_df, diff_pdfs_df, diff_planets_df, diff_scenarios_df, diff_stars_df = get_all_from_db(diff_engine_connection)
      
      #get from old_database
      old_completeness_df, old_contrast_curves_df, old_orbitfits_df, old_orbits_df, old_pdfs_df, old_planets_df, old_scenarios_df, old_stars_df = get_all_from_db(connection)
      
      
      
      #merge with old, compare each
      
      # upsert planets
      # Have to do name, because indices don't match, logic applies down unless otherwise in comment
      print("merging planets")
      merged_planets = upsert_dataframe(old_planets_df, diff_planets_df, "pl_name")
      
      
      # upsert completeness
      print("supposed to merge completeness")
      # TODO, this is highly likely to be wrong, forced to used indices here, because only unique, unfortunately incorrect, must base it upsert on the varying indices later
      # merged_completeness = upsert_dataframe(old_completeness_df, diff_completeness_df, "completeness_id")
      
      # upsert stars
      print("merging stars")
      merged_stars = upsert_dataframe(old_stars_df, diff_stars_df, "st_name")
      
      # upsert orbitfits
      print("merging oribt fits")
      # merged_orbitfits = upsert_dataframe(old_orbitfits_df, diff_orbitfits_df, "pl_name")
      
      # upsert orbits
      # print("merging orbits")
      # merged_orbits = upsert_dataframe(old_orbits_df, diff_orbits_df, "pl_name")
      
      # upsert contrast curves
      # TODO, fix the column name, for unique one later?
      print("merging curves")
      merged_contrast_curves = upsert_dataframe(old_contrast_curves_df, diff_contrast_curves_df, "r_lamD")
      
      
      # # upsert pdfs
      # print("merging pdfs")
      # merged_pdfs = upsert_dataframe(old_pdfs_df, diff_pdfs_df, "Name")
      
      # No need to upsert scenarios, as it's updated locally
      
      #write back to original database with new values,
      # TODO: optionally, store old database in a different database for archive
      # orbit fits wrong
      print("Merging")
      final_writeSQL(new_engine, merged_planets, merged_stars, None, None, None, aliases=None, contrastCurves=None, scenarios= scenarios, completeness=None)    
      
      print("Done")