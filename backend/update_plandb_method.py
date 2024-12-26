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


# TODO: make method

#Always keep cache false, because it essentially does nothing if it's on and updates based on month
cache = False
datestr = Time.now().datetime.strftime("%Y-%m")

# TODO need to pass in db? (read from seperately and pass in, or read in here, prob read in here, with no parameters, just update())
# TODO test if you can just pass in sios_engine name and new_engine name as the same to overwrite original, if not add that functionality
def updateDatabase(user : str, password : str, sios_engine_name : str, diff_engine_name : str, new_engine_name: str):

  # Setup SQL and MySQL engines
    
  #TODO Change these when necessary
  sios_engine = create_engine('mysql+pymysql://'+user+':'+password+'@localhost/' + sios_engine_name,echo=True)
  diff_sios_engine = create_engine('mysql+pymysql://'+user+':'+password+'@localhost/' + diff_engine_name,echo=True)
  new_engine = create_engine('mysql+pymysql://'+user+':'+password+'@localhost/' + new_engine_name,echo=True)

  with sios_engine.connect() as connection:
    
    # Get ipac data
    print("Getting IPAC Data")

    old_ipac_data, new_ipac_data = get_store_ipac()
    
    new_ipac_data.to_excel("plandb.sioslab.com/backend/sheets/new_ipac_data.xlsx")
    old_ipac_data.to_excel("plandb.sioslab.com/backend/sheets/old_ipac_data.xlsx")
        
    print(f"New IPAC: {new_ipac_data}")
    print(f"Old IPAC: {old_ipac_data}")
    
    
    #TODO: be able to insert custom ipac data, for test, possibly using flag
    #TODO: Test new row differences, because currently its zero, make test data that is slightly different (similar to ipac data)
    print("calculating row differences")
    change_ipac_df = get_ipac_differences(old_ipac_data, new_ipac_data)
    change_ipac_df.to_excel("plandb.sioslab.com/backend/sheets/change_ipac.xlsx")
    print(f"Changed: {change_ipac_df}")   
      
    # for entry in log: 
    #   print(f"Reason: {entry['reason']}")
    #   print(f"Description: {entry['description']}") 
    #   print("Details:", entry['details']) 
    #   print("-" * 40)   
    
    if change_ipac_df.empty:
      print("No changes detected, zero rows have been updated/added")
      sys.exit() 
            
    # get photodict
    photdict_path = Path(f'plandb.sioslab.com/cache/update_photdict_2022-05.p')
    infile="plandb.sioslab.com/backend/allphotdata_2015.npz"
    photdict = get_photo_data(photdict_path, infile, cache)
    
    print(photdict)

    # get bandzip
    bandzip_path = Path(f'plandb.sioslab.com/cache/update_bandzip_{datestr}.p')
    bandzip = get_bandzip(bandzip_path, cache)
    
    print(bandzip)
    
    # get orbdata, orbfits
    print("Generating orbdata and orbfits")
    orbdata_path = Path(f'plandb.sioslab.com/cache/update_orbdata_{datestr}.p')
    orbfits_path = Path(f'plandb.sioslab.com/cache/update_orbfits_{datestr}.p')
    orbdata, orbfits = get_orbdata(orbdata_path, orbfits_path, change_ipac_df, bandzip, photdict, cache)
    
    # orbdata.to_excel("plandb.sioslab.com/backend/sheets/orbata.xlsx")
    # orbdata.to_excel("plandb.sioslab.com/backend/sheets/orbfits.xlsx")
    
    print(orbdata)
    
    # get ephemeris
    ephemeris_orbdata_path = Path(f'plandb.sioslab.com/cache/update_ephemeris_orbdata_{datestr}.p')
    ephemeris_orbfits_path = Path(f'plandb.sioslab.com/cache/update_ephemeris_orbfits_{datestr}.p')
    ephemeris_orbitfits, ephemeris_orbdata = get_ephemerisdata(ephemeris_orbdata_path, ephemeris_orbfits_path, change_ipac_df, orbfits, orbdata, bandzip, photdict, cache)
    ephemeris_orbitfits.to_excel("plandb.sioslab.com/backend/sheets/ephemeris_orbfits.xlsx")
        
    # get quadrature
    print("Quadrature")
    quadrature_data_path = Path(f'plandb.sioslab.com/cache/update_quadrature_data_{datestr}.p')
    quadrature_data = get_quadrature(quadrature_data_path, ephemeris_orbitfits, bandzip, photdict, cache)
    quadrature_data.to_excel("plandb.sioslab.com/backend/sheets/quadrature_data.xlsx")
    
    contr_data_path = Path(f'plandb.sioslab.com/cache/update_contr_data_{datestr}.p')
    exosims_json = 'plandb.sioslab.com/ci_perf_exosims.json'
    
    contr_data = get_contrastness(contr_data_path, exosims_json, quadrature_data, cache)
    contr_data.to_excel("plandb.sioslab.com/backend/sheets/contr_data.xlsx")
    
    comps_path = Path(f'plandb.sioslab.com/cache/update_comps_{datestr}.p')
    compdict_path = Path(f'plandb.sioslab.com/cache/update_compdict_{datestr}.p')
    comps_data_path = Path(f'plandb.sioslab.com/cache/update_comps_data_{datestr}.p')
        
        
    comps, compdict, comps_data = get_completeness(comps_path, compdict_path, comps_data_path, contr_data, bandzip, photdict, exosims_json, cache)
    comps.to_excel("plandb.sioslab.com/backend/sheets/comps.xlsx")
    comps_data.to_excel("plandb.sioslab.com/backend/sheets/comps_data.xlsx")
    #None for compdict, as its dictionary
    
    plandata_path = Path(f'plandb.sioslab.com/cache/update_plandata_{datestr}.p')
    stdata_path = Path(f'plandb.sioslab.com/cache/update_stdata_{datestr}.p')
    table_orbitfits_path = Path(f'plandb.sioslab.com/cache/update_table_orbitfits_{datestr}.p')
    
    # Orbitfits got updated, maybe change to new var
    plan_data, stdata, orbitfits = get_generated_tables(plandata_path, stdata_path, table_orbitfits_path, change_ipac_df, quadrature_data, comps_data, cache)
    plan_data.to_excel("plandb.sioslab.com/backend/sheets/plandata.xlsx")
    stdata.to_excel("plandb.sioslab.com/backend/sheets/stdata.xlsx")
    orbitfits.to_excel('plandb.sioslab.com/backend/sheets/later_orbitfits.xlsx')
        
    
    # Do compileContrastness and compile contrastness
    # Look into and possibly remove the - to _
    contr_curvs2_path = Path(f'plandb.sioslab.com/cache/cont_curvs2_{datestr.replace("-", "_")}')
    compiled_contr_curvs_path = Path(f'plandb.sioslab.com/cache/cont_curvs_{datestr.replace("-", "_")}')
    # compiled_contrast_curves, newpdfs = get_compiled_contrast(compiled_contr_curvs_path, stdata, comps_data, change_ipac_df, contr_curvs2_path)
    # compiled_contrast_curves = compileContrastCurves(stdata, compiled_contr_curvs_path)
    # compiled_contrast_curves.to_excel("plandb.sioslab.com/backend/sheets/compiled_contrast_curves.xlsx")
    # With current code, since get_compiled_contrast isnt fully working with new pdfs, new pdfs should be empty
    # newpdfs.to_excel("plandb.sioslab.com/backend/sheets/newpdfs.xlsx")
    
    
    # compile completeness
    compiled_completeness_path = Path(f"plandb.sioslab.com/cache/compiled_completeness_{datestr}.p")
    compiled_completeness = get_compiled_completeness(compiled_completeness_path, comps_data)
    compiled_completeness.to_excel("plandb.sioslab.com/backend/sheets/compiled_completeness.xlsx")
    
    
    # MakeSQL for the temporary database, creates diff engine, -> upsert diff engine with current engine
    scenarios = pd.read_csv("plandb.sioslab.com/cache/scenario_angles.csv")  
    # writeSQL(engine, plandata=planets, stdata=stars, orbitfits=orbitfits, orbdata=orbits, pdfs=newpdfs, aliases=None,contrastCurves=contrast_curves,scenarios=scenarios, completeness=completeness)
    
    # TODO: is there unwanted side effects of using temp_writeSQL instead of actual writeSQL
    temp_writeSQL(diff_sios_engine, plandata=plan_data, stdata=stdata, orbitfits=orbitfits, orbdata=orbdata, pdfs=None, aliases=None, contrastCurves=None, scenarios=scenarios, completeness=compiled_completeness)

    # Get from diff_database
    diff_engine_connection = diff_sios_engine.connect()
    diff_completeness_df, diff_contrast_curves_df, diff_orbitfits_df, diff_orbits_df, diff_pdfs_df, diff_planets_df, diff_scenarios_df, diff_stars_df = get_all_from_db(diff_engine_connection)
    
    # Get from old_database
    old_completeness_df, old_contrast_curves_df, old_orbitfits_df, old_orbits_df, old_pdfs_df, old_planets_df, old_scenarios_df, old_stars_df = get_all_from_db(connection)
      
    # Upsert planets
    # Planets don't have pl_id, they only have pl_name which is indexed after
    print("Merging planets")
    merged_planets = upsert_general(old_planets_df, diff_planets_df, "pl_name")

    
    # Upsert completeness
    # print("Merging completeness")
    
    # print(old_completeness_df, diff_completeness_df)
    # # Merge based on pl_id, foreign key relation, base on foreign key relation, might do the same for rest
    # # TODO, this is highly likely to be wrong, forced to used indices here, because only unique, unfortunately incorrect, must base it upsert on the varying indices later
    # # TODO: Iterate through each pl_id, since these are base off pl_id, correspond with pl_name, and then reset those completeness rows for that pl (remove and then add the new ones that were)
    # merged_completeness = upsert_general(old_completeness_df, diff_completeness_df, "pl_id")
    
    # Upsert stars
    # TODO Star differences are based off that the planets table has foreign key st_id, therefore, to properly update stars, must go through planets, see what has been updated, and then go down through those planets, and their stars, and then if that planet has changed, update that star
    # Same logic as with planets
    print("Merging stars")
    merged_stars = upsert_general(old_stars_df, diff_stars_df, "st_name")  
    
    print('old orbitfits')
    print(old_orbitfits_df)
    print("diff orbitfits")
    print(diff_orbitfits_df)
    old_orbitfits_df.to_excel('plandb.sioslab.com/backend/sheets/original_orbitfits.xlsx')
    diff_orbitfits_df.to_excel('plandb.sioslab.com/backend/sheets/diff_orbitfits.xlsx')
    # print("diff between diff and old, this should be nothing")
    # print(diff_orbits_df)
    
    # For these later upserts, only way to properly upsert is to detect the change from the earlier value, like the difference in planets from ipac, and then categorize the change as a result from upated information or new information. If new information (new planet), just add but if updated information, going to have to track down the previous. Maybe it's possible for me to locally store database versions, so it can be quickly updated based on path of changes that happened  
    # Upsert orbitfits
    # TODO: maybe the error is here
    print("Merging orbit fits")
    merged_orbitfits = upsert_general(old_orbitfits_df, diff_orbitfits_df, "pl_id")
        
    # TODO: compile/calc contrast curves still needs fixing 
    # TODO: make a seperate database for todays data, the old data, and test by comparing todays data to old data + update
    
    # Upsert orbits
    print("Merging orbits")
    merged_orbits = upsert_general(old_orbits_df, diff_orbits_df, "pl_id")
    
    # Upsert contrast curvess  
    print("Merging curves")
    merged_contrast_curves = upsert_general(old_contrast_curves_df, diff_contrast_curves_df, "st_id") 
    
    
    # Upsert pdfs 
    # TODO: same as orbit fits
    # print("Merging pdfs")
    # merged_pdfs = upsert_df(old_pdfs_df, diff_pdfs_df, "Name")
    
    # No need to upsert scenarios, as it's updated locally
    
    # write back to original database with new values,
    # TODO: optionally, store old database in a different database for archive
    print("Merging and final write")
    write_update_SQL(new_engine, merged_planets, merged_stars, merged_orbitfits, merged_orbits, pdfs=None, aliases=None, contrastCurves= None, scenarios=None, completeness=None)    
    
    print("Done")
    
    # TODO: Print all total changes
    # TODO: Correct the merges/compiles with no changes, ending up outputting an empty table, (account for table upserts with no changes) (Handle no new updates case). For example, final completeness would be empty if there are no changes to completeness in the update.
    
    
  """_summary_
  """
def updateDatabaseTest(user : str, password : str, sios_engine_name : str, diff_engine_name : str, new_engine_name: str):

  # Setup SQL and MySQL engines
    
  #TODO Change these when necessary
  sios_engine = create_engine('mysql+pymysql://'+user+':'+password+'@localhost/' + sios_engine_name,echo=True)
  diff_sios_engine = create_engine('mysql+pymysql://'+user+':'+password+'@localhost/' + diff_engine_name,echo=True)
  new_engine = create_engine('mysql+pymysql://'+user+':'+password+'@localhost/' + new_engine_name,echo=True)

  with sios_engine.connect() as connection:
    
    # Get ipac data
    print("Getting IPAC Data")


    ################ This is changed from regular updateDatabase(), so that we use archived matching ipac data ##################
    # old_ipac_data, new_ipac_data = get_store_ipac()

    old_ipac_data_path = Path(f'plandb.sioslab.com/cache/data_cache_2022-05.p')
    old_ipac_data = pd.read_pickle(old_ipac_data_path)
      
    new_ipac_data = getIPACdata()
    
    new_ipac_data.to_excel("plandb.sioslab.com/backend/sheets/new_ipac_data.xlsx")
    old_ipac_data.to_excel("plandb.sioslab.com/backend/sheets/old_ipac_data.xlsx")
        
    print(f"New IPAC: {new_ipac_data}")
    print(f"Old IPAC: {old_ipac_data}")
    
    
    #TODO: be able to insert custom ipac data, for test, possibly using flag
    #TODO: Test new row differences, because currently its zero, make test data that is slightly different (similar to ipac data)
    print("calculating row differences")
    change_ipac_df = get_ipac_differences(old_ipac_data, new_ipac_data, ["pl_name"])
    change_ipac_df.to_excel("plandb.sioslab.com/backend/sheets/change_ipac.xlsx")
    print(f"Changed: {change_ipac_df}")   
      
    # for entry in log: 
    #   print(f"Reason: {entry['reason']}")
    #   print(f"Description: {entry['description']}") 
    #   print("Details:", entry['details']) 
    #   print("-" * 40)   
    
    # input1 = input("test")
    
    if change_ipac_df.empty:
      print("No changes detected, zero rows have been updated/added")
      sys.exit() 
            
    # get photodict
    photdict_path = Path(f'plandb.sioslab.com/cache/update_photdict_2022-05.p')
    infile="plandb.sioslab.com/backend/allphotdata_2015.npz"
    photdict = get_photo_data(photdict_path, infile, cache)
    
    print(photdict)

    # get bandzip
    bandzip_path = Path(f'plandb.sioslab.com/cache/update_bandzip_{datestr}.p')
    bandzip = get_bandzip(bandzip_path, cache)
    
    print(bandzip)
    
    # get orbdata, orbfits
    print("Generating orbdata and orbfits")
    orbdata_path = Path(f'plandb.sioslab.com/cache/update_orbdata_{datestr}.p')
    orbfits_path = Path(f'plandb.sioslab.com/cache/update_orbfits_{datestr}.p')
    orbdata, orbfits = get_orbdata(orbdata_path, orbfits_path, change_ipac_df, bandzip, photdict, cache)
    
    # orbdata.to_excel("plandb.sioslab.com/backend/sheets/orbata.xlsx")
    # orbdata.to_excel("plandb.sioslab.com/backend/sheets/orbfits.xlsx")
    
    print(orbdata)
    
    # get ephemeris
    ephemeris_orbdata_path = Path(f'plandb.sioslab.com/cache/update_ephemeris_orbdata_{datestr}.p')
    ephemeris_orbfits_path = Path(f'plandb.sioslab.com/cache/update_ephemeris_orbfits_{datestr}.p')
    ephemeris_orbitfits, ephemeris_orbdata = get_ephemerisdata(ephemeris_orbdata_path, ephemeris_orbfits_path, change_ipac_df, orbfits, orbdata, bandzip, photdict, cache)
    ephemeris_orbitfits.to_excel("plandb.sioslab.com/backend/sheets/ephemeris_orbfits.xlsx")
        
    # get quadrature
    print("Quadrature")
    quadrature_data_path = Path(f'plandb.sioslab.com/cache/update_quadrature_data_{datestr}.p')
    quadrature_data = get_quadrature(quadrature_data_path, ephemeris_orbitfits, bandzip, photdict, cache)
    quadrature_data.to_excel("plandb.sioslab.com/backend/sheets/quadrature_data.xlsx")
    
    contr_data_path = Path(f'plandb.sioslab.com/cache/update_contr_data_{datestr}.p')
    exosims_json = 'plandb.sioslab.com/ci_perf_exosims.json'
    
    contr_data = get_contrastness(contr_data_path, exosims_json, quadrature_data, cache)
    contr_data.to_excel("plandb.sioslab.com/backend/sheets/contr_data.xlsx")
    
    comps_path = Path(f'plandb.sioslab.com/cache/update_comps_{datestr}.p')
    compdict_path = Path(f'plandb.sioslab.com/cache/update_compdict_{datestr}.p')
    comps_data_path = Path(f'plandb.sioslab.com/cache/update_comps_data_{datestr}.p')
        
        
    comps, compdict, comps_data = get_completeness(comps_path, compdict_path, comps_data_path, contr_data, bandzip, photdict, exosims_json, cache)
    comps.to_excel("plandb.sioslab.com/backend/sheets/comps.xlsx")
    comps_data.to_excel("plandb.sioslab.com/backend/sheets/comps_data.xlsx")
    #None for compdict, as its dictionary
    
    plandata_path = Path(f'plandb.sioslab.com/cache/update_plandata_{datestr}.p')
    stdata_path = Path(f'plandb.sioslab.com/cache/update_stdata_{datestr}.p')
    table_orbitfits_path = Path(f'plandb.sioslab.com/cache/update_table_orbitfits_{datestr}.p')
    
    # Orbitfits got updated, maybe change to new var
    # TODO maybe needs all ipac data, to update plandata here
    plan_data, stdata, orbitfits = get_generated_tables(plandata_path, stdata_path, table_orbitfits_path, new_ipac_data, quadrature_data, comps_data, cache)
    plan_data.to_excel("plandb.sioslab.com/backend/sheets/plandata.xlsx")
    stdata.to_excel("plandb.sioslab.com/backend/sheets/stdata.xlsx")
    orbitfits.to_excel('plandb.sioslab.com/backend/sheets/later_orbitfits.xlsx')
        
    
    # Do compileContrastness and compile contrastness
    # Look into and possibly remove the - to _
    contr_curvs2_path = Path(f'plandb.sioslab.com/cache/cont_curvs2_{datestr.replace("-", "_")}')
    compiled_contr_curvs_path = Path(f'plandb.sioslab.com/cache/cont_curvs_{datestr.replace("-", "_")}')
    # compiled_contrast_curves, newpdfs = get_compiled_contrast(compiled_contr_curvs_path, stdata, comps_data, change_ipac_df, contr_curvs2_path)
    # compiled_contrast_curves = compileContrastCurves(stdata, compiled_contr_curvs_path)
    # compiled_contrast_curves.to_excel("plandb.sioslab.com/backend/sheets/compiled_contrast_curves.xlsx")
    # With current code, since get_compiled_contrast isnt fully working with new pdfs, new pdfs should be empty
    # newpdfs.to_excel("plandb.sioslab.com/backend/sheets/newpdfs.xlsx")
    
    
    # compile completeness
    compiled_completeness_path = Path(f"plandb.sioslab.com/cache/compiled_completeness_{datestr}.p")
    compiled_completeness = get_compiled_completeness(compiled_completeness_path, comps_data)
    compiled_completeness.to_excel("plandb.sioslab.com/backend/sheets/compiled_completeness.xlsx")
    
    
    # MakeSQL for the temporary database, creates diff engine, -> upsert diff engine with current engine
    scenarios = pd.read_csv("plandb.sioslab.com/cache/scenario_angles.csv")  
    # writeSQL(engine, plandata=planets, stdata=stars, orbitfits=orbitfits, orbdata=orbits, pdfs=newpdfs, aliases=None,contrastCurves=contrast_curves,scenarios=scenarios, completeness=completeness)
    
    # TODO: is there unwanted side effects of using temp_writeSQL instead of actual writeSQL
    temp_writeSQL(diff_sios_engine, plandata=plan_data, stdata=stdata, orbitfits=orbitfits, orbdata=orbdata, pdfs=None, aliases=None, contrastCurves=None, scenarios=scenarios, completeness=compiled_completeness)

    # Get from diff_database
    diff_engine_connection = diff_sios_engine.connect()
    diff_completeness_df, diff_contrast_curves_df, diff_orbitfits_df, diff_orbits_df, diff_pdfs_df, diff_planets_df, diff_scenarios_df, diff_stars_df = get_all_from_db(diff_engine_connection)
    
    # Get from old_database
    old_completeness_df, old_contrast_curves_df, old_orbitfits_df, old_orbits_df, old_pdfs_df, old_planets_df, old_scenarios_df, old_stars_df = get_all_from_db(connection)
      
    # Upsert planets
    # Planets don't have pl_id, they only have pl_name which is indexed after
    print("Merging planets")
    merged_planets = upsert_general(old_planets_df, diff_planets_df, "pl_name")

    
    # Upsert completeness
    # print("Merging completeness")
    
    # print(old_completeness_df, diff_completeness_df)
    # # Merge based on pl_id, foreign key relation, base on foreign key relation, might do the same for rest
    # # TODO, this is highly likely to be wrong, forced to used indices here, because only unique, unfortunately incorrect, must base it upsert on the varying indices later
    # # TODO: Iterate through each pl_id, since these are base off pl_id, correspond with pl_name, and then reset those completeness rows for that pl (remove and then add the new ones that were)
    # merged_completeness = upsert_general(old_completeness_df, diff_completeness_df, "pl_id")
    
    # Upsert stars
    # TODO Star differences are based off that the planets table has foreign key st_id, therefore, to properly update stars, must go through planets, see what has been updated, and then go down through those planets, and their stars, and then if that planet has changed, update that star
    # Same logic as with planets
    print("Merging stars")
    merged_stars = upsert_general(old_stars_df, diff_stars_df, "st_name")  
    
    print('old orbitfits')
    print(old_orbitfits_df)
    print("diff orbitfits")
    print(diff_orbitfits_df)
    old_orbitfits_df.to_excel('plandb.sioslab.com/backend/sheets/original_orbitfits.xlsx')
    diff_orbitfits_df.to_excel('plandb.sioslab.com/backend/sheets/diff_orbitfits.xlsx')
    # print("diff between diff and old, this should be nothing")
    # print(diff_orbits_df)
    
    # For these later upserts, only way to properly upsert is to detect the change from the earlier value, like the difference in planets from ipac, and then categorize the change as a result from upated information or new information. If new information (new planet), just add but if updated information, going to have to track down the previous. Maybe it's possible for me to locally store database versions, so it can be quickly updated based on path of changes that happened  
    # Upsert orbitfits
    # TODO: maybe the error is here
    print("Merging orbit fits")
    merged_orbitfits = upsert_general(old_orbitfits_df, diff_orbitfits_df, "pl_id")
        
    # TODO: compile/calc contrast curves still needs fixing 
    # TODO: make a seperate database for todays data, the old data, and test by comparing todays data to old data + update
    
    # Upsert orbits
    print("Merging orbits")
    merged_orbits = upsert_general(old_orbits_df, diff_orbits_df, "pl_id")
    
    # Upsert contrast curvess  
    print("Merging curves")
    merged_contrast_curves = upsert_general(old_contrast_curves_df, diff_contrast_curves_df, "st_id") 
    
    
    # Upsert pdfs 
    # TODO: same as orbit fits
    # print("Merging pdfs")
    # merged_pdfs = upsert_df(old_pdfs_df, diff_pdfs_df, "Name")
    
    # No need to upsert scenarios, as it's updated locally
    
    # write back to original database with new values,
    # TODO: optionally, store old database in a different database for archive
    print("Merging and final write")
    write_update_SQL(new_engine, merged_planets, merged_stars, merged_orbitfits, merged_orbits, pdfs=None, aliases=None, contrastCurves= None, scenarios=None, completeness=None)    
    
    
    #Re-index
    
    print("Done")
    
    # TODO: Print all total changes
    # TODO: Correct the merges/compiles with no changes, ending up outputting an empty table, (account for table upserts with no changes) (Handle no new updates case). For example, final completeness would be empty if there are no changes to completeness in the update.
    
    
