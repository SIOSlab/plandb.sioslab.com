import requests
import pandas
from StringIO import StringIO
import pickle
import astropy.units as u
import astropy.constants as const
#import EXOSIMS.PlanetPhysicalModel.Forecaster
from sqlalchemy import create_engine
import getpass,keyring
import numpy as np
import re


def substitute_data(original_data):
    query_ext = """https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exomultpars&select=*&format=csv"""
    r_ext = requests.get(query_ext)
    data_ext = pandas.read_csv(StringIO(r_ext.content))

    planets_of_int = pandas.read_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\output.csv")
    names = planets_of_int[["pl_name"]]

    regex = re.compile(r"(<a.*f>)|(</a>)")
    regex_year = re.compile(r"[A-Za-z\.&; \-]")

    names = names.values
    print(names[0][0])
    extended = data_ext[data_ext["mpl_name"] == names[0][0]]
    # extended = extended.reset_index()
    # extended.at[0, "mpl_reflink"] = regex.sub("", extended.at[0, "mpl_reflink"])
    # print(extended.loc[:, "mpl_reflink"])
    # print(extended)

    n = 1
    # Gets all planets that are named in output.txt
    while n < len(names):
        extended = pandas.concat([extended, data_ext[data_ext["mpl_name"] == names[n][0]]])
        n = n + 1


    extended = extended.reset_index(drop=True)

    # cols2 = ["mpl_reflink", "mpl_name", "mpl_orbper", "mpl_orbpererr1", "mpl_orbpererr2", "mpl_orbsmax", "mpl_orbsmaxerr1",
    #                            "mpl_orbsmaxerr2", "mpl_orbeccen", "mpl_orbeccenerr1", "mpl_orbeccenerr2", "mpl_orbincl",
    #                            "mpl_orbinclerr1", "mpl_orbinclerr2", "mpl_bmassj", "mpl_bmassjerr1", "mpl_bmassjerr2",
    #                            "mpl_orbtper", "mpl_orbtpererr1", "mpl_orbtpererr2", "mpl_orblper", "mpl_orblpererr1",
    #                            "mpl_orblpererr2"]
    cols = list(extended.columns.values)

    print(cols)

    smax = cols.index("mpl_orbsmax")
    per = cols.index("mpl_orbper")
    eccen = cols.index("mpl_orbeccen")
    periaps_time = cols.index("mpl_orbtper")
    argument_periaps = cols.index("mpl_orblper")
    pl_name = cols.index("mpl_name")
    default_column = cols.index("mpl_def")
    author = cols.index("mpl_reflink")

    # extended = extended[cols]

    length = len(extended.values)
    num_cols = len(extended.columns)

    cols.append("ref_author")
    cols.append("publication_year")
    cols.append("best_data")
    n = 0

    extended_arr = np.append(np.asarray(extended), np.zeros((length, 1)), axis=1)
    extended_arr = np.append(np.asarray(extended_arr), np.zeros((length, 1)), axis=1)

    while n < length:
        # extended.at[n, "mpl_reflink"] = regex.sub("", extended.at[n, "mpl_reflink"])
        # extended.at[n, "pub_year"] = regex_year.sub("", extended.at[n, "mpl_reflink"])

        extended_arr[n][num_cols] = regex.sub("", extended_arr[n][author])
        extended_arr[n][num_cols+1] = int(regex_year.sub("", extended_arr[n][num_cols]), 10)
        n = n + 1

    n = 0

    extended_arr = np.append(extended_arr, np.zeros((length, 1)), axis=1)
    extended2 = pandas.DataFrame(extended_arr)

    while n < len(names):
        planet_rows = extended2.loc[extended2[pl_name] == names[n][0]]

        sorted_rows = planet_rows.sort_values(by=[num_cols+1], axis=0, ascending=False)
        sorted_arr = sorted_rows.values
        i = 0
        good_idx = sorted_rows.index[0]
        good_lvl = 0
        for index, row in sorted_rows.iterrows():
            # Has everything
            if good_lvl < 3 and (not pandas.isnull(row[per])and not pandas.isnull(row[smax])
                                 and not pandas.isnull(row[eccen]) and not pandas.isnull(row[periaps_time])
                                 and not pandas.isnull(row[argument_periaps])):
                good_idx = index
                good_lvl = 3
                break
            # Has either periapsis time or argument of pariapsis
            elif good_lvl < 2 and (not pandas.isnull(row[per]) and not pandas.isnull(row[smax])
                                   and not pandas.isnull(row[eccen]) and (not pandas.isnull(row[periaps_time])
                                                                          or not pandas.isnull(row[argument_periaps]))):
                good_idx = index
                good_lvl = 2
            # Has eccentricity
            elif good_lvl < 1 and (not pandas.isnull(row[per]) and not pandas.isnull(row[smax])
                                   and not pandas.isnull(row[eccen])):
                good_idx = index
                good_lvl = 1
            # 1st doesn't have basic info
            elif index == good_idx and (pandas.isnull(row[per]) or pandas.isnull(row[smax])):
                good_idx = -1
            # Previous row needed to be replaced
            elif good_idx == -1 and (not pandas.isnull(row[per]) and not pandas.isnull(row[smax])):
                good_idx = index
                good_lvl = 1

        if good_idx == -1:
            good_idx = sorted_rows.index[0]

        extended2.at[good_idx, num_cols + 2] = 1
        n = n + 1

    extended2.columns = cols
    extended_data = extended2
    colmap = {k: k[1:] if (k.startswith('mst_') | k.startswith('mpl_')) else k for k in extended_data.keys()}
    extended_data = extended_data.rename(columns=colmap)
    # extended_data = extended_data.rename(columns={# 'pl_smaxreflink': 'pl_orbsmaxreflink',
    #                                               # 'pl_eccenreflink': 'pl_orbeccenreflink', #*doesn't exist
    #                                               # 'st_metreflink': 'st_metfereflink', #*doesn't exist
    #                                               })

    n = 0
    # print(extended_data.columns)
    while n < len(names):
        row_extended = extended_data.loc[(extended_data["best_data"] == 1) & (extended_data["pl_name"] == names[n][0])]
        row_original = original_data.loc[(original_data["pl_name"] == names[n][0])]
        idx = row_original.index
        if int(row_extended["pl_def"]) == 0:
            row_extended = row_extended.drop(columns=["ref_author", "publication_year", "best_data"])
            row_extended.index = idx

            original_data.loc[idx] = original_data.loc[idx].replace(original_data.loc[idx], np.nan)
            original_data.update(row_extended)
            # print(idx)
        n = n + 1
    return original_data


#grab the data
query = """https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=compositepars&select=*&format=csv"""
r = requests.get(query)
data = pandas.read_csv(StringIO(r.content))

query2 = """https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=*&format=csv"""
r2 = requests.get(query2)
data2 = pandas.read_csv(StringIO(r2.content))

#strip leading 'f' on data colnames
colmap = {k:k[1:] if (k.startswith('fst_') | k.startswith('fpl_')) else k for k in data.keys()}
data = data.rename(columns=colmap)
#sma, eccen, metallicity cols were renamed so name them back for merge
data = data.rename(columns={'pl_smax':'pl_orbsmax',
                            'pl_smaxerr1':'pl_orbsmaxerr1',
                            'pl_smaxerr2':'pl_orbsmaxerr2',
                            'pl_smaxlim':'pl_orbsmaxlim',
                            'pl_smaxreflink':'pl_orbsmaxreflink',
                            'pl_eccen':'pl_orbeccen',
                            'pl_eccenerr1':'pl_orbeccenerr1',
                            'pl_eccenerr2':'pl_orbeccenerr2',
                            'pl_eccenlim':'pl_orbeccenlim',
                            'pl_eccenreflink':'pl_orbeccenreflink',
                            'st_met':'st_metfe',
                            'st_meterr1':'st_metfeerr1',
                            'st_meterr2':'st_metfeerr2',
                            'st_metreflink':'st_metfereflink',
                            'st_metlim':'st_metfelim',
                            })

#sort by planet name
data = data.sort_values(by=['pl_name']).reset_index(drop=True)
data2 = data2.sort_values(by=['pl_name']).reset_index(drop=True)

#merge data sets
data = data.combine_first(data2)


# data.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\original_data.csv")
new_data = substitute_data(data)
# new_data.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\modified_data.csv")

print(new_data['st_mass'].values)

i = 0
while i < len(new_data.values):
    print(new_data['st_mass'].values[i])
    print(np.isnan(new_data['st_mass'].values)[i])
    i = i + 1

# i = 0
# while i < len(new_data.values):
#     row = new_data.loc[i]
#     print(row.loc["pl_name"])
#     print(np.isnan(row.loc["st_mass"]))
#     i = i + 1


# extended_final = extended[["mpl_reflink", "mpl_publ_date", "mpl_name", "mpl_orbper", "mpl_orbsmax", "mpl_orbeccen",
#                            "mpl_orbincl", "mpl_bmassj", "mpl_orbtper", "mpl_orblper", "mpl_eqt", "mpl_insol"]]


# extended2.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\extended_orbit_data.csv")

# pickle.dump(data_ext, open("test.p", "wb"))
# print(data_ext)

import sqlalchemy.types
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import select

# plannames = planets_of_int['mpl_name'].values
#
# namemxchar = np.array([len(n) for n in plannames]).max()
#
# engine = create_engine('mysql+pymysql://ds264@127.0.0.1/dsavrans_plandb',echo=False)
# conn = engine.connect()
# meta = MetaData(bind=None)
# table = Table('KnownPlanets', meta, autoload=True, autoload_with=engine)
# stmt = select([table])
#
# results = conn.execute(stmt).fetchall()
# results_2 = np.asarray(results)
# results_3 = pandas.DataFrame(results_2)

# for item in results_2:
#     results_3.__add__(item)



# results_3.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\known_planets.csv")

# data.to_sql('KnownPlanets', engine, chunksize=100, if_exists='replace',
#             dtype={'pl_name': sqlalchemy.types.String(namemxchar),
#                    'pl_hostname': sqlalchemy.types.String(namemxchar - 2),
#                    'pl_letter': sqlalchemy.types.CHAR(1)})
#
# result = engine.execute("ALTER TABLE KnownPlanets ENGINE=InnoDB")
# result = engine.execute("ALTER TABLE KnownPlanets ADD INDEX (pl_name)")
# result = engine.execute("ALTER TABLE KnownPlanets ADD INDEX (pl_hostname)")
#
# orbdata.to_sql('PlanetOrbits', engine, chunksize=100, if_exists='replace',
#                dtype={'Name': sqlalchemy.types.String(namemxchar)})
# result = engine.execute("ALTER TABLE PlanetOrbits ENGINE=InnoDB")
# result = engine.execute("ALTER TABLE PlanetOrbits ADD INDEX (Name)")
# result = engine.execute(
#     "ALTER TABLE PlanetOrbits ADD FOREIGN KEY (Name) REFERENCES KnownPlanets(pl_name) ON DELETE NO ACTION ON UPDATE NO ACTION");
