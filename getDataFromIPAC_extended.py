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

    # planets_of_int = pandas.read_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\output.csv")
    # names = planets_of_int[["pl_name"]]

    regex = re.compile(r"(<a.*f>)|(</a>)")
    regex_year = re.compile(r"[A-Za-z'#\.&; \-]")

    names = original_data[["pl_name"]]
    names = names.values
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
    cols2 = ["mpl_reflink", "mpl_name", "mpl_orbper", "mpl_orbsmax",
             "mpl_orbeccen", "mpl_orbincl", "mpl_bmassj", "mpl_orbtper",  "mpl_orblper", "mpl_radj",
             "mst_mass", "ref_author", "publication_year", "best_data"]
    cols = list(extended.columns.values)

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

    extended_arr = np.append(extended.values, np.zeros((length, 1)), axis=1)
    extended_arr = np.append(extended_arr, np.zeros((length, 1)), axis=1)

    # author_col = np.chararray((1, length))
    author_col = list()
    date_col = np.zeros(length)

    while n < length:
        # extended.at[n, "mpl_reflink"] = regex.sub("", extended.at[n, "mpl_reflink"])
        # extended.at[n, "pub_year"] = regex_year.sub("", extended.at[n, "mpl_reflink"])

        extended_arr[n][num_cols] = regex.sub("", extended_arr[n][author])
        extended_arr[n][num_cols+1] = int(regex_year.sub("", extended_arr[n][num_cols]), 10)
        # In order to catch issues where an extra number is added to the front of the year due to weird formatting
        if extended_arr[n][num_cols+1] > 3000:
            extended_arr[n][num_cols + 1] = extended_arr[n][num_cols+1] % 10000

        author_col.append(regex.sub("", extended_arr[n][author]))
        # print(int(regex_year.sub("", author_col[n], 10)))
        date_col[n] = int(regex_year.sub("", author_col[n]), 10)
        n = n + 1

    n = 0

    refs = extended["mpl_reflink"].values
    best_data = np.zeros(length)
    extended["ref_author"] = pandas.Series(author_col, index=extended.index)
    extended["publication_year"] = pandas.Series(date_col, index=extended.index)
    extended["best_data"] = pandas.Series(best_data, index=extended.index)
    extended["mpl_orbperreflink"] = pandas.Series(refs, index=extended.index)
    extended["mpl_orbsmaxreflink"] = pandas.Series(refs, index=extended.index)
    extended["mpl_orbeccenreflink"] = pandas.Series(refs, index=extended.index)
    extended["mpl_bmassreflink"] = pandas.Series(refs, index=extended.index)
    extended["mpl_radreflink"] = pandas.Series(refs, index=extended.index)
    extended["mst_massreflink"] = pandas.Series(refs, index=extended.index)

    while n < len(names):
        planet_rows = extended.loc[extended["mpl_name"] == names[n][0]]
        print(names[n])

        sorted_rows = planet_rows.sort_values(by=["publication_year"], axis=0, ascending=False)
        good_idx = sorted_rows.index[0]
        good_lvl = 0
        for index, row in sorted_rows.iterrows():
            base_need = (not pandas.isnull(row["mpl_orbsmax"]) or not pandas.isnull(row["mpl_orbper"])) and \
                        (not pandas.isnull(row["mpl_bmassj"]) or not pandas.isnull(row["mpl_radj"]))

            # Has everything
            if good_lvl < 4 and (base_need
                                 and not pandas.isnull(row["mpl_orbeccen"]) and not pandas.isnull(row["mpl_orbtper"])
                                 and not pandas.isnull(row["mpl_orblper"]) and not pandas.isnull(row["mpl_orbincl"])):
                good_idx = index
                good_lvl = 4
                break

            # Has everything except inclination
            if good_lvl < 3 and (base_need
                                 and not pandas.isnull(row["mpl_orbeccen"]) and not pandas.isnull(row["mpl_orbtper"])
                                 and not pandas.isnull(row["mpl_orblper"])):
                good_idx = index
                good_lvl = 3

            # Has either periapsis time or argument of pariapsis
            elif good_lvl < 2 and (base_need
                                   and not pandas.isnull(row["mpl_orbeccen"]) and (not pandas.isnull(row["mpl_orbtper"])
                                                                          or not pandas.isnull(row["mpl_orblper"]))):
                good_idx = index
                good_lvl = 2
            # Has eccentricity
            elif good_lvl < 1 and (base_need
                                   and not pandas.isnull(row["mpl_orbeccen"])):
                good_idx = index
                good_lvl = 1
            # 1st doesn't have basic info
            elif index == good_idx and not base_need:
                good_idx = -1
            # Previous row needed to be replaced
            elif good_idx == -1 and base_need:
                good_idx = index
                good_lvl = 1

        if good_idx == -1:
            good_idx = sorted_rows.index[0]

        extended.at[good_idx, "best_data"] = 1
        n = n + 1

    extended_data = extended

    # extended_limited.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\extended_orbit_data_lim_ref.csv")

    colmap = {k: k[1:] if (k.startswith('mst_') | k.startswith('mpl_')) else k for k in extended_data.keys()}
    extended_data = extended_data.rename(columns=colmap)

    columns_to_replace = ["pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_orbperlim", "pl_orbsmax",
                          "pl_orbsmaxerr1", "pl_orbsmaxerr2", "pl_orbsmaxlim", "pl_orbeccen",
                          "pl_orbeccenerr1", "pl_orbeccenerr2", "pl_orbeccenlim", "pl_orbtper",
                          "pl_orbtpererr1", "pl_orbtpererr2", "pl_orbtperlim", "pl_orblper",
                          "pl_orblpererr1", "pl_orblpererr2", "pl_orblperlim", "pl_bmassj",
                          "pl_bmassjerr1", "pl_bmassjerr2", "pl_bmassjlim", "pl_radj", "pl_radjerr1",
                          "pl_radjerr2", "pl_radjlim", "pl_orbincl", "pl_orbinclerr1", "pl_orbinclerr2",
                          "pl_orbincllim", "pl_orbperreflink", "pl_orbsmaxreflink", "pl_orbeccenreflink",
                          "pl_bmassreflink", "pl_radreflink", "pl_bmassprov"]

    n = 0
    while n < len(names):
        row_extended = extended_data.loc[(extended_data["best_data"] == 1) & (extended_data["pl_name"] == names[n][0])]
        row_original = original_data.loc[(original_data["pl_name"] == names[n][0])]
        idx = row_original.index
        if int(row_extended["pl_def"]) == 0:
            final_columns = list(columns_to_replace)
            if not np.isnan(row_extended["st_mass"].values):
                final_columns.append("st_mass")
                final_columns.append("st_masserr1")
                final_columns.append("st_masserr2")
                final_columns.append("st_masslim")
                final_columns.append("st_massreflink")

            row_extended = row_extended.drop(columns=["ref_author", "publication_year", "best_data"])
            row_extended.index = idx
            row_extended_replace = row_extended[final_columns]

            original_data.loc[idx, final_columns] = original_data.loc[idx].replace(original_data.loc[idx], np.nan)
            original_data.update(row_extended_replace)
        n = n + 1
    return original_data

#
# #grab the data
# query = """https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=compositepars&select=*&format=csv"""
# r = requests.get(query)
# data = pandas.read_csv(StringIO(r.content))
#
# query2 = """https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=*&format=csv"""
# r2 = requests.get(query2)
# data2 = pandas.read_csv(StringIO(r2.content))
#
# #strip leading 'f' on data colnames
# colmap = {k:k[1:] if (k.startswith('fst_') | k.startswith('fpl_')) else k for k in data.keys()}
# data = data.rename(columns=colmap)
# #sma, eccen, metallicity cols were renamed so name them back for merge
# data = data.rename(columns={'pl_smax':'pl_orbsmax',
#                             'pl_smaxerr1':'pl_orbsmaxerr1',
#                             'pl_smaxerr2':'pl_orbsmaxerr2',
#                             'pl_smaxlim':'pl_orbsmaxlim',
#                             'pl_smaxreflink':'pl_orbsmaxreflink',
#                             'pl_eccen':'pl_orbeccen',
#                             'pl_eccenerr1':'pl_orbeccenerr1',
#                             'pl_eccenerr2':'pl_orbeccenerr2',
#                             'pl_eccenlim':'pl_orbeccenlim',
#                             'pl_eccenreflink':'pl_orbeccenreflink',
#                             'st_met':'st_metfe',
#                             'st_meterr1':'st_metfeerr1',
#                             'st_meterr2':'st_metfeerr2',
#                             'st_metreflink':'st_metfereflink',
#                             'st_metlim':'st_metfelim',
#                             })
#
# #sort by planet name
# data = data.sort_values(by=['pl_name']).reset_index(drop=True)
# data2 = data2.sort_values(by=['pl_name']).reset_index(drop=True)
#
# #merge data sets
# data = data.combine_first(data2)
#
#
# # data.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\original_data_refs.csv")
# new_data = substitute_data(data)
# new_data.to_csv("C:\\Users\\NathanelKinzly\\github\\orbits\\plandb.sioslab.com\\modified_data_refs_2.csv")
