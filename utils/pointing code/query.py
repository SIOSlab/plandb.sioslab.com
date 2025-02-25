# code used for querying for reference data
# needs starspd.csv to be in current working directory
# starspd.csv has 2 columns with star names and HD numbers
import pandas as pd
from astroquery.simbad import Simbad
import time

simbad = Simbad()
simbad.add_votable_fields("ra", "dec", "pmra", "pmdec", "plx")


starspd = pd.read_csv("starspd.csv")
starshd = starspd["HD#"].to_list()
starsnames = starspd["Name"].to_list()

cols = [
    "MAIN_ID",
    "RA",
    "DEC",
    "RA_PREC",
    "DEC_PREC",
    "COO_ERR_MAJA",
    "COO_ERR_MINA",
    "COO_ERR_ANGLE",
    "COO_QUAL",
    "COO_WAVELENGTH",
    "COO_BIBCODE",
    "RA_2",
    "DEC_2",
    "PMRA",
    "PMDEC",
    "PLX_VALUE",
    "SCRIPT_NUMBER_ID",
]
df = pd.DataFrame(cols).transpose()
df.to_csv("simbad_output.csv", index=False, mode="a", header=False)
for star_num in starshd:
    hname = "HD" + str(star_num)
    result_table = simbad.query_object(hname)
    df = result_table.to_pandas()
    df.to_csv(
        "simbad_output.csv", index=False, mode="a", header=False
    )  # outputs specified fields to csv in current working directory
    print(star_num)
    time.sleep(10)
