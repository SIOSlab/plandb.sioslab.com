"""Column definition Script"""

from EXOSIMS.util.getExoplanetArchive import queryExoplanetArchive
import numpy as np
import getpass
import keyring
from sqlalchemy import create_engine
import re
import pandas

# First we get all of NEXSCI's definitions:
psmeta = queryExoplanetArchive(
    r"select * from TAP_SCHEMA.columns where table_name = 'ps'"
)
pscpmeta = queryExoplanetArchive(
    r"select * from TAP_SCHEMA.columns where table_name = 'pscomppars'"
)

meta = psmeta.append(pscpmeta)
meta.drop_duplicates(subset=["column_name", "datatype", "description"], inplace=True)
meta.reset_index(drop=True, inplace=True)
meta.loc[meta["column_name"] == "hostname", "column_name"] = "st_name"
for r in meta.itertuples():
    k = r.column_name
    if k.startswith("st_") and (k != "st_id" and k != "st_name" and k != "st_radv"):
        meta.at[r.Index, "column_name"] = k[3:]


# Next we get our tables:
username = "root"
passwd = keyring.get_password("tphon_sql_login", username)
engine = create_engine(
    "mysql+pymysql://" + username + ":" + passwd + "@127.0.0.1/plandb",
    echo=False,
    encoding="utf8",
    pool_pre_ping=True,
)

tables = np.hstack(engine.execute("Show tables").fetchall())

outdicts = {}
for t in tables:
    cols = np.vstack(engine.execute("SHOW COLUMNS FROM {}".format(t)).fetchall())[:, 0]
    cols = cols[cols != "index"]
    outdicts[t] = {"Column": cols}

keydescs = {
    "dMag": "Delta Mag",
    "radius": "Orbital radius",
    "pPhi": "Geometric Albedo times phase function",
}

clouddict = {
    "000": "No Clouds",
    "001": "f_sed = 0.01",
    "003": "f_sed = 0.03",
    "010": "f_sed = 0.1",
    "030": "f_sed = 0.3",
    "100": "f_sed = 1.0",
    "300": "f_sed = 3.0",
    "600": "f_sed = 6.0",
    "med": "median",
    "max": "maximum",
    "min": "minimum",
}

photcols = tuple(["{}".format(k) for k in keydescs.keys()])
quadcols = tuple(["quad_{}".format(k) for k in keydescs.keys()])

# Fill in column names wherever we can find them in NEXSCI or patter match
for t in tables:
    desc = np.zeros(outdicts[t]["Column"].shape, dtype=object) * np.nan
    for j, c in enumerate(outdicts[t]["Column"]):
        if c in meta["column_name"].values:
            desc[j] = meta.loc[meta["column_name"] == c, "description"].values[0]
        elif c.startswith(quadcols):
            tmp = re.match("quad_(\S+)_(\S{3})C{0,1}_(\d{3})NM", c)
            if tmp:
                tmp = tmp.groups()
                desc[j] = "{} {} at {} nm at quadrature".format(
                    keydescs[tmp[0]], clouddict[tmp[1]], tmp[2]
                )
        elif c.startswith(photcols):
            tmp = re.match("(\S+)_(\S{3})C{0,1}_(\d{3})NM", c)
            if tmp:
                tmp = tmp.groups()
                desc[j] = "{} {} at {} nm".format(
                    keydescs[tmp[0]], clouddict[tmp[1]], tmp[2]
                )
    outdicts[t]["Description"] = desc

with pandas.ExcelWriter("col_descriptions.xlsx") as writer:
    for t in tables:
        out = pandas.DataFrame(outdicts[t])
        out.to_excel(writer, sheet_name=t, index=False)

"""
At this point, we have a partially filled out col_descriptions spreadsheet with a tab
for each table.  Now you have to go and manually fill in the blanks.  Oh, and do a
global search/replace for % to 'percent'.
"""


#####
# Now we can go back and update our table schemas to include all comments
p = re.compile("`(\S+)`[\s\S]+")

for t in tables:
    coldefs = pandas.read_excel("col_descriptions.xlsx", sheet_name=t)

    res = engine.execute("show create table %s" % t).fetchall()
    res = res[0]["Create Table"]
    res = res.split("\n")

    keys = []
    defs = []
    for r in res:
        r = r.strip().strip(",")
        if "COMMENT" in r:
            continue
        m = p.match(r)
        if m:
            keys.append(m.groups()[0])
            defs.append(r)

    for k, d in zip(keys, defs):
        if k not in coldefs["Column"].values:
            continue
        comm = """ALTER TABLE `{}` CHANGE `{}` {} COMMENT "{}";""".format(
            t, k, d, coldefs.loc[coldefs["Column"] == k, "Description"].values[0]
        )
        print(comm)
        r = engine.execute(comm)
