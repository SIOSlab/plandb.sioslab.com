from EXOSIMS.util.getExoplanetArchive import queryExoplanetArchive
import numpy as np
import getpass
import keyring
from sqlalchemy import create_engine


psmeta = queryExoplanetArchive(r"select * from TAP_SCHEMA.columns where table_name = 'ps'")
pscpmeta = queryExoplanetArchive(r"select * from TAP_SCHEMA.columns where table_name = 'pscomppars'")

meta = psmeta.append(pscpmeta)
meta.drop_duplicates(subset=["column_name", "datatype", "description"],inplace=True)
meta.reset_index(drop=True, inplace=True)

username='root'
passwd = keyring.get_password('tphon_sql_login', username)
engine = create_engine('mysql+pymysql://'+username+':'+passwd+'@127.0.0.1/plandb', echo=False, encoding='utf8', pool_pre_ping=True)

tables = np.hstack(engine.execute("Show tables").fetchall())

outdicts = {}
for t in tables:
    cols = np.vstack(engine.execute("SHOW COLUMNS FROM {}".format(t)).fetchall())[:,0]
    cols = cols[cols != "index"]
    outdicts[t] = {'Column':cols}

