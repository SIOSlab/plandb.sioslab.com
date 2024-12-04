import pandas as pd
from sqlalchemy import create_engine

before_engine = create_engine('mysql+pymysql://'+"andrewchiu"+':'+"Password123!"+'@localhost/TestBeforeUpdate',echo=True)


today_df = pd.read_sql(before_engine)
print(today_df)