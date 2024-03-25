import pandas as pd
from sqlalchemy import create_engine

# MySQL connection parameters
username = 'root'
password = ''
host = 'localhost'
port = '3306'
database_name = 'test'

# Create a DataFrame with sample data
data = {'id': [1, 2, 3, 4, 5],
        'column1': ['value1', 'value2', 'value3', 'value4', 'value5']}

df = pd.DataFrame(data)

# Create SQLAlchemy engine to connect to MySQL
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database_name}')

# Write DataFrame to MySQL table
table_name = 'test'
df.to_sql(table_name, con=engine, if_exists='replace', index=False)

# Confirm creation by fetching data from the table
df_from_sql = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)
print(df_from_sql)


