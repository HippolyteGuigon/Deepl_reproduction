import os 
from google.cloud import bigquery

all_tables_id='''SELECT table_id
FROM `deepl-reprodution.processed_data.__TABLES__`
'''

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

client = bigquery.Client()

query_job = client.query(all_tables_id)

results = query_job.result()

for row in results:
    print(row.table_id)
