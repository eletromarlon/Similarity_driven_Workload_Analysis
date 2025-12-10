import pandas as pd

splits = {'train': 'synthetic_text_to_sql_train.snappy.parquet', 'test': 'synthetic_text_to_sql_test.snappy.parquet'}
df = pd.read_parquet("hf://datasets/gretelai/synthetic_text_to_sql/" + splits["train"])


print(df.head())