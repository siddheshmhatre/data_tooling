import os
import pandas as pd
from model import KenlmModel

root_dir = 'data/codeparrot/github-code-clean'
kenlm_model_dir = os.path.join(root_dir, 'lm_sp/All.arpa.bin')
sp_model_dir = os.path.join(root_dir, 'lm_sp/All.sp.model')
kenlm_model = KenlmModel(kenlm_model_dir, sp_model_dir)

perp_threshold = 280

filename = '/admin/home-siddhesh1793/part-00000-8c813e13-6509-4cf4-9261-5c641a04f6d1-c000.snappy.parquet'
df = pd.read_parquet(filename)

df['perplexity'] = df['text'].apply(lambda x : kenlm_model.get_perplexity(x))
df.to_parquet('extracted_text_perp.parquet')