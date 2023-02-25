import os
from model import KenlmModel
from datasets import load_dataset

verbose = False
root_dir = 'data/codeparrot/github-code-clean'
kenlm_model_dir = os.path.join(root_dir, 'lm_sp/All.arpa.bin')
sp_model_dir = os.path.join(root_dir, 'lm_sp/All.sp.model')
langs = ['C', 'Python', 'Java', 'PHP', 'C++']

python_model = KenlmModel(kenlm_model_dir, sp_model_dir)
dataset = load_dataset(
	'codeparrot/github-code-clean',
	split='train',
	streaming=True,
	languages=langs
)
max_num_samples = 10000
current_count = 0
perplexity_by_lang = {lang : 0 for lang in langs}
count = {lang : 0 for lang in langs}
for sample in dataset:
	code = sample['code']
	lang = sample['language']
	perp = python_model.get_perplexity(code)

	if verbose:
		print (f"Perplexity: {perp}, Language: {sample['language']}")
	perplexity_by_lang[lang] += perp
	count[lang] += 1

	current_count += 1

	if current_count >= max_num_samples:
		break

for lang in perplexity_by_lang:
	avg_perp = perplexity_by_lang[lang] / count[lang]
	print (f"Perplexity of {lang} : {avg_perp}")