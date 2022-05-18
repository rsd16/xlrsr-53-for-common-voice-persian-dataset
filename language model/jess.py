import kenlm

model = kenlm.Model('fa_wiki.binary')
print("score: ", model.score('کشور ایران شهر تهران', bos=True, eos=True))
print("score: ", model.score('کشور تهران شهر ایران', bos=True, eos=True))
# score:  -11.683658599853516
# score:  -15.572178840637207