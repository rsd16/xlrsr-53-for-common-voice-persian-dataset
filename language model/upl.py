'''with open("fa_wiki.arpa", "r", encoding='utf-8') as read_file, open("5gram_fa_wiki.arpa", "w", encoding='utf-8') as write_file:
	has_added_eos = False
	for line in read_file:
		if not has_added_eos and "ngram 1=" in line:
			count=line.strip().split("=")[-1]
			write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
		elif not has_added_eos and "<s>" in line:
			write_file.write(line)
			write_file.write(line.replace("<s>", "</s>"))
			has_added_eos = True
		else:
			write_file.write(line)'''

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained('ahabahab8/wav2vec2-large-xlsr-53-fine-tuned-farsi')

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
print(sorted_vocab_dict)

from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()), kenlm_model_path='5gram_fa_wiki.arpa')

from transformers import Wav2Vec2ProcessorWithLM

processor_with_lm = Wav2Vec2ProcessorWithLM(feature_extractor=processor.feature_extractor,
    										tokenizer=processor.tokenizer, decoder=decoder)

processor_with_lm.save_pretrained('nope/wav2vec2-large-xlsr-53-fine-tuned-farsi-5gram')