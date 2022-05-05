import os
import re
#from typing import List, Dict, Tuple
import pandas as pd
from scipy.io import wavfile
from pythainlp.tokenize import word_tokenize
#from spell_correction import correct_sentence
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from pydub import AudioSegment
from pythainlp.tokenize import word_tokenize, syllable_tokenize
from datasets import load_dataset, load_from_disk, load_metric
import hazm
import string
import torch
import os
#os.environ['CUDA_VISIBLE_DEVICES']='2, 3'

torch.cuda.empty_cache()

#print(torch.cuda.memory_summary(device=None, abbreviated=False))

print(torch.cuda.is_available())




_normalizer = hazm.Normalizer()

chars_to_ignore = [
    ",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
    "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬",'ٔ', ",", "?", 
    ".", "!", "-", ";", ":",'"',"“", "%", "‘", "”", "�", "–", "…", "_", "”", '“', '„',
    'ā', 'š',
#     "ء",
]

# In case of farsi
chars_to_ignore = chars_to_ignore + list(string.ascii_lowercase + string.digits)

chars_to_mapping = {
    'ك': 'ک', 'دِ': 'د', 'بِ': 'ب', 'زِ': 'ز', 'ذِ': 'ذ', 'شِ': 'ش', 'سِ': 'س', 'ى': 'ی',
    'ي': 'ی', 'أ': 'ا', 'ؤ': 'و', "ے": "ی", "ۀ": "ه", "ﭘ": "پ", "ﮐ": "ک", "ﯽ": "ی",
    "ﺎ": "ا", "ﺑ": "ب", "ﺘ": "ت", "ﺧ": "خ", "ﺩ": "د", "ﺱ": "س", "ﻀ": "ض", "ﻌ": "ع",
    "ﻟ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻪ": "ه", "ﻮ": "و", 'ﺍ': "ا", 'ة': "ه",
    'ﯾ': "ی", 'ﯿ': "ی", 'ﺒ': "ب", 'ﺖ': "ت", 'ﺪ': "د", 'ﺮ': "ر", 'ﺴ': "س", 'ﺷ': "ش",
    'ﺸ': "ش", 'ﻋ': "ع", 'ﻤ': "م", 'ﻥ': "ن", 'ﻧ': "ن", 'ﻭ': "و", 'ﺭ': "ر", "ﮔ": "گ",
        
    # "ها": "  ها", "ئ": "ی",
    "۱۴ام": "۱۴ ام",
        
    "a": " ای ", "b": " بی ", "c": " سی ", "d": " دی ", "e": " ایی ", "f": " اف ",
    "g": " جی ", "h": " اچ ", "i": " آی ", "j": " جی ", "k": " کی ", "l": " ال ",
    "m": " ام ", "n": " ان ", "o": " او ", "p": " پی ", "q": " کیو ", "r": " آر ",
    "s": " اس ", "t": " تی ", "u": " یو ", "v": " وی ", "w": " دبلیو ", "x": " اکس ",
    "y": " وای ", "z": " زد ",
    "\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ", "\ufeff": " ",
}


def multiple_replace(text, chars_to_mapping):
    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))

def remove_special_characters(text, chars_to_ignore_regex):
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
    return text

def normalizer(text, chars_to_ignore=chars_to_ignore, chars_to_mapping=chars_to_mapping):
    chars_to_ignore_regex = f"""[{"".join(chars_to_ignore)}]"""
    text = text.lower().strip()

    text = _normalizer.normalize(text)
    text = multiple_replace(text, chars_to_mapping)
    text = remove_special_characters(text, chars_to_ignore_regex)
    text = re.sub(" +", " ", text)
    text = text.replace('آ', 'ا')
    _text = []
    for word in text.split():
        try:
            word = int(word)
            _text.append(words(word))
        except:
            _text.append(word)
            
    text = " ".join(_text) + " "
    text = text.strip()

    if not len(text) > 0:
        return None
    
    return text + " "

data_dir = "cv-corpus-9.0-2022-04-27/fa"

#########################################################################################################

train = pd.read_csv(f"{data_dir}/train.tsv", sep="\t")
train["path"] = data_dir + "/clips/" + train["path"]
print(f"Step 0: {len(train)}")

train["status"] = train["path"].apply(lambda path: True if os.path.exists(path) else None)
train = train.dropna(subset=["path"])
train = train.drop("status", 1)
print(f"Step 1: {len(train)}")

train["sentence"] = train["sentence"].apply(lambda t: normalizer(t))
train = train.dropna(subset=["sentence"])
print(f"Step 2: {len(train)}")
print(train['sentence'])

train = train.reset_index(drop=True)
print(train.head())

train = train[["path", "sentence"]][:50]
train.to_csv("new_train.csv", sep="\t", encoding="utf-8", index=False)

#########################################################################################################

dev = pd.read_csv(f"{data_dir}/dev.tsv", sep="\t")
dev["path"] = data_dir + "/clips/" + dev["path"]
print(f"Step 0: {len(dev)}")

dev["status"] = dev["path"].apply(lambda path: True if os.path.exists(path) else None)
dev = dev.dropna(subset=["path"])
dev = dev.drop("status", 1)
print(f"Step 1: {len(dev)}")

dev["sentence"] = dev["sentence"].apply(lambda t: normalizer(t))
dev = dev.dropna(subset=["sentence"])
print(f"Step 2: {len(dev)}")
print(dev['sentence'])

dev = dev.reset_index(drop=True)
print(dev.head())

dev = dev[["path", "sentence"]][:50]
dev.to_csv("new_dev.csv", sep="\t", encoding="utf-8", index=False)

#########################################################################################################

test = pd.read_csv(f"{data_dir}/test.tsv", sep="\t")
test["path"] = data_dir + "/clips/" + test["path"]
print(f"Step 0: {len(test)}")

test["status"] = test["path"].apply(lambda path: True if os.path.exists(path) else None)
test = test.dropna(subset=["path"])
test = test.drop("status", 1)
print(f"Step 1: {len(test)}")

test["sentence"] = test["sentence"].apply(lambda t: normalizer(t))
test = test.dropna(subset=["sentence"])
print(f"Step 2: {len(test)}")
print(test['sentence'])

test = test.reset_index(drop=True)
print(test.head())

test = test[["path", "sentence"]][:50]
test.to_csv("new_test.csv", sep="\t", encoding="utf-8", index=False)

##############################################################################################################

common_voice_train = load_dataset("csv", data_files={"train": "new_train.csv"}, delimiter="\t")["train"]
common_voice_dev = load_dataset("csv", data_files={"dev": "new_dev.csv"}, delimiter="\t")["dev"]
common_voice_test = load_dataset("csv", data_files={"test": "new_test.csv"}, delimiter="\t")["test"]

print(common_voice_train)
print(common_voice_dev)
print(common_voice_test)

from datasets import ClassLabel
import random
import pandas as pd
#from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    print(df.head())

show_random_elements(common_voice_train.remove_columns(["path"]), num_examples=20)

def normalizer(batch, chars_to_ignore=chars_to_ignore, chars_to_mapping=chars_to_mapping):

    chars_to_ignore_regex = f"""[{"".join(chars_to_ignore)}]"""
    text = batch["sentence"].lower().strip()

    text = _normalizer.normalize(text)
    text = multiple_replace(text, chars_to_mapping)
    text = remove_special_characters(text, chars_to_ignore_regex)
    text = re.sub(" +", " ", text)
    _text = []
    for word in text.split():
        try:
            word = int(word)
            _text.append(words(word))
        except:
            _text.append(word)
            
    text = " ".join(_text) + " "
    text = text.strip()

    if not len(text) > 0:
        return None

    #text = generate_N_grams(text)
    
    batch["sentence"] = text
    
    return batch







print(common_voice_train[0]["sentence"])
print(common_voice_dev[0]["sentence"])
print(common_voice_test[0]["sentence"])

common_voice_train = common_voice_train.map(normalizer, fn_kwargs={"chars_to_ignore": chars_to_ignore, "chars_to_mapping": chars_to_mapping})
common_voice_dev = common_voice_dev.map(normalizer, fn_kwargs={"chars_to_ignore": chars_to_ignore, "chars_to_mapping": chars_to_mapping})
common_voice_test = common_voice_test.map(normalizer, fn_kwargs={"chars_to_ignore": chars_to_ignore, "chars_to_mapping": chars_to_mapping})

print(common_voice_train[0]["sentence"])
print(common_voice_dev[0]["sentence"])
print(common_voice_test[0]["sentence"])

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=4, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_dev = common_voice_dev.map(extract_all_chars, batched=True, batch_size=4, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=4, keep_in_memory=True, remove_columns=common_voice_test.column_names)

vocab_list = list(sorted(set(vocab_train["vocab"][0]) | set(vocab_dev["vocab"][0]) | set(vocab_test["vocab"][0])))
vocab_list = [vocab for vocab in vocab_list if vocab not in [" ", "\u0307"]]
print(len(vocab_list))
print(vocab_list)

vocab_list = list(sorted(set(vocab_train["vocab"][0]) | set(vocab_dev["vocab"][0]) | set(vocab_test["vocab"][0])))
vocab_list = [vocab for vocab in vocab_list if vocab not in [" ", "\u0307"]]
print(len(vocab_list))
print(vocab_list)

special_vocab = ["<pad>", "<s>", "</s>", "<unk>", "|"]
vocab_dict = {v: k for k, v in enumerate(special_vocab + vocab_list)}
print(len(vocab_dict))
print(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
	json.dump(vocab_dict, vocab_file)

from transformers.trainer_utils import get_last_checkpoint

save_dir = "model checkpoints/"

last_checkpoint = None
if os.path.exists(save_dir):
    last_checkpoint = get_last_checkpoint(save_dir)

print(last_checkpoint if last_checkpoint else str(None))

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json", 
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        max_length=31
    )

text = "از مهمونداری کنار بکشم"
print(" ".join(tokenizer.tokenize(text)))
print(tokenizer.decode(tokenizer.encode(text)))

from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

if len(processor.tokenizer.get_vocab()) == len(processor.tokenizer):
	print(len(processor.tokenizer))

if not os.path.exists(save_dir):
	print("Saving ...")
	processor.save_pretrained(save_dir)
	print("Saved!")

'''
def generate_N_grams(batch, ngram=5):
    print(batch)
    words = [word for word in text.split(" ")]
    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    return ans

common_voice_train = common_voice_train.map(generate_N_grams, remove_columns=common_voice_train.column_names, batch_size=4, batched=True)
common_voice_dev = common_voice_dev.map(generate_N_grams, remove_columns=common_voice_dev.column_names, batch_size=4, batched=True)
common_voice_test = common_voice_test.map(generate_N_grams, remove_columns=common_voice_test.column_names, batch_size=4, batched=True)
'''

import torchaudio
import librosa


target_sampling_rate = 16_000

def speech_file_to_array_fn(batch):
	speech_array, sampling_rate = torchaudio.load(batch["path"])
	speech_array = speech_array.squeeze().numpy()
	speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, target_sampling_rate)

	batch["speech"] = speech_array
	batch["sampling_rate"] = target_sampling_rate
	batch["duration_in_seconds"] = len(batch["speech"]) / target_sampling_rate
	batch["target_text"] = batch["sentence"]
	return batch

common_voice_train = common_voice_train.map(speech_file_to_array_fn, remove_columns=common_voice_train.column_names)
common_voice_dev = common_voice_dev.map(speech_file_to_array_fn, remove_columns=common_voice_dev.column_names)
common_voice_test = common_voice_test.map(speech_file_to_array_fn, remove_columns=common_voice_test.column_names)

print(common_voice_train[0]["sampling_rate"])
print(common_voice_test[0]["sampling_rate"])


min_duration_in_seconds = 5.0
max_duration_in_seconds = 10.0

def filter_by_max_duration(batch):
	return min_duration_in_seconds <= batch["duration_in_seconds"] <= max_duration_in_seconds

print(f"Split sizes [BEFORE]: {len(common_voice_train)} train and {len(common_voice_test)} validation.")



_common_voice_train = common_voice_train.filter(filter_by_max_duration)
_common_voice_dev = common_voice_dev
_common_voice_test = common_voice_test
# _common_voice_test = common_voice_test.filter(filter_by_max_duration, num_proc=4)

print(f"Split sizes [AFTER]: {len(_common_voice_train)} train and {len(_common_voice_test)} validation.")

# check that all files have the correct sampling rate
def prepare_dataset(batch):
	assert (
        len(set(batch["sampling_rate"])) == 1), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

	batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

	with processor.as_target_processor():
		batch["labels"] = processor(batch["target_text"]).input_ids

	return batch

_common_voice_train = _common_voice_train.map(prepare_dataset, remove_columns=_common_voice_train.column_names, batch_size=4, batched=True)
_common_voice_dev = _common_voice_dev.map(prepare_dataset, remove_columns=_common_voice_dev.column_names, batch_size=4, batched=True)
_common_voice_test = _common_voice_test.map(prepare_dataset, remove_columns=_common_voice_test.column_names, batch_size=4, batched=True)

_common_voice_train.set_format(type='torch', columns=['input_values', 'labels'])
_common_voice_dev.set_format(type='torch', columns=['input_values', 'labels'])
_common_voice_test.set_format(type='torch', columns=['input_values', 'labels'])


###############################################################################################################

#torch.cuda.empty_cache()

#print(torch.cuda.memory_summary(device=None, abbreviated=False))

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True
	max_length: Optional[int] = None
	max_length_labels: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	pad_to_multiple_of_labels: Optional[int] = None

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]

		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
			)
		with self.processor.as_target_processor():
			labels_batch = self.processor.pad(
				label_features,
				padding=self.padding,
				max_length=self.max_length_labels,
                #max_length=64,
				pad_to_multiple_of=self.pad_to_multiple_of_labels,
				return_tensors="pt",
				)

		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

		batch["labels"] = labels

		return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")

import random


def compute_metrics(pred):
	pred_logits = pred.predictions
	pred_ids = np.argmax(pred_logits, axis=-1)

	pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

	pred_str = processor.batch_decode(pred_ids)

	label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

	if isinstance(label_str, list):
		if isinstance(pred_str, list) and len(pred_str) == len(label_str):
			for index in random.sample(range(len(label_str)), 3):
				print(f'reference: "{label_str[index]}"')
				print(f'predicted: "{pred_str[index]}"')
		else:
			for index in random.sample(range(len(label_str)), 3):
				print(f'reference: "{label_str[index]}"')
				print(f'predicted: "{pred_str}"')

	wer = wer_metric.compute(predictions=pred_str, references=label_str)

	return {"wer": wer}


from transformers import Wav2Vec2ForCTC, Wav2Vec2Config

configuration = Wav2Vec2Config(hidden_size=256, num_hidden_layers=6, num_attention_heads=6, intermediate_size=1024)

model_args ={}

print('haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
print(len(processor.tokenizer.get_vocab()))
print('haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53" if not last_checkpoint else last_checkpoint, 
    #model_name_or_path if not last_checkpoint else last_checkpoint,
    attention_dropout=0.1,
    #hidden_size=256,
    #num_hidden_layers=8,
    #num_attention_heads=2,
    #intermediate_size=256,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    ctc_zero_infinity=True,
    bos_token_id=processor.tokenizer.bos_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer.get_vocab()),
    no_repeat_ngram_size=5
)

model.config = configuration


model.freeze_feature_extractor()

#model.to(torch.device("cuda"))

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=save_dir,
    group_by_length=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=0.5,
    #fp16=True,
    #save_steps=10,
    #eval_steps=10,
    #logging_steps=10,
    learning_rate=1e-4,
    #warmup_steps=500,
    #save_total_limit=2,
    no_cuda=True
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=_common_voice_train,
    eval_dataset=_common_voice_test,
    tokenizer=processor.feature_extractor,
)

torch.cuda.empty_cache()

train_result = trainer.train()


metrics = train_result.metrics
max_train_samples = len(_common_voice_train)
metrics["train_samples"] = min(max_train_samples, len(_common_voice_train))

trainer.save_model()

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()







import numpy as np
import pandas as pd

import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, load_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")