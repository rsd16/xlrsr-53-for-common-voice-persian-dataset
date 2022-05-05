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

# !wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-9.0-2022-04-27/cv-corpus-9.0-2022-04-27-fa.tar.gz
# !tar -xvf cv-corpus-9.0-2022-04-27-fa.tar.gz --no-same-owner

'''
def get_full_path(cv_root, path):
    """Get full path from `path` instance in cv data"""
    f_path = f"{cv_root}/fa/clips/{path}"
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"File `{f_path}` does not exists")
    return f_path

def get_char(texts):
    """Get unique char from list of documents"""
    return sorted(set([char for sent in texts for char in sent]))

def get_audio_len(audio_path):
    """Get audio duration in second"""
    #sr, wav = wavfile.read(audio_path.replace("mp3", "wav").replace("clips", "wav"))
    #return len(wav) / sr
    #print(audio_path)
    audio = AudioSegment.from_mp3(audio_path)
    return len(audio) / 1000  # pydub duration works in ms

def sec_to_hour(second):
    """Convert second to XXH:YYm:ZZs format"""
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"{int(hour)}H:{int(minute)}m:{second:.2f}s"

cv_root = "cv-corpus-9.0-2022-04-27"

train = pd.read_csv(f"{cv_root}/fa/train.tsv", delimiter="\t")
train["set"] = "train"
#print(train)

dev = pd.read_csv(f"{cv_root}/fa/dev.tsv", delimiter="\t")
dev["set"] = "dev"
#print(dev)

test = pd.read_csv(f"{cv_root}/fa/test.tsv", delimiter="\t")
test["set"] = "test"
#print(test)

data = pd.read_csv(f"{cv_root}/fa/validated.tsv", delimiter="\t")
#print(data)

command = "sox {mp3_path} -t wav -r {sr} -c 1 -b 16 - |"

skip_rules = [
    r"[a-zA-Z]",
]

# Rules mapping obtained from exploring data
mapping_char = {
    r"!": " ",
    r'"': " ",
    r"'": " ",
    r",": " ",
    r"-": " ",
    r"\.{2,}": " ",
    r"\.$": "",  # full stop at end of sentence
    r"([ก-์])\.([ก-์])": r"\1. \2",  # บจก.XXX -> บจก. XXX
    r":": " ",
    r";": " ",
    r"\?": " ",
    r"‘": " ", 
    r"’": " ",
    r"“": " ", 
    r"”": " ",
    r"~": " ",
    r"—": " ",
    r"\.": " ",
}

# Audio EDA
audios = data["path"].map(lambda path: get_full_path(cv_root=cv_root, path=path)).tolist()
print(len(audios))

audios_len = [get_audio_len(f) for f in tqdm(audios)]

#plt.hist(audios_len, bins=50)
#plt.xlabel("Time (sec)")
#plt.ylabel("Frequency")
#plt.title("File duration distribution")
#plt.show()

print(f"Total: {sec_to_hour(np.sum(audios_len))}")
print(f"Mean: {np.mean(audios_len):.2f}")
print(f"Std.: {np.std(audios_len):.2f}")
print(f"Max: {np.max(audios_len):.2f}")
print(f"Min: {np.min(audios_len):.2f}")

# Transcription EDA
texts = data[["path", "sentence"]]
texts["path"] = texts["path"].map(lambda x: get_full_path(cv_root, x))
texts = texts.values.tolist()
#texts = [correct_sentence(text) for text in tqdm(texts)]
print(len(texts))

s = [print(f"`{c}`, ", end="") for c in get_char([x[-1] for x in texts])]
print(len(s))

##############################################################################################

# Unicodes
persian_alpha_codepoints = '\u0621-\u0628\u062A-\u063A\u0641-\u0642\u0644-\u0648\u064E-\u0651\u0655\u067E\u0686\u0698\u06A9\u06AF\u06BE\u06CC'
persian_num_codepoints = '\u06F0-\u06F9'
arabic_numbers_codepoints = '\u0660-\u0669'
space_codepoints ='\u0020\u2000-\u200F\u2028-\u202F'
additional_arabic_characters_codepoints = '\u0629\u0643\u0649-\u064B\u064D\u06D5'


ss = []
for item in dev['sentence']:
    # remove words between paranthesis and thos paranthesis too.
    item = re.sub(re.compile(r'\([^)]*\)'), '', item)

    #print(item)

    # split with tab and remove nim fasele
    item = item.replace('\u200c', ' ')

    #Remove Hour formats : hh:mm:ss 
    item = re.sub('\d{2}\:\d{2}:\d{2}', '',item)
    item = re.sub('\d{2}\:\d{2}', '',item)
    item = re.sub('\d{1}\:\d{2}', '',item)

    # Remove Date formats : yy(yy):mm(mm):ss(ss)
    item = re.sub(r'[0-9]{2}[\/,:][0-9]{2}[\/,:][0-9]{2,4}', "", item)
    item = re.sub(r'[۰-۹]{2,4}[\/,:][۰-۹]{2,4}[\/,:][۰-۹]{2,4}', "", item)

    # Just remain persian alphabete and numbers
    item = re.sub(r"[^" + persian_alpha_codepoints + persian_num_codepoints + additional_arabic_characters_codepoints+
                    arabic_numbers_codepoints + space_codepoints + '1234567890\n' + "]", "", item)

    # change all kinds of sapce with normal space
    item = re.sub(r"[" +
          space_codepoints+ "]", " ", item)

    # change nim fasele with space
    item = re.sub(r"[\u200c]", " ", item)


    # Remove or Substitude some characters.   ـ
    # این نتوین   ً و ء حفظ میشه
    item = re.sub(r"[" + 'ّ'  + 'ٌ' + 'ـ' + 'َ' + 'ِ' + 'ٕ'  + 'ٍ' + 'ُ' + 'ْ' + "]", '', item)

    # Be careful this editor VSC shows the character in wrong order
    item = re.sub('ؤ', 'و', item)
    item = re.sub('ة', 'ه', item)
    item = re.sub('ك', 'ک', item)
    item = re.sub('ى', 'ی', item)
    item = re.sub('ي', 'ی', item)
    item = re.sub('ە', 'ه', item)
    item = re.sub('ئ', 'ی', item)
    item = re.sub('أ', 'ا', item)
    item = re.sub('إ', 'ا', item)



    # remove multiple spaces with just one space 
    item = re.sub(' +', ' ', item)

    # remove multiple strings from first and last of lines
    item = item.strip()

    # if line is just numbers. ignore it.
    if re.sub(r"[" + space_codepoints + "]", "", item).isnumeric():
        continue

    # # NOTE : THIS PART IS SO Dependent On TEXT TYPE. next Line is for : Leipzig2 => news_2011
    # if 'کلید واژه'  in line:
    #     continue
    # if 'نظرات خوانندگان'  in line:
    #     continue
    # num2word persian
    #line = re.sub(r"(\d+)", lambda x : convert(int(x.group(0))), line)


    # TEST
    item = re.sub(r"[\n]", " ", item)

    ss.append(item)

dev['setn'] = ss

#dev.to_csv('ssr.tsv')

def preprocess_data(example, tok_func = word_tokenize):
    example['sentence'] = ' '.join(tok_func(example['sentence']))
    return example

datasets = load_dataset("ssdf.py", "fa")
print(datasets)

datasets = datasets.map(preprocess_data)
print(datasets)

##############################################################################################################

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

test = test[["path", "sentence"]]
test.to_csv("srx.csv", sep="\t", encoding="utf-8", index=False)
'''

import numpy as np
import pandas as pd

import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, load_metric

#import IPython.display as ipd

model_name_or_path = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(model_name_or_path, device)

processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = processor(
        batch["speech"], 
        sampling_rate=processor.feature_extractor.sampling_rate, 
        return_tensors="pt", 
        padding=True
    )

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits 

    pred_ids = torch.argmax(logits, dim=-1)

    batch["predicted"] = processor.batch_decode(pred_ids)
    return batch


dataset = load_dataset("csv", data_files={"test": "new_test.csv"}, delimiter="\t")["test"]
dataset = dataset.map(speech_file_to_array_fn)
result = dataset.map(predict, batched=True, batch_size=4)

wer = load_metric("wer")
print("WER: {:.2f}".format(100 * wer.compute(predictions=result["predicted"], references=result["sentence"])))

max_items = np.random.randint(0, len(result), 20).tolist()
for i in max_items:
    reference, predicted =  result["sentence"][i], result["predicted"][i]
    print("reference:", reference)
    print("predicted:", predicted)
    print('---')