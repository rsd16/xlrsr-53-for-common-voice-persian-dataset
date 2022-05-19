'''
Written by Alireza Rashidi.
Python 3.8.x.
This file does prediction with the model, “wav2vec2-large-xlsr-53”, on Mozilla Common Voice dataset for Persian language,
mainly with PyTorch and HuggingFace packages.
'''


import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, load_metric


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch['path'])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)
    
    batch['speech'] = speech_array
    
    return batch

def predict(batch):
    features = processor(batch['speech'], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors='pt', padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits 

    pred_ids = torch.argmax(logits, dim=-1)

    batch['predicted'] = processor.batch_decode(pred_ids)
    
    return batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = Wav2Vec2Processor.from_pretrained('ahabahab8/wav2vec2-large-xlsr-53-fine-tuned-farsi')

model = Wav2Vec2ForCTC.from_pretrained('ahabahab8/wav2vec2-large-xlsr-53-fine-tuned-farsi').to(device)

dataset = load_dataset('csv', data_files={'test': 'new_test.csv'}, delimiter='\t')['test']
dataset = dataset.map(speech_file_to_array_fn)
result = dataset.map(predict, batched=True, batch_size=4)

wer = load_metric('wer')
print('WER: {:.2f}'.format(100 * wer.compute(predictions=result['predicted'], references=result['sentence'])))

max_items = np.random.randint(0, len(result), 20).tolist()

for i in max_items:
    reference, predicted =  result['sentence'][i], result['predicted'][i]
    print('reference:', reference)
    print('predicted:', predicted)
    print('---')
