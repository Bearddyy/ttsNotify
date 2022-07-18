import os
import torch
import sys

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

text = sys.argv[1]

#Split the text into chunks of size 1000
chunkSize = 1000
chunks = [text[i:i+chunkSize] for i in range(0, len(text), chunkSize)]
chunks = [chunks[0]]

sample_rate = 48000
speaker='en_1'

wowSpeakers = ['en_10', 'en_110', 'en_111', 'en_116', 'en_30','en_48', 'en_52', 'en_65', 'en_71', 'en_94', 'en_97', 'en_98', 'en_101', 'en_103']
veryGoodSpeakers = ['en_25', 'en_41', 'en_48', 'en_51', 'en_52', 'en_62', 'en_63', 'en_64', 'en_65', 'en_67', 'en_68', 'en_71','en_94','en_96','en_97',
'en_98', 'en_101', 'en_103', 'en_110', 'en_111', ]
goodSpeakers = ['en_10','en_12', 'en_17', 'en_22','en_24','en_30','en_32','en_45','en_47',
 'en_56', 'en_59', 'en_61', 'en_61',  'en_70', 'en_82', 'en_84', 'en_85', 'en_86', 'en_90',
 'en_93', 'en_98', 'en_101', 'en_107']

allSpeakers = [f"en_{i}" for i in range(0, 117)]

for speaker in wowSpeakers:
    #run the model on each chunk
    for i,chunk in enumerate(chunks):
        audio_paths = model.save_wav(audio_path=f"{speaker}.wav",
        text=chunk,
        speaker=speaker,
        sample_rate=sample_rate)
