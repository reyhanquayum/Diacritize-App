# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Arabic Speech Corpus"""


import os

import datasets
from datasets.tasks import AutomaticSpeechRecognition
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

_CITATION = """\
@phdthesis{halabi2016modern,
  title={Modern standard Arabic phonetics for speech synthesis},
  author={Halabi, Nawar},
  year={2016},
  school={University of Southampton}
}
"""

_DESCRIPTION = """\
This Speech corpus has been developed as part of PhD work carried out by Nawar Halabi at the University of Southampton.
The corpus was recorded in south Levantine Arabic
(Damascian accent) using a professional studio. Synthesized speech as an output using this corpus has produced a high quality, natural voice.
Note that in order to limit the required storage for preparing this dataset, the audio
is stored in the .flac format and is not converted to a float32 array. To convert, the audio
file to a float32 array, please make use of the `.map()` function as follows:


```python
import soundfile as sf

def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch

dataset = dataset.map(map_to_array, remove_columns=["file"])
```
"""

_URL = "http://en.arabicspeechcorpus.com/arabic-speech-corpus.zip"


class ArabicSpeechCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for ArabicSpeechCorpu."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(ArabicSpeechCorpusConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


class ArabicSpeechCorpus(datasets.GeneratorBasedBuilder):
    """ArabicSpeechCorpus dataset."""

    BUILDER_CONFIGS = [
        ArabicSpeechCorpusConfig(name="clean", description="'Clean' speech."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=48_000),
                    "phonetic": datasets.Value("string"),
                    "orthographic": datasets.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
            homepage=_URL,
            citation=_CITATION,
            task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download_and_extract(_URL)
        archive_path = os.path.join(archive_path, "arabic-speech-corpus")
        return [
            datasets.SplitGenerator(name="train", gen_kwargs={"archive_path": archive_path}),
            datasets.SplitGenerator(name="test", gen_kwargs={"archive_path": os.path.join(archive_path, "test set")}),
        ]

    
            
# Load pre-trained model and tokenizer
model_name = "asafaya/bert-base-arabic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Load Arabic text from file
with open("quran-simple-no-ayat-numbers.txt", "r", encoding="utf-8") as f:
    arabic_text = f.read()


# Tokenize the text
tokens = tokenizer(arabic_text, padding=True, truncation=True, return_tensors="pt", max_length=512)


# Predict vowels
with torch.no_grad():
    outputs = model(**tokens)
vowels = []
for token, prediction in zip(tokens["input_ids"][0], outputs.logits.argmax(2)[0]):
    token = tokenizer.decode(token.item())
    vowels.append(token)
    vowels[-1] += " " if prediction == 1 else ""
with open("quran-simple-no-ayat-numbers-with-vowels-2.txt", "w", encoding="utf-8") as f:
    for vowel in vowels:
        f.write(f"{vowel}\n")
text_with_vowels = "\n".join(vowels)
with open("quran-simple-no-ayat-numbers-with-vowels.txt", "w", encoding="utf-8") as f:
    f.write(text_with_vowels)
text_with_vowels = ""
for token, prediction in zip(tokens["input_ids"][0], outputs.logits.argmax(2)[0]):
    token = tokenizer.decode(token.item())
    vowels.append(token)
    if prediction == 1:
        text_with_vowels += f"{vowels[-1]}\n"
    else:
        text_with_vowels += f"{vowels[-1]}\n"
# # Process the predictions to get vowels
# vowels = []
# for token, prediction in zip(tokens["input_ids"][0], outputs.logits.argmax(2)[0]):
#     token = tokenizer.decode(token.item())
#     if token.startswith("##"):
#         vowels[-1] += token[2:]
#     else:
#         vowels.append(token)
        
#     vowels[-1] += " " if prediction == 1 else ""
# # Process the predictions to get vowels
# with open("quran-simple-no-ayat-numbers-with-vowels-2.txt", "w", encoding="utf-8") as f:
#     for token, prediction in zip(tokens["input_ids"][0], outputs.logits.argmax(2)[0]):
#         token = tokenizer.decode(token.item())
#         if token.startswith("##"):
#             vowels[-1] += token[2:]
#         else:
#             vowels.append(token)
#         if prediction == 1:
#             f.write(f"{vowels[-1]}*{vowels[-1]}*\n")
#         else:
#             f.write(f"{vowels[-1]}\n")
# # Combine original text with predicted vowels
# text_with_vowels = ",".join(vowels)

# # Write to a new file
# with open("quran-simple-no-ayat-numbers-with-vowels.txt", "w", encoding="utf-8") as f:
#     for i in text_with_vowels:
#         if i==",":
#             f.write("\n")
#         else:
#             f.write(text_with_vowels.strip(","))
# # Process the predictions to get vowels
# text_with_vowels = ""
# for token, prediction in zip(tokens["input_ids"][0], outputs.logits.argmax(2)[0]):
#     token = tokenizer.decode(token.item())
#     if token.startswith("##"):
#         vowels[-1] += token[2:]
#     else:
#         print(token)
#         vowels.append(token)
#     if prediction == 1:
#         text_with_vowels += f"*{vowels[-1]}*\n"
#     else:
#         text_with_vowels += f"{vowels[-1]}\n"

# # Write to a new file
# with open("quran-simple-no-ayat-numbers-with-vowels-3.txt", "w", encoding="utf-8") as f:
#     f.write(text_with_vowels.strip())

