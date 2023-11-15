import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(r"C:\Users\Mohamed\Documents\Fall 2023 - 2024\Senior Project in CS\sysPath")
os.chdir(dname)

import time
import nltk
import random
import pandas as pd
import preProcessData

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from transformers import MarianMTModel, MarianTokenizer

if not os.path.isdir(r"..\Datasets\Augmented Datasets"):
    os.mkdir(r"..\Datasets\Augmented Datasets")



def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]
    return formated_bach



englishModelName = "Helsinki-NLP/opus-mt-ar-en"
arabicModelName = "Helsinki-NLP/opus-mt-en-ar"

englishModeltkn = MarianTokenizer.from_pretrained(englishModelName)
arabicModeltkn = MarianTokenizer.from_pretrained(arabicModelName)

englishModel = MarianMTModel.from_pretrained(englishModelName)
arabicModel = MarianMTModel.from_pretrained(arabicModelName)



def perform_translation(batch_texts, model, tokenizer, language="en"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    
    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return translated_texts



def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): # type: ignore
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)



def synonym_replacement(words, n):
    words = nltk.word_tokenize(words)
    stop_words = set(stopwords.words('english'))
    new_words = words.copy()
    
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence



def dataAugmentation(dataset):
    augDataset = pd.DataFrame(columns=["tweet", "dialect", "sarcasm"])
    backtransDataset = pd.DataFrame(columns=["tweet", "dialect", "sarcasm"])
    synonymrepDataset = pd.DataFrame(columns=["tweet", "dialect", "sarcasm"])

    sarcasmTweets = dataset[dataset.sarcasm == 1]["tweet"].tolist()
    sarcasmTweets_dialect = dataset[dataset.sarcasm ==1]["dialect"].tolist()


    for index in range(len(sarcasmTweets)):
        englishVersion = perform_translation([sarcasmTweets[index]], englishModel, englishModeltkn, "en")
        synreplacement_EnglishVer = synonym_replacement(" ".join(englishVersion), len(englishVersion))

        try:
            transArabicVer = perform_translation([englishVersion], arabicModel, arabicModeltkn, "ar")
        except:
            print(transArabicVer) # type: ignore

        augDataset.loc[len(augDataset.index)] = [ # type: ignore
                                                    " ".join(transArabicVer), # type: ignore
                                                    sarcasmTweets_dialect[index],
                                                    True
                                                ]
        backtransDataset.loc[len(augDataset.index)] =   [ # type: ignore
                                                            " ".join(transArabicVer), # type: ignore
                                                            sarcasmTweets_dialect[index],
                                                            True
                                                        ]

        try:
            synreplacement_ArabicVer = perform_translation([synreplacement_EnglishVer], arabicModel, arabicModeltkn, "ar")
        except:
            print(synreplacement_ArabicVer) # type: ignore

        augDataset.loc[len(augDataset.index)] = [ # type: ignore
                                                    " ".join(synreplacement_ArabicVer), # type: ignore
                                                    sarcasmTweets_dialect[index],
                                                    True
                                                ]
        synonymrepDataset.loc[len(augDataset.index)] = [ # type: ignore
                                                            " ".join(synreplacement_ArabicVer), # type: ignore
                                                            sarcasmTweets_dialect[index],
                                                            True
                                                        ]
        
    return augDataset, backtransDataset, synonymrepDataset



def siftData(data, name):
    data = data.drop_duplicates(subset=["tweet"])
    data = data.dropna()
    data = data.reset_index(drop=True)
    data.to_csv(rf"..\Datasets\Augmented Datasets\{name}.csv", index=False)



def dataProcessing(dataset):

    data = preProcessData.cleanData(dataset.copy(deep=True))
    print("\n-------        cleanData Done!        -------\n")

    augDataset, backtransDataset, synonymrepDataset = dataAugmentation(data.copy(deep=True))
    print("\n---------- dataAugmentation Done! ----------\n")

    augDataset = preProcessData.cleanData(augDataset.copy(deep=True))
    backtransDataset = preProcessData.cleanData(backtransDataset.copy(deep=True))
    synonymrepDataset = preProcessData.cleanData(synonymrepDataset.copy(deep=True))

    print("\n-------        cleanData Done!        -------\n")

    siftData(augDataset.copy(deep=True), "augDataset")
    siftData(backtransDataset.copy(deep=True), "backtransDataset")
    siftData(synonymrepDataset.copy(deep=True), "synonymrepDataset")



dataset = pd.read_csv(r"..\Datasets\originalCombined.csv")
startTime = time.time()
dataProcessing(dataset.copy(deep=True))
endTime = time.time()

executionTime = endTime - startTime
print(f"execution time: {executionTime}s")