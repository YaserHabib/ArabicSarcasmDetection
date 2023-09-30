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
        for l in syn.lemmas(): 
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
    augDataset = pd.DataFrame(columns=["tweet", "dialect", "sentiment", "sarcasm"])
    backtransDataset = pd.DataFrame(columns=["tweet", "dialect", "sentiment", "sarcasm"])
    synonymrepDataset = pd.DataFrame(columns=["tweet", "dialect", "sentiment", "sarcasm"])

    sarcasmTweets = dataset[dataset.sarcasm == 1]["tweet"].tolist()
    sarcasmTweets_dialect = dataset[dataset.sarcasm ==1]["dialect"].tolist()
    sarcasmTweets_sentiment = dataset[dataset.sarcasm ==1]["sentiment"].tolist()


    for index in range(len(sarcasmTweets)):
        englishVersion = perform_translation([sarcasmTweets[index]], englishModel, englishModeltkn, "en")
        synreplacement_EnglishVer = synonym_replacement(" ".join(englishVersion), len(englishVersion))

        try:
            transArabicVer = perform_translation([englishVersion], arabicModel, arabicModeltkn, "ar")
        except:
            print(transArabicVer)

        augDataset.loc[len(augDataset.index)] = [
                                                    transArabicVer,
                                                    sarcasmTweets_dialect[index],
                                                    sarcasmTweets_sentiment[index],
                                                    True
                                                ]
        backtransDataset.loc[len(augDataset.index)] =   [
                                                            transArabicVer,
                                                            sarcasmTweets_dialect[index],
                                                            sarcasmTweets_sentiment[index],
                                                            True
                                                        ]

        try:
            synreplacement_ArabicVer = perform_translation([synreplacement_EnglishVer], arabicModel, arabicModeltkn, "ar")
        except:
            print(synreplacement_ArabicVer)

        augDataset.loc[len(augDataset.index)] = [
                                                    synreplacement_ArabicVer,
                                                    sarcasmTweets_dialect[index],
                                                    sarcasmTweets_sentiment[index],
                                                    True
                                                ]
        synonymrepDataset.loc[len(augDataset.index)] = [
                                                            synreplacement_ArabicVer,
                                                            sarcasmTweets_dialect[index],
                                                            sarcasmTweets_sentiment[index],
                                                            True
                                                        ]
        
        print(transArabicVer,"\n",synreplacement_ArabicVer,"\n\n")
    return augDataset, backtransDataset, synonymrepDataset



def siftData(data, name):
    data = data.drop_duplicates(subset=["tweet"])
    data = data.dropna(axis=1)
    data = data.reset_index(drop=True)
    data.to_csv(rf"..\Datasets\{name}.csv", index=False)



def dataProcessing(dataset):

    data = preProcessData.cleanData(dataset.copy(deep=True))
    print("\n-------        cleanData Done!        -------\n")

    augDataset, backtransDataset, synonymrepDataset = dataAugmentation(data.copy(deep=True))
    print("\n---------- dataAugmentation Done! ----------\n")

    augDataset = preProcessData.cleanData(augDataset.copy(deep=True))
    backtransDataset = preProcessData.cleanData(backtransDataset.copy(deep=True))
    synonymrepDataset = preProcessData.cleanData(synonymrepDataset.copy(deep=True))

    print("\n-------        cleanData Done!        -------\n")

    SiftData(augDataset.copy(deep=True), augDataset)
    SiftData(backtransDataset.copy(deep=True), backtransDataset)
    SiftData(synonymrepDataset.copy(deep=True), synonymrepDataset)



dataset = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv")
startTime = time.time()
dataProcessing(dataset.copy(deep=True))
endTime = time.time()

executionTime = endTime - startTime
print(f"execution time: {executionTime}s")