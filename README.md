
# Sarcasm Detection from Arabic Tweets

This project focuses on developing and training multiple machine and deep learning models to identify sarcasm in Arabic tweets. Our goal is to enhance natural language processing capabilities for the Arabic language, dealing with diverse dialects and linguistic nuances.

## Authors

- [Abdulhadi Alaraj](https://github.com/AbdulhadiAlaraj)
- [Yaser Habib](https://github.com/YaserHabib)
- [Mohamed Erfan](https://github.com/MohamedElfares)
- [Abderahmane Benkheira](https://github.com/AbderahmaneBenkheira)


## Run Locally

Clone the project

```bash
  git clone https://github.com/YaserHabib/ArabicSarcasmDetection
```

Go to the project directory

```bash
  cd ArabicSarcasmDetection
```

Create Virtual Enivronment

```bash
python -m venv myenv
```
 - On Windows:
```bash
myenv\Scripts\activate
```
 - On macOS and Linux:
 ```bash
source myenv/bin/activate
```
Install dependencies

```bash
  pip install -r requirements.txt
```
## Datasets

 - [ArSarcasmV2 Dataset](https://github.com/iabufarha/ArSarcasm-v2)
 - [iSarcasmEval Dataset](https://github.com/iabufarha/iSarcasmEval)

## Usage/Examples

### Preprocessing

Our preprocessing pipeline includes cleaning and normalizing the tweets, removing noise such as emojis and English text, lemmatizing Arabic words, and removing stopwords and punctuation. Additionally, we tokenize the tweets and encode categorical variables like dialect and sarcasm.

![App Screenshot](https://github.com/YaserHabib/ArabicSarcasmDetection/blob/69d622e0901087621948a5faa4cc47c954e57d2e/Imgs/Cleaning%20Process2.PNG)
#### Usage:
Import the preProcessData.py file and call the function:
```python
preProcessData.cleanData(dataset)
```

- **Input:** DataFrame (dataset) containing tweet data.
- **Function:** Cleans the dataset by removing duplicates and NA entries, and then processes each tweet. This processing includes removing URLs, non-spacing marks, punctuation, numbers, emojis, and extra whitespaces; normalizing Arabic text; removing English text and stopwords; and performing Arabic lemmatization. Finally, tweets with only whitespace are removed.
- **Output:** DataFrame with cleaned tweets.


### Augmentation
The data augmentation process enriches our dataset by introducing variations of the original sarcastic tweets. This is done by translating the tweets to English, performing synonym replacement, and translating them back to Arabic. We also ensure the uniqueness of the augmented data by removing duplicates.

#### Key Features of Data Augmentation:
- **Back-translation:** Arabic tweets are translated to English and then back to Arabic to introduce syntactic diversity.
- **Synonym Replacement:** Synonyms are used to replace certain words in the English-translated tweets to generate different versions, which are then translated back to Arabic.
- **Dataset Diversification:** The augmented dataset is combined with the original dataset to create a more diverse and robust dataset for training.

#### Usage:

```python
def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]
    return formated_bach
```

- **Input:** Language code (language_code) and a list of text strings (batch_texts).
- **Function:** Prepares each text string in the batch for translation by appending a target language code at the beginning, following a specific format required by the translation model.
- **Output:** List of formatted text strings ready for translation.


```python
def perform_translation(batch_texts, model, tokenizer, language="en"):
```
- **Input:** List of texts (batch_texts), model, tokenizer, optional language code (default is English)
- **Function:** Formats texts for translation, uses the model to translate them, and decodes the results back to strings.
- **Output:** List of translated texts.


```python
def get_synonyms(word):
```

- **Input:** Single word (word).
- **Function:** Finds synonyms for the given word using the NLTK WordNet synsets.
- **Output:** List of synonym strings.


```python
def synonym_replacement(words, n):
```

- **Input:** String of words (words) and a number n indicating how many synonyms to replace.
- **Function:** Tokenizes the input string, finds synonyms for as many words as specified by n (ignoring stop words), and replaces them in the original string.
- **Output:** Modified string with n synonyms replaced.

```python
def dataAugmentation(dataset):
```
- **Input:** DataFrame (dataset) containing tweets and their corresponding sarcasm labels.
- **Function:** Iteratively translates sarcastic tweets to English, performs synonym replacement, and translates back to Arabic, capturing both back-translated and synonym-replaced versions by calling the previous functions.
- **Output:** Three DataFrames with the original, back-translated, and synonym-replaced tweets.

```python
def siftData(data, name):
```
- **Input:** DataFrame (data) and a string (name) indicating the filename to save.
- **Function:** Removes duplicate and NA entries from the DataFrame and saves it to a CSV file with the given name.
- **Output:** None (saves the cleaned DataFrame to a CSV file).

```python
def dataProcessing(dataset):
```

- **Input:** DataFrame (dataset) with raw tweet data.
- **Function:** Cleans the dataset, applies data augmentation, and then cleans the augmented datasets before saving them.
- **Output:** None (augmented data is saved to files).

To further enhance our dataset, we employ OpenAI's GPT-4 to generate additional sarcastic tweets. This step is crucial to balance our dataset, especially considering the scarcity of sarcastic tweets compared to non-sarcastic ones. We automate the generation and processing of tweets, ensuring they meet our criteria for sarcasm, sentiment, and dialect.

Overview: This script uses OpenAI's GPT models to generate sarcastic tweets in Arabic for dataset augmentation. It iterates through a loop to generate batches of tweets, ensuring diversity in the dataset.

Functionality:

- **API Setup:** Initializes OpenAI API with a key read from a file.
- **Data Generation Loop:** Repeatedly sends requests to the GPT model to generate tweets, using a predefined prompt designed to elicit sarcastic responses.
- **Response Parsing and Dataset Augmentation:** Extracts the generated tweets from the model's response, adds additional attributes (dialect, sentiment, sarcasm), and appends them to the existing dataset.
- **Saving and Tracking:** Saves the augmented tweets to a CSV file and keeps track of the total number of tokens used in the requests.
- **Inputs:**

        1. Path to the API key file ("../key.txt").
        2. Initial dataset for augmentation ("dataset_GPT.csv").
- **Outputs:**

        1. Augmented dataset with new tweets ("dataset_GPT.csv").
        2. Total count of tokens used in the GPT interactions.
- **Key Components:**

        1. Use of OpenAI's ChatCompletion.create to generate tweet-like responses.
        2. Regular expression parsing to extract tweets from the GPT response.
        3. Time delay (sleep) to manage request rate.

The script is designed to handle responses in a specific format and may need adjustments based on the actual output of the GPT model and the structure of the initial dataset.

#### ***Note***
Please ensure you comply with OpenAI's usage policies when using the model for data augmentation. The code provided is for demonstration purposes and should be used within the limits of the API and with proper error handling.


## Related

Here is the Streamlit web app Repository that deploys these models (Under Construction)

https://github.com/AbdulhadiAlaraj/SarcasmViz


## Roadmap

- Additional browser support

- Add more integrations


## Acknowledgements

We would like to thank Dr. Zaher, our academic supervisor at the University of Sharjah, for his invaluable guidance, mentorship, and support. His expertise, insightful feedback, and unwavering encouragement have been instrumental in helping us navigate the complexities of this project. We would also like to extend our heartfelt thanks to our dear friends and family who have stood by us and provided us with much-needed support and motivation.

 - [AraBERT](https://github.com/aub-mind/arabert)
