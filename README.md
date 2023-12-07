
# Sarcasm Detection from Arabic Tweets

This project focuses on developing and training multiple machine and deep learning models to identify sarcasm in Arabic tweets. Our goal is to enhance natural language processing capabilities for the Arabic language, dealing with diverse dialects and linguistic nuances.

# Table of Contents
- **[Introduction](#sarcasm-detection-from-arabic-tweets)**
- **[Authors](#authors)**
- **[Run Locally](#run-locally)**
  - *Clone the project*
  - *Set up Virtual Environment*
  - *Install dependencies*
- **[Datasets](#datasets)**
- **[Dataset Aggregation](#dataset-aggregation)**
- **[Preprocessing](#preprocessing)**
- **[Augmentation](#augmentation)**
- **[Machine Learning Model Training](#machine-learning-model-training)**
- **[CNN-based Sarcasm Detection Pipeline](#cnn-based-sarcasm-detection-pipeline)**
- **[RNN-based Sarcasm Detection Pipeline](#rnn-based-sarcasm-detection-pipeline)**
- **[Training AraBERT for Sarcasm Detection](#training-arabert-for-sarcasm-detection)**
- **[Related Projects](#related)**
- **[Roadmap](#roadmap)**
- **[Acknowledgements](#acknowledgements)**

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
## Dataset Aggregation
This script is designed to aggregate and preprocess several datasets for sarcasm detection in Arabic tweets. The datasets include Arsarcasm-v2, iSarcasmEval, and their variations.

* **Functionality:**

    * **Dataset Loading:** Reads multiple datasets from remote CSV files.
    * **Data Transformation:** Renames columns for consistency and extracts specific columns ("tweet", "dialect", "sarcasm").
    * **Dataset Concatenation:** Combines all datasets into a single DataFrame.
    * **Data Cleaning:** Removes duplicate and missing entries from the combined dataset.
    * **Randomization and Resetting:** Shuffles the dataset and resets indices for unbiased data processing.
    * **Finalization and Saving:** Saves the cleaned, combined dataset to a CSV file.
* **Key Components:**

    * Loading data from various sources including Arsarcasm-v2 and iSarcasmEval.
    * Data transformation and normalization for consistency across datasets.
    * Aggregation of multiple datasets into a unified DataFrame.
    * Data cleaning techniques to ensure quality and reliability.
* **Inputs:**
    * Remote URLs of CSV files containing individual datasets.
* **Outputs:**

    * A single, cleaned, and combined dataset (originalCombined.csv).
* **Additional Details:**

    * The script prints the count of sarcastic and non-sarcastic tweets, providing an overview of class distribution.
    * It also displays the first few rows of the combined dataset for a quick preview.
Run the script to download, preprocess, and combine the datasets.
The resulting CSV file can be used for further analysis or machine learning tasks related to sarcasm detection.

```python
python datasetAggregation.py
```
## Preprocessing

Our preprocessing pipeline includes cleaning and normalizing the tweets, removing noise such as emojis and English text, lemmatizing Arabic words, and removing stopwords and punctuation. Additionally, we tokenize the tweets and encode categorical variables like dialect and sarcasm.

![App Screenshot](https://github.com/YaserHabib/ArabicSarcasmDetection/blob/69d622e0901087621948a5faa4cc47c954e57d2e/Imgs/Cleaning%20Process2.PNG)
### Usage:
Import the preProcessData.py file and call the function:
```python
preProcessData.cleanData(dataset)
```

- **Input:** DataFrame (dataset) containing tweet data.
- **Function:** Cleans the dataset by removing duplicates and NA entries, and then processes each tweet. This processing includes removing URLs, non-spacing marks, punctuation, numbers, emojis, and extra whitespaces; normalizing Arabic text; removing English text and stopwords; and performing Arabic lemmatization. Finally, tweets with only whitespace are removed.
- **Output:** DataFrame with cleaned tweets.

## Augmentation
The data augmentation process enriches our dataset by introducing variations of the original sarcastic tweets. This is done by translating the tweets to English, performing synonym replacement, and translating them back to Arabic. We also ensure the uniqueness of the augmented data by removing duplicates.

### Key Features of Data Augmentation:
- **Back-translation:** Arabic tweets are translated to English and then back to Arabic to introduce syntactic diversity.
- **Synonym Replacement:** Synonyms are used to replace certain words in the English-translated tweets to generate different versions, which are then translated back to Arabic.
- **Dataset Diversification:** The augmented dataset is combined with the original dataset to create a more diverse and robust dataset for training.

### Usage:

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

**Overview:** This script uses OpenAI's GPT models to generate sarcastic tweets in Arabic for dataset augmentation. It iterates through a loop to generate batches of tweets, ensuring diversity in the dataset.

**Functionality:**

- **API Setup:** Initializes OpenAI API with a key read from a file.
- **Data Generation Loop:** Repeatedly sends requests to the GPT model to generate tweets, using a predefined prompt designed to elicit sarcastic responses.
- **Response Parsing and Dataset Augmentation:** Extracts the generated tweets from the model's response, adds additional attributes (dialect, sentiment, sarcasm), and appends them to the existing dataset.
- **Saving and Tracking:** Saves the augmented tweets to a CSV file and keeps track of the total number of tokens used in the requests.
* **Inputs:**

    * Path to the API key file ("../key.txt").
    * Initial dataset for augmentation ("dataset_GPT.csv").
* **Outputs:**

    * Augmented dataset with new tweets ("dataset_GPT.csv").
    * Total count of tokens used in the GPT interactions.
* **Key Components:**

    * Use of OpenAI's ChatCompletion.create to generate tweet-like responses.
    * Regular expression parsing to extract tweets from the GPT response.
    * Time delay (sleep) to manage request rate.

The script is designed to handle responses in a specific format and may need adjustments based on the actual output of the GPT model and the structure of the initial dataset.

### ***Note***
Please ensure you comply with OpenAI's usage policies when using the model for data augmentation. The code provided is for demonstration purposes and should be used within the limits of the API and with proper error handling.

## Machine Learning Model Training

This script establishes a complete machine learning pipeline for sarcasm detection in Arabic tweets, utilizing AraBERT for feature extraction and various classifiers for prediction.

* **Functionality:**

    * **Environment Setup:** Configures the file paths and adds necessary directories to the system path.
    * **Feature Extraction:** Uses the AraBERT model for transforming tweets into feature vectors.
    * **Model Training:** Trains multiple classification models on the extracted features.
    * **Model Evaluation:** Evaluates the performance of each model and plots confusion matrices.
    * **Model Serialization:** Saves the trained models for future use.


* **Key Functions:**

    * **extract_arabert_features:** Extracts features from texts using AraBERT.
    * **process_in_batches:** Processes texts in batches to extract features.
    * **plotCM:** Plots a confusion matrix for model evaluation.
    * **Training loop:** Fits various machine learning models to the training data.
* **Inputs:**

   * Preprocessed dataset with 'tweets'(string) and 'sarcasm'(0 or 1) labels.
* **Outputs:**

    * Trained machine learning models.
    * Confusion matrix plots for each model.
* **Additional Details:**

The script splits the dataset into training and testing sets.
It employs a variety of classifiers, including logistic regression, ridge classifier, linear SVC, K-nearest neighbors, and naive Bayes.
The performance of each classifier is evaluated based on accuracy and other metrics.
Usage:

Run the script after ensuring all dependencies are installed and the dataset is in the specified path.
The script will automatically process the data, train the models, evaluate their performance, and save them.

```python
python Ml_pipelines.py
```

## CNN-based Sarcasm Detection Pipeline
This script sets up and trains a Convolutional Neural Network (CNN) with LSTM layers for detecting sarcasm in Arabic tweets. It utilizes multiple datasets and incorporates techniques like SMOTE for dealing with class imbalances.

* **Functionality:**

    * **Environment and Directory Setup:** Adjusts file paths, working directories, and TensorFlow logging levels.
    * **Data Preparation and Loading:** Loads multiple datasets for sarcasm detection, including ArSarcasm-v2 and iSarcasmEval.
    * **Model Configuration:** Defines a CNN model with LSTM layers and embedding, optimized for sarcasm detection.
    * **Data Processing and Model Training:** Preprocesses data, splits it into training, validation, and testing sets, optionally applies SMOTE, and trains the model.
    * **Evaluation and Result Recording:** Evaluates the model's performance and records results in various formats, including plots and Excel files.
* **Key Components:**

    * *configModel:* Configures the CNN model architecture.
    * Model training with performance evaluation.
    * Result visualization and recording, including precision, recall, F1-score, and accuracy.

* **Inputs:**

    * Multiple datasets with tweets, dialects, and sarcasm labels.
* **Outputs:**

    * Trained CNN models for each dataset.
    * Performance plots and Excel sheets documenting model metrics.
* **Additional Details:**

    * The script includes tensorboard callbacks for logging.
    * Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset when needed.
    * Generates confusion matrices and accuracy/loss plots for each trained model.
Ensure that the necessary environment is set up, the data is available, and that you are in the correct folder before running the script. The script will automatically process the data, train the CNN models for each dataset, and save the results.
```python
python Model.py
```

## RNN-based Sarcasm Detection Pipeline
This script involves setting up a Recurrent Neural Network (RNN) model, specifically using LSTM and GRU layers, for the purpose of detecting sarcasm in Arabic tweets.

* **Functionality:**

    * **Environment Setup:** Configures file paths and TensorFlow logging level.
    * **Data Loading:** Loads multiple sarcasm datasets, including ArSarcasm-v2 and iSarcasmEval.
    * **Model Configuration:** Sets up an RNN model incorporating LSTM and GRU layers with an embedding layer, tailored for text classification.
    * **Data Preprocessing and Model Training:** Processes data for model compatibility, performs train-test splitting, applies SMOTE if needed, and trains the RNN model.
    * **Evaluation and Visualization:** Evaluates the model's performance on test data, plots performance metrics, and records results.
* **Key Components:**

    * *configModel:* Function to configure the RNN model architecture.
    * Data preprocessing steps including tokenization and embedding matrix creation.
    * Model training with performance evaluations like precision, recall, F1-score, and accuracy.
    * Result visualization and recording, including performance plots and Excel sheet generation.
* **Inputs:**
    * Datasets containing tweets, dialects, and sarcasm labels.
* **Outputs:**

    * Trained RNN models for each dataset.
    * Plots and Excel sheets documenting model performance.
* **Additional Details:**

    * The script utilizes TensorBoard for logging model training metrics.
    * SMOTE is optionally used to balance the datasets based on sarcasm class distribution.
    * Outputs include training and validation loss and accuracy plots, confusion matrices, and an Excel record of model performance across different datasets.

Ensure that the necessary environment is set up, the data is available, and that you are in the correct folder before running the script.
The script will preprocess the data, train the RNN model for each dataset, evaluate, and save the results.

```python
python Model.py
```
## Training AraBERT for Sarcasm Detection
This script is for training an AraBERT model, a BERT variant pre-trained on Arabic, for sarcasm detection in Arabic text. The model is fine-tuned on a labeled dataset of tweets.

* **Functionality:**

    * **Environment Setup:** Adjusts file paths and working directories.
    * **Data Preprocessing:** Tokenizes and preprocesses the dataset using AraBERT's tokenizer and custom preprocessing functions.
    * **Model Setup:** Loads the AraBERT model with a sequence classification head and configures training parameters.
    * **Training Process:** The model is trained on the preprocessed data, with early stopping to prevent overfitting.
    * **Evaluation and Visualization:** Evaluates the model on a test dataset and generates plots for accuracy, loss, and a confusion matrix.

* **Key Components:**

    * *tokenize_and_preprocess:* Tokenizes and preprocesses text data.
    * AraBERT model loading and configuration.
    * Training with callback for early stopping.
    * Evaluation using classification report and confusion matrix.
    * Plotting accuracy, loss, and confusion matrix.

* **Inputs:**

    * Preprocessed dataset with labeled sarcasm tweets.
* **Outputs:**

    * Trained AraBERT model.
    * Plots for model accuracy, loss, and confusion matrix.
    * Classification report for model evaluation.
* **Models Used:**
    * [AraBERTv02](https://github.com/aub-mind/arabert)

* **Additional Details:**

    * The script includes optional code for layer freezing and learning rate scheduling, which can be uncommented for advanced training configurations.
    * The model is compiled with specific hyperparameters and a binary cross-entropy loss function suitable for a two-class classification task.
Run the script after ensuring all dependencies are installed and the dataset is prepared. The script will preprocess the data, train the model, evaluate its performance, and save the results in specified directories.

```python
python Arabert.py
```
## Related

Here is the Streamlit web app Repository that deploys these models (Under Construction)

https://github.com/AbdulhadiAlaraj/SarcasmViz


## Roadmap

- Additional browser support

- Add more integrations


## Acknowledgements

We would like to thank Dr. Zaher, our academic supervisor at the University of Sharjah, for his invaluable guidance, mentorship, and support. His expertise, insightful feedback, and unwavering encouragement have been instrumental in helping us navigate the complexities of this project. We would also like to extend our heartfelt thanks to our dear friends and family who have stood by us and provided us with much-needed support and motivation.
