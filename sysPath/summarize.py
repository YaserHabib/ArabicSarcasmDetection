import numpy as np
import seaborn as sns
import preProcessData #type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt


from keras.utils import plot_model
from keras.utils import pad_sequences
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer

from imblearn.over_sampling import SMOTE

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score




def prepareData(dataset):
    dataset.info()
    print(f"\n{dataset.head()}")

    dataset = preProcessData.preProcessData(dataset.copy(deep = True))

    dataset.info()
    print(f"\n{dataset.head()}")



    # prepare tokenizer
    T = Tokenizer()
    T.fit_on_texts(dataset["tweet"].tolist())
    vocab_size = len(T.word_index) + 1



    # integer encode the documents
    encoded_docs = T.texts_to_sequences(dataset["tweet"].tolist())
    # print("encoded_docs:\n",encoded_docs)



    # pad documents to a max length of 4 words
    max_length = len(max(np.array(dataset["tweet"]), key = len))
    padded_docs = pad_sequences(encoded_docs, maxlen = max_length, padding = "post")
    print("\npadded_docs:\n",padded_docs)



    # load the whole embedding into memory
    w2v_embeddings_index = {}
    TOTAL_EMBEDDING_DIM = 300
    embeddings_file = r"../../full_grams_cbow_300_twitter/full_grams_cbow_300_twitter.mdl"
    w2v_model = KeyedVectors.load(embeddings_file)



    for word in w2v_model.wv.index_to_key:
        w2v_embeddings_index[word] = w2v_model.wv[word]

    print("\nLoaded %s word vectors."% len(w2v_embeddings_index))



    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, TOTAL_EMBEDDING_DIM))

    for word, i in T.word_index.items():
        embedding_vector = w2v_embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("\nEmbedding Matrix shape:", embedding_matrix.shape)
    
    return dataset, max_length, vocab_size, TOTAL_EMBEDDING_DIM, embedding_matrix, padded_docs



def summarize(model, datasetName):
    print(f"\n{model.summary()}")
    print("\n # Wait just Fitting model on training data")
    plot_model(model, to_file = f'summary [{datasetName}].png', show_shapes = True, show_layer_names = True, dpi = 1000)



def TrainTestSplit(padded_docs, dataset):
    # splits into traint, validation, and test
    train_tweet, test_tweet, train_labels, test_labels = train_test_split(padded_docs, dataset["sarcasm"].to_numpy(), test_size = 0.1, random_state = 42)
    train_tweet, val_tweet, train_labels, val_labels = train_test_split(train_tweet, train_labels, test_size = 0.05, random_state = 42)

    return train_tweet, test_tweet, train_labels, test_labels, val_tweet, val_labels



def smote(tweet_train, labeled_train):
    sm = SMOTE()
    tweet_train, labeled_train = sm.fit_resample(tweet_train, labeled_train) # type: ignore
    return tweet_train, labeled_train



def fit(model, train_labels, train_tweet, val_tweet, val_labels):
    # fit the model
    class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes = np.unique(train_labels), y = train_labels)
    class_weights = dict(enumerate(class_weights))
    result = model.fit(train_tweet, train_labels, epochs = 20, verbose = 1, validation_data = (val_tweet, val_labels), class_weight = class_weights) # type: ignore

    return result



def modelEvaluation(model, test, true):
    predicted = np.round(model.predict(test))

    precision = precision_score(true, predicted)
    accuracy = accuracy_score(true, predicted)
    recall = recall_score(true, predicted)
    f1 = f1_score(true, predicted)

    classificationReport = classification_report(true, predicted, target_names = ["non-Sarcasm", "Sarcasm"])

    return predicted, precision, accuracy, recall, f1, classificationReport



def display(datasetName, classificationReport, ratio, smoteStatus, time):
    print(f"\n\nDataset used: {datasetName}")
    print(f"sarcasm to nonsarcasm: {ratio:.2f}")
    print(f"SMOTE: {smoteStatus}")
    print(f"Execution Time: {int(time)}s")
    print(classificationReport) # type: ignore
    print("\n\n" + "▒"*100 + "\n")



def recordResult(datasetName, classificationReport, ratio, smoteStatus, time):
    with open(r"Description - SMOTE OFF.txt", "a") as descriptionFile:
        descriptionFile.write("="*100)
        descriptionFile.write(f"\nDataset used: {datasetName}\n")
        descriptionFile.write(f"sarcasm to nonsarcasm: {ratio:.2f}")
        descriptionFile.write(f"SMOTE: {smoteStatus}")
        descriptionFile.write(f"\nExecution Time: {int(time)}s\n\n")
        descriptionFile.write(classificationReport) # type: ignore
        descriptionFile.write("\n\n" + "="*100 + "\n")

        descriptionFile.close()



def plotPerformance(result, datasetName):
    # Plot results
    acc = result.history['accuracy']
    val_acc = result.history['val_accuracy']
    loss = result.history['loss']
    val_loss = result.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'g', label = 'Training accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(f"Training vs validation accuracy - {datasetName}", dpi = 1000)

    plt.close()

    plt.plot(epochs, loss, 'g', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f"Training vs validation loss - {datasetName}", dpi = 1000)

    plt.close()



def saveFig(test_labels, predicted, accuracy, datasetName):
    confusionMatrix = confusion_matrix(test_labels, predicted)

    ax = plt.subplot()
    sns.heatmap(confusionMatrix, annot = True, fmt = 'g', ax = ax, cmap = "viridis") # annot = True to annotate cells, ftm = 'g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Accuracy: {accuracy*100:.2f}%")
    ax.xaxis.set_ticklabels(["non-Sarcasm", "Sarcasm"])
    ax.yaxis.set_ticklabels(["non-Sarcasm", "Sarcasm"])

    plt.savefig(f"CNN - {datasetName}.png", dpi = 1000)
    plt.close()



def recordXLSX(data, datasetName, nonSarcasmCount, SarcasmCount, ratio, precision, recall, f1, accuracy, smote_state = False):
    data.loc[len(data.index)] = [datasetName, nonSarcasmCount, SarcasmCount, smote_state, ratio, precision, recall, f1, accuracy] # type: ignore




def barPlot(data, modelStructure = "deepLearning"):
    datasetName = data["Dataset Name"].tolist()
    accuracy = data["Accuracy"].tolist()
    f1 = data["F1-Score"].tolist()

    measures = {
        "Accuracy": accuracy,
        "F1-score": f1
    }

    x = np.arange(len(datasetName))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, (attribute, measurement) in enumerate(measures.items()):
        rects = ax.bar(x + (idx * width), measurement, width, label=attribute)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Comparison of model performance across various datasets.")
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(datasetName)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{modelStructure}.png", dpi=1000)