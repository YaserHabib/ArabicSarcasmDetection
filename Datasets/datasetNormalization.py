import pandas as pd

ArsarcasmTrain = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv")[['tweet','sarcasm', "dialect"]]
ArsarcasmTest = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/testing_data.csv")[['tweet','sarcasm', 'dialect']]
iSarcasmEvalTrain = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.Ar.csv")[['text','sarcastic', 'dialect']]
iSarcasmEvalTestA = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_Ar_test.csv")[['text','sarcastic', 'dialect']]
iSarcasmEvalTestC = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_C_Ar_test.csv")[['text_0','text_1', 'sarcastic_id', 'dialect']]



iSarcasmEvalTrain.columns = ['tweet', 'sarcasm', 'dialect']
iSarcasmEvalTestC['sarcastic_tweet'] = iSarcasmEvalTestC.apply(lambda row: row['text_1'] if row['sarcastic_id'] == 1 else row['text_0'], axis=1)
iSarcasmEvalTestC['non_sarcastic_tweet'] = iSarcasmEvalTestC.apply(lambda row: row['text_0'] if row['sarcastic_id'] == 1 else row['text_1'], axis=1)

sarcastic_df = pd.DataFrame({'tweet': iSarcasmEvalTestC['sarcastic_tweet'], 'sarcasm': 1, 'dialect': iSarcasmEvalTestC['dialect']})
non_sarcastic_df = pd.DataFrame({'tweet': iSarcasmEvalTestC['non_sarcastic_tweet'], 'sarcasm': 0, 'dialect': iSarcasmEvalTestC['dialect']})
iSarcasmEvalTestC = pd.concat([sarcastic_df, non_sarcastic_df], ignore_index=True)



# Combine all three DataFrames
dataset = pd.concat([ArsarcasmTrain, ArsarcasmTest, iSarcasmEvalTrain, iSarcasmEvalTestC], ignore_index=True)
dataset = dataset.drop_duplicates()
dataset = dataset.dropna()
dataset = dataset.sample(frac=1)

# Print the counts
print(dataset['sarcasm'].value_counts())

dataset.to_csv(rf"Datasets\originalCombined.csv", index=False)