import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import pandas as pd

ArsarcasmTrain = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv")[["tweet", "dialect", "sarcasm"]]
ArsarcasmTest = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/testing_data.csv")[["tweet", "dialect", "sarcasm"]]
iSarcasmEvalTrain = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.Ar.csv")[["text","dialect", "sarcastic"]]
iSarcasmEvalTrain_rephrase = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.Ar.csv")[["rephrase", "dialect", "sarcastic"]]
iSarcasmEvalTestA = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_Ar_test.csv")[["text", "dialect", "sarcastic"]]
iSarcasmEvalTestC = pd.read_csv(r"https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_C_Ar_test.csv")[["text_0", "text_1", "dialect", "sarcastic_id"]]



iSarcasmEvalTrain.columns = ["tweet", "dialect", "sarcasm"]
iSarcasmEvalTrain_rephrase.columns = ["tweet", "dialect", "sarcasm"]
iSarcasmEvalTestA.columns = ["tweet", "dialect", "sarcasm"]
iSarcasmEvalTestC["sarcastic_tweet"] = iSarcasmEvalTestC.apply(lambda row: row["text_1"] if row["sarcastic_id"] == 1 else row["text_0"], axis=1)
iSarcasmEvalTestC["non_sarcastic_tweet"] = iSarcasmEvalTestC.apply(lambda row: row["text_0"] if row["sarcastic_id"] == 1 else row["text_1"], axis=1)

sarcastic_df = pd.DataFrame({"tweet": iSarcasmEvalTestC["sarcastic_tweet"], "dialect": iSarcasmEvalTestC["dialect"], "sarcasm": 1})
non_sarcastic_df = pd.DataFrame({"tweet": iSarcasmEvalTestC["non_sarcastic_tweet"], "dialect": iSarcasmEvalTestC["dialect"], "sarcasm": 0})
iSarcasmEvalTestC = pd.concat([sarcastic_df, non_sarcastic_df], ignore_index=True)

# Combine all three DataFrames
dataset = pd.concat([ArsarcasmTrain, ArsarcasmTest, iSarcasmEvalTrain, iSarcasmEvalTrain_rephrase, iSarcasmEvalTestA, iSarcasmEvalTestC], ignore_index=True)
dataset = dataset.drop_duplicates(subset="tweet")
dataset = dataset.dropna()
dataset = dataset.sample(frac=1)
dataset = dataset.reset_index(drop=True)
dataset = dataset[["tweet", "dialect", "sarcasm"]]

# Print the counts
print(f"\n{dataset['sarcasm'].value_counts()}\n")
print(f"\n{dataset.head()}\n")
print(f"\n{dataset.info()}\n")

dataset.to_csv(r"originalCombined.csv", encoding="utf-8", index=False)