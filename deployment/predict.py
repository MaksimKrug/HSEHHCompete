import re
from glob import glob

import fasttext
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from joblib import load
from navec import Navec
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.model import CustomDataset, FinalModel

tokenizer_inapropriate_message = AutoTokenizer.from_pretrained(
    "./data/", local_files_only=True
)
model_inapropriate_message = AutoModelForSequenceClassification.from_pretrained(
    "./data/", local_files_only=True
)

# embeddings
navec_model = Navec.load("./data/navec_hudlit_v1_12B_500K_300d_100q.tar")
fasttext_model = fasttext.load_model("./data/cc.ru.300.bin")

# read data
df = pd.read_csv("predict_example.csv")

# PREPROCESSING
# load standard scaler
std_scaler = load("shared/std_scaler.bin")


def preprocessing(df):
    df.fillna(
        value={
            "city": "<unk>",
            "position": "<unk>",
            "positive": "<unk>",
            "negative": "<unk>",
        },
        inplace=True,
    )
    # lowercase
    df[["positive", "negative", "city", "position"]] = df[
        ["positive", "negative", "city", "position"]
    ].apply(lambda x: x.str.lower())

    #
    df["positive"] = df["positive"].str.replace("\*{6,6}", "организация")
    df["negative"] = df["negative"].str.replace("\*{6,6}", "организация")

    # Standard Scaler
    scaler_columns = [
        "salary_rating",
        "team_rating",
        "managment_rating",
        "career_rating",
        "workplace_rating",
        "rest_recovery_rating",
    ]
    df[scaler_columns] = std_scaler.transform(df[scaler_columns])

    # reset index
    df.reset_index(drop=True, inplace=True)

    def update_storages(positive, negative, metadata):
        positive_sentences.append(positive)
        negative_sentences.append(negative)
        meta.append(metadata)

    def detect_inapropriate(positive, negative):
        positive_tokens = tokenizer_inapropriate_message(
            positive, return_tensors="pt", max_length=512
        )
        positive_innapropriate = model_inapropriate_message(**positive_tokens)
        positive_innapropriate = torch.argmax(
            torch.nn.Softmax()(positive_innapropriate.logits)
        ).item()
        negative_tokens = tokenizer_inapropriate_message(
            negative, return_tensors="pt", max_length=512
        )
        negative_innapropriate = model_inapropriate_message(**negative_tokens)
        negative_innapropriate = torch.argmax(
            torch.nn.Softmax()(negative_innapropriate.logits)
        ).item()

        return positive_innapropriate, negative_innapropriate

    positive_sentences, negative_sentences, meta = [], [], []

    for idx, data in df.iterrows():

        # text
        positive = data["city"] + " " + data["position"] + " " + data["positive"]
        negative = data["city"] + " " + data["position"] + " " + data["negative"]

        # metadata
        metadata = data[
            [
                "salary_rating",
                "team_rating",
                "managment_rating",
                "career_rating",
                "workplace_rating",
                "rest_recovery_rating",
            ]
        ].tolist()

        # innapropriate message
        positive_innapropriate, negative_innapropriate = detect_inapropriate(
            positive, negative
        )
        positive_make_sense = int(re.search("[а-я]", positive) != None)
        negative_make_sense = int(re.search("[а-я]", negative) != None)
        metadata += (
            [positive_innapropriate]
            + [negative_innapropriate]
            + [int(positive == negative)]
            + [positive_make_sense]
            + [negative_make_sense]
        )

        # update storages
        update_storages(positive, negative, metadata)


    for i in range(len(df)):
        batch = np.array(
            [positive_sentences[i], negative_sentences[i], list(meta[i])],
            dtype="object",
        )
        np.save(f"./examples/pred_data_{i}", batch)


preprocessing(df)

# Create dataset and dataloader
sent_size = 112
batch_size = 128

pred_data = glob("./examples/*.npy")
dataset_pred = CustomDataset(pred_data, sent_size, False, navec_model, fasttext_model)
dataloader_pred = DataLoader(dataset_pred, batch_size=batch_size, shuffle=False)

# Get preds
def get_preds(df):
    preds_list = []

    model = FinalModel.load_from_checkpoint(f"./shared/final_model_1.ckpt")
    trainer = pl.Trainer(gpus=0)

    preds = trainer.predict(model, dataloader_pred)
    preds_list.append(preds)

    preds = np.asarray([j.numpy().tolist() for i in preds_list[0] for j in i])
    submit_preds = []
    const = 0.5
    thresholds = [const]

    for pred in preds:
        pred = (pred > thresholds).astype(int).tolist()

        if sum(pred) == 0:
            count_zero += 1
            submit_preds.append("0")
        else:
            temp = ",".join([str(i) for i in range(9) if pred[i] == 1])
            submit_preds.append(temp)

    df["Preds"] = submit_preds

    return df

# Get preds
df = get_preds(df)

# Save preds
df.to_csv("data/preds.csv", index=False)
