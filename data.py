import re

import nltk
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer

# For innapropriate classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer_inapropriate_message = AutoTokenizer.from_pretrained(
    "Skoltech/russian-inappropriate-messages"
)
model_inapropriate_message = AutoModelForSequenceClassification.from_pretrained(
    "Skoltech/russian-inappropriate-messages"
)

import os

# For augmentations
import nlpaug.augmenter.word as naw

os.environ["TOKENIZERS_PARALLELISM"] = "false"

aug_contextual = naw.ContextualWordEmbsAug(
    model_path="bert-base-multilingual-uncased", aug_p=0.1
)
aug_swap = naw.RandomWordAug(action="swap")
aug_back_translation = naw.BackTranslationAug(
    from_model_name="Helsinki-NLP/opus-mt-ru-en",
    to_model_name="Helsinki-NLP/opus-mt-en-ru",
)

# Embeddings
from navec import Navec

path = "data/navec_hudlit_v1_12B_500K_300d_100q.tar"
navec = Navec.load(path)

# Stopwords
stopwords = nltk.corpus.stopwords.words("russian")


def preprocessing(train: pd.DataFrame, test: pd.DataFrame):
    # NaNs preprocessing
    train.fillna(
        value={
            "city": "<unk>",
            "position": "<unk>",
            "positive": "<unk>",
            "negative": "<unk>",
        },
        inplace=True,
    )
    test.fillna(
        value={
            "city": "<unk>",
            "position": "<unk>",
            "positive": "<unk>",
            "negative": "<unk>",
        },
        inplace=True,
    )

    # lowercase
    train[["positive", "negative", "city", "position"]] = train[
        ["positive", "negative", "city", "position"]
    ].apply(lambda x: x.str.lower())
    test[["positive", "negative", "city", "position"]] = test[
        ["positive", "negative", "city", "position"]
    ].apply(lambda x: x.str.lower())

    #
    train["positive"] = train["positive"].str.replace("\*{6,6}", "организация")
    train["negative"] = train["negative"].str.replace("\*{6,6}", "организация")
    test["positive"] = test["positive"].str.replace("\*{6,6}", "организация")
    test["negative"] = test["negative"].str.replace("\*{6,6}", "организация")

    # Standard Scaler
    scaler = StandardScaler()
    scaler_columns = [
        "salary_rating",
        "team_rating",
        "managment_rating",
        "career_rating",
        "workplace_rating",
        "rest_recovery_rating",
    ]
    train[scaler_columns] = scaler.fit_transform(train[scaler_columns])
    test[scaler_columns] = scaler.transform(test[scaler_columns])

    # Target to MultiLabel
    train["preprocessed_target"] = train["target"].apply(
        lambda x: [1 if str(i) in x.split(",") else 0 for i in range(9)]
    )

    # reset index
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, test


def get_vocab(train):
    # vocab
    tokenizer = nltk.RegexpTokenizer(r"[а-я]+|<unk>|[a-z]+")
    word2idx = {"<pad>": 0, "<unk>": 1}
    idx = 2

    # create vocab
    for text_column in ["positive", "negative"]:
        text = train[text_column].values
        tokens = [tokenizer.tokenize(sent) for sent in text]
        for sent in tokens:
            for word in sent:
                if word in stopwords:
                    continue
                word_emb = navec.get(word)
                if word not in word2idx and word_emb is not None:
                    word2idx[word] = idx
                    idx += 1

    # idx2word
    idx2word = {j: i for j, i in word2idx.items()}

    return word2idx, idx2word


def augmentations(df, is_test=False):
    def update_storages(positive, negative, metadata, labels, is_test=False):
        positive_sentences.append(positive)
        negative_sentences.append(negative)
        meta.append(metadata)
        labels.append(label)

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

    positive_sentences, negative_sentences, meta, labels = [], [], [], []

    for idx, data in tqdm(df.iterrows(), total=df.shape[0]):

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

        # target
        try:
            label = data["preprocessed_target"]
        except:
            label = "0"

        # For test
        if is_test:
            # update storages
            update_storages(positive, negative, metadata, labels, is_test)
            continue

        # update storages
        update_storages(positive, negative, metadata, labels)

        # Augmentations
        # Contextual
        if np.random.randint(10) in [1, 2, 3]:
            try:
                new_positive = aug_contextual.augment(positive).lower()
                new_negative = aug_contextual.augment(negative).lower()
                if new_positive != positive and new_negative != negative:
                    update_storages(new_positive, new_negative, metadata, labels)
            except:
                pass
        # Swap
        new_positive = aug_swap.augment(positive).lower()
        new_negative = aug_swap.augment(positive).lower()
        if new_positive != positive and new_negative != negative:
            update_storages(new_positive, new_negative, metadata, labels)
        # Back translation
        if np.random.randint(10) in [1, 2, 3]:
            try:
                new_positive = aug_back_translation.augment(positive)
                new_negative = aug_back_translation.augment(negative)
                if new_positive != positive and new_negative != negative:
                    update_storages(new_positive, new_negative, metadata, labels)
            except:
                pass

    return positive_sentences, negative_sentences, meta, labels

