import nltk
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from navec import Navec
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer_bert,
        tokenizer_lstm,
        sent_size,
        train_mode,
        model_type="both",
    ):
        # utils
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_lstm = tokenizer_lstm
        self.train_mode = train_mode
        self.sent_size = sent_size
        self.model_type = model_type
        self.navec = Navec.load("../data/navec_hudlit_v1_12B_500K_300d_100q.tar")

        # init features
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        # get sent
        data_idx = np.load(self.data_path[idx], allow_pickle=True)
        positive = data_idx[0].replace("******", "компания").replace("\xa0", " ").replace(r"\s+", "\s").strip()
        negative = data_idx[1].replace("******", "компания").replace("\xa0", " ").replace(r"\s+", "\s").strip()
        metadata = data_idx[2]

        # metadata
        metadata += [len(positive.split()) / 112]
        metadata += [len(negative.split()) / 112]
        metadata = torch.FloatTensor(metadata)


        # target
        if self.train_mode:
            target = torch.FloatTensor(data_idx[3])

        if self.model_type in ["bert", "both"]:
            # positive
            bert_positive = self.tokenizer_bert(
                positive,
                padding="max_length",
                truncation=True,
                max_length=self.sent_size,
                return_tensors="pt",
            )
            # negative
            bert_negative = self.tokenizer_bert(
                negative,
                padding="max_length",
                truncation=True,
                max_length=self.sent_size,
                return_tensors="pt",
            )
            if self.train_mode:
                bert_return = (bert_positive, bert_negative, metadata, target)
            else:
                bert_return = (bert_positive, bert_negative, metadata)

        if self.model_type in ["lstm", "both"]:
            # positive
            lstm_positive = [
                self.navec.get(w) for w in self.tokenizer_lstm.tokenize(positive)
            ]
            if lstm_positive == []:
                lstm_positive = [self.navec.get("<unk>")]
            lstm_positive = np.stack(
                [i if i is not None else self.navec.get("<unk>") for i in lstm_positive]
            )
            lstm_positive = np.pad(
                lstm_positive[: self.sent_size],
                [((max(0, self.sent_size - len(lstm_positive))), 0), (0, 0)],
                mode="constant",
            )
            lstm_positive = np.stack(lstm_positive)
            # negative
            lstm_negative = [
                self.navec.get(w) for w in self.tokenizer_lstm.tokenize(negative)
            ]
            if lstm_negative == []:
                lstm_negative = [self.navec.get("<unk>")]

            lstm_negative = np.stack(
                [i if i is not None else self.navec.get("<unk>") for i in lstm_negative]
            )

            lstm_negative = np.pad(
                lstm_negative[: self.sent_size],
                [((max(0, self.sent_size - len(lstm_negative))), 0), (0, 0)],
                mode="constant",
            )
            lstm_negative = np.stack(lstm_negative)
            if self.train_mode:
                lstm_return = (lstm_positive, lstm_negative, metadata, target)
            else:
                lstm_return = (lstm_positive, lstm_negative, metadata)

        # grab output
        if self.model_type == "bert":
            return {"bert": bert_return}
        elif self.model_type == "lstm":
            return {"lstm": lstm_return}
        elif self.model_type == "both":
            return {"bert": bert_return, "lstm": lstm_return}


def init_RUBert(
    is_train: bool = False, model_path: str = "Skoltech/russian-sensitive-topics",
):
    """
    rubert-base-cased
    rubert-base-cased-sentence
    distilrubert-base-cased-conversational
    """
    RUBert = BertModel.from_pretrained(model_path)
    for i in RUBert.named_parameters():
        i[1].requires_grad = is_train

    return RUBert


class Model(pl.LightningModule,):
    def __init__(
        self,
        lr,
        weight_decay,
        is_train,
        linear1_meta_size=64,
        linear1_token_size=64,
        linear2_size=128,
        dropout1_weight=0.2,
        dropout2_weight=0.2,
    ):
        super().__init__()

        # save hyperparams
        self.save_hyperparameters()

        # utils
        self.lr = lr
        self.weight_decay = weight_decay
        self.is_train = is_train
        self.linear1_meta_size = linear1_meta_size
        self.linear1_token_size = linear1_token_size
        self.linear2_size = linear2_size
        self.dropout1_weight = dropout1_weight
        self.dropout2_weight = dropout2_weight

        # metrics
        self.metric_accuracy = torchmetrics.Accuracy()
        self.metric_f1 = torchmetrics.F1(num_classes=8, average="samples")

        # RuBert
        self.bert = init_RUBert(self.is_train)

        # Linears
        self.linear1_metadata = nn.Linear(13, self.linear1_meta_size)
        self.linear2_metadata = nn.Linear(self.linear1_meta_size, self.linear2_size)

        self.linear1_positive = nn.Linear(768, self.linear1_token_size)
        self.linear1_negative = nn.Linear(768, self.linear1_token_size)
        self.linear_tokens = nn.Linear(2 * self.linear1_token_size, self.linear2_size)

        self.linear_out = nn.Linear(2 * self.linear2_size, 8)

        # utils
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(self.dropout1_weight)
        self.dropout2 = nn.Dropout(self.dropout2_weight)

    def forward(self, positive, negative, metadata):

        # RuBert layer
        positive = self.bert(
            input_ids=positive["input_ids"].squeeze(1),
            attention_mask=positive["attention_mask"].squeeze(1),
        )
        positive = positive[1]
        positive = torch.reshape(positive, (positive.shape[0], -1))
        positive = self.dropout1(positive)

        negative = self.bert(
            input_ids=negative["input_ids"].squeeze(1),
            attention_mask=negative["attention_mask"].squeeze(1),
        )
        negative = negative[1]
        negative = torch.reshape(negative, (negative.shape[0], -1))
        negative = self.dropout1(negative)

        # Linear layers
        positive = self.relu(self.linear1_positive(positive))
        positive = self.dropout2(positive)
        negative = self.relu(self.linear1_positive(negative))
        negative = self.dropout2(negative)

        tokens = torch.cat((positive, negative), dim=-1)
        tokens = self.relu(self.linear_tokens(tokens))

        metadata = self.relu(self.linear1_metadata(metadata))
        metadata = self.dropout2(metadata)
        metadata = self.relu(self.linear2_metadata(metadata))

        x = torch.cat((tokens, metadata), dim=-1)

        # Output
        out = self.linear_out(x)
        out = torch.nn.Sigmoid()(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.1, verbose=True
        )
        # learning rate scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }

    def training_step(self, batch, batch_idx):
        positive, negative, metadata, target = batch["bert"]
        out = self(positive, negative, metadata)
        loss = torch.nn.BCELoss(reduction="none")(out, target)
        weights = torch.tensor(
            [1,1,1,1,1,1,1,1], device=self.device
        )
        loss = (loss * weights).mean()
        accuracy = self.metric_accuracy(out, target.int())
        f1 = self.metric_f1(out, target.int())

        # save logs
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)

        return {"loss": loss, "accuracy": accuracy, "F1": f1}

    def validation_step(self, batch, batch_idx):
        positive, negative, metadata, target = batch["bert"]
        out = self(positive, negative, metadata)
        loss = torch.nn.BCELoss(reduction="none")(out, target)
        weights = torch.tensor(
            [1,1,1,1,1,1,1,1], device=self.device
        )
        loss = (loss * weights).mean()
        accuracy = self.metric_accuracy(out, target.int())
        f1 = self.metric_f1(out, target.int())

        # save logs
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("hp_metric", f1)

        return {"loss": loss, "accuracy": accuracy, "F1": f1}

    def predict_step(self, batch, batch_idx):
        try:
            positive, negative, metadata = batch["bert"]
        except:
            positive, negative, metadata, target = batch["bert"]
        out = self(positive, negative, metadata)
        return out

    # def validation_epoch_end(self, outs):
    #     # log epoch metric
    #     metric = np.mean([i["F1"].item() for i in outs])
    #     if metric < 0.78:
    #         raise KeyboardInterrupt


class LSTMModel(pl.LightningModule,):
    def __init__(
        self,
        lr=1e-4,
        weight_decay=1e-6,
        hidden_size=4,
        bidirectional=True,
        dropout_lstm=0.2,
        dropout_linear=0.2,
        linear1_meta=256,
        linear2_size=256,
    ):
        super().__init__()

        # save hyperparams
        self.save_hyperparameters()

        # utils
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout_lstm = dropout_lstm
        self.dropout_linear = dropout_linear
        self.linear2_size = linear2_size
        self.linear1_meta = linear1_meta

        # metrics
        self.metric_accuracy = torchmetrics.Accuracy()
        self.metric_f1 = torchmetrics.F1(num_classes=8, average="samples")

        # lstm layers
        self.lstm_layer_positive = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            bias=False,
            bidirectional=self.bidirectional,
            dropout=self.dropout_lstm,
            batch_first=True,
        )
        self.lstm_layer_negative = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            bias=False,
            bidirectional=self.bidirectional,
            dropout=self.dropout_lstm,
            batch_first=True,
        )

        # linear layers
        self.linear1_positive = nn.Linear(hidden_size * (bidirectional + 1), 300)
        self.linear1_negative = nn.Linear(hidden_size * (bidirectional + 1), 300)
        self.linear1_metadata = nn.Linear(13, self.linear1_meta)
        self.linear2_metadata = nn.Linear(self.linear1_meta, self.linear2_size)
        self.linear2_tokens = nn.Linear(1320 * 2, self.linear2_size)
        self.linear_out = nn.Linear(self.linear2_size * 2, 8)

        # utils
        self.relu = nn.ReLU6(inplace=True)
        self.dropout = nn.Dropout(self.dropout_linear)

    def forward(self, positive, negative, metadata):

        # metadata
        metadata = self.relu(self.linear1_metadata(metadata))
        metadata = self.dropout(metadata)
        metadata = self.relu(self.linear2_metadata(metadata))

        # lstm outputs
        positive_lstm_out, (positive_h_t, positive_c_t) = self.lstm_layer_positive(
            positive
        )
        negative_lstm_out, (negative_h_t, negative_c_t) = self.lstm_layer_negative(
            negative
        )

        # pooling
        # positive
        positive_maxpool = nn.MaxPool1d(2)(torch.permute(positive_lstm_out, (0, 2, 1)))
        positive_averagepool = nn.AvgPool1d(2)(
            torch.permute(positive_lstm_out, (0, 2, 1))
        )
        positive_maxpool = torch.permute(positive_maxpool, (0, 2, 1))
        positive_averagepool = torch.permute(positive_averagepool, (0, 2, 1))
        linear_maxpool = self.linear1_positive(positive_maxpool)
        linear_averagepool = self.linear1_positive(positive_averagepool)
        positive_out = torch.cat((linear_maxpool, linear_averagepool), dim=1)
        positive_out = positive + positive_out
        positive_out = nn.MaxPool2d(5)(positive_out)
        positive_out = torch.reshape(positive_out, (positive_out.shape[0], -1))

        # negative
        negative_maxpool = nn.MaxPool1d(2)(torch.permute(negative_lstm_out, (0, 2, 1)))
        negative_averagepool = nn.AvgPool1d(2)(
            torch.permute(negative_lstm_out, (0, 2, 1))
        )
        negative_maxpool = torch.permute(negative_maxpool, (0, 2, 1))
        negative_averagepool = torch.permute(negative_averagepool, (0, 2, 1))
        linear_maxpool = self.linear1_negative(negative_maxpool)
        linear_averagepool = self.linear1_negative(negative_averagepool)
        negative_out = torch.cat((linear_maxpool, linear_averagepool), dim=1)
        negative_out = negative + negative_out
        negative_out = nn.MaxPool2d(5)(negative_out)
        negative_out = torch.reshape(negative_out, (negative_out.shape[0], -1))

        # linear outputs
        tokens_out = self.relu(torch.cat((positive_out, negative_out), dim=1))
        tokens_out = self.dropout(tokens_out)
        tokens_out = self.relu(self.linear2_tokens(tokens_out))

        # Output
        out = torch.cat((tokens_out, metadata), dim=1)
        out = self.linear_out(out)
        out = nn.Sigmoid()(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.1, verbose=True
        )
        # learning rate scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }

    def training_step(self, batch, batch_idx):
        positive, negative, metadata, target = batch["lstm"]
        out = self(positive, negative, metadata)
        loss = nn.BCELoss(reduction="none")(out, target)
        weights = torch.tensor(
            [1,1,1,1,1,1,1,1], device=self.device
        )
        loss = (loss * weights).mean()
        accuracy = self.metric_accuracy(out, target.int())
        f1 = self.metric_f1(out, target.int())

        # save logs
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)

        return {"loss": loss, "accuracy": accuracy, "F1": f1}

    def validation_step(self, batch, batch_idx):
        positive, negative, metadata, target = batch["lstm"]
        out = self(positive, negative, metadata)
        loss = nn.BCELoss(reduction="none")(out, target)
        weights = torch.tensor(
            [1,1,1,1,1,1,1,1], device=self.device
        )
        loss = (loss * weights).mean()
        accuracy = self.metric_accuracy(out, target.int())
        f1 = self.metric_f1(out, target.int())

        # save logs
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("hp_metric", f1)

        return {"loss": loss, "accuracy": accuracy, "F1": f1}

    def predict_step(self, batch, batch_idx):
        try:
            positive, negative, metadata = batch["lstm"]
        except:
            positive, negative, metadata, target = batch["lstm"]
        out = self(positive, negative, metadata)
        return out
