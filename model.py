import fasttext
import nltk
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from navec import Navec
from torch import nn
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self, data_path, sent_size, train_mode, navec_model, fasttext_model,
    ):
        # utils
        self.tokenizer = nltk.RegexpTokenizer(r"[а-я]+|<unk>|<pad>")
        self.train_mode = train_mode
        self.sent_size = sent_size
        self.navec = navec_model
        self.fasttext_model = fasttext_model

        # init features
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        # get sent
        data_idx = np.load(self.data_path[idx], allow_pickle=True)
        positive = (
            data_idx[0]
            .replace("******", "компания")
            .replace("\xa0", " ")
            .replace(r"\s+", "\s")
            .strip()
        )
        negative = (
            data_idx[1]
            .replace("******", "компания")
            .replace("\xa0", " ")
            .replace(r"\s+", "\s")
            .strip()
        )
        metadata = data_idx[2]

        # metadata
        metadata += [len(positive.split()) / 112]
        metadata += [len(negative.split()) / 112]
        metadata = torch.FloatTensor(metadata)

        # tokens
        positive_tokens = [
            w
            for w in self.tokenizer.tokenize(positive)
            if self.navec.get(w) is not None
        ]
        negative_tokens = [
            w
            for w in self.tokenizer.tokenize(negative)
            if self.navec.get(w) is not None
        ]
        if positive_tokens == []:
            positive_tokens = ["<unk>"]
        if negative_tokens == []:
            negative_tokens = ["<unk>"]

        # navec
        positive_navec = self.get_navec(positive_tokens)
        negative_navec = self.get_navec(negative_tokens)

        # fasttext
        positive_fasttext = self.get_fasttext(positive_tokens)
        negative_fasttext = self.get_fasttext(negative_tokens)

        # mean
        positive_mean = (positive_navec + positive_fasttext) / 2
        negative_mean = (negative_navec + negative_fasttext) / 2

        # return
        if self.train_mode:
            target = torch.FloatTensor(data_idx[3])
            return (
                positive_navec,
                negative_navec,
                positive_fasttext,
                negative_fasttext,
                positive_mean,
                negative_mean,
                metadata,
                target,
            )
        else:
            return (
                positive_navec,
                negative_navec,
                positive_fasttext,
                negative_fasttext,
                positive_mean,
                negative_mean,
                metadata,
            )

    def get_navec(self, tokens):
        tokens = np.stack([self.navec.get(w) for w in tokens])
        tokens = np.pad(
            tokens[: self.sent_size],
            [((max(0, self.sent_size - len(tokens))), 0), (0, 0)],
            mode="constant",
        )
        tokens = np.stack(tokens)

        return tokens

    def get_fasttext(self, tokens):
        tokens = np.stack([self.fasttext_model.get_word_vector(w) for w in tokens])
        tokens = np.pad(
            tokens[: self.sent_size],
            [((max(0, self.sent_size - len(tokens))), 0), (0, 0)],
            mode="constant",
        )
        tokens = np.stack(tokens)

        return tokens


class WordAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(WordAttention, self).__init__()

        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        out_attention = torch.nn.Tanh()(self.attention(x))
        out_context = torch.nn.Softmax(dim=1)(self.context(out_attention))
        out = (out_context * x).sum(1)

        return out_context.permute(0, 2, 1), out


class AttentionModel(nn.Module):
    def __init__(self, hidden_size, bidirectional, linear1_size, linear2_size):
        super(AttentionModel, self).__init__()

        # utils
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.linear1_size = linear1_size
        self.linear2_size = linear2_size

        # attention layers
        self.attention_positive = WordAttention(
            self.hidden_size * (self.bidirectional + 1)
        )
        self.attention_negative = WordAttention(
            self.hidden_size * (self.bidirectional + 1)
        )

        # linear layers
        self.linear1_positive = nn.Linear(
            self.hidden_size * (self.bidirectional + 1), self.linear1_size
        )
        self.linear1_negative = nn.Linear(
            self.hidden_size * (self.bidirectional + 1), self.linear1_size
        )
        self.linear2 = nn.Linear(2 * self.linear1_size, self.linear2_size)

        # utils
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lstm_positive, lstm_negative):
        # attention
        _, out_positive = self.attention_positive(lstm_positive)
        _, out_negative = self.attention_negative(lstm_negative)

        # fc
        x_positive = self.linear1_positive(out_positive)
        x_negative = self.linear1_negative(out_negative)
        x = torch.cat((x_positive, x_negative), dim=1)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        return x  # batch x self.linear2_size


class FinalModel(pl.LightningModule,):
    def __init__(
        self,
        lr=2e-4,
        weight_decay=1e-6,
        hidden_size=112,
        bidirectional=True,
        drop_lstm=0.2,
        drop_linear=0.3,
        linear1_size=512,
        linear2_size=1024,
    ):
        super().__init__()

        # save hyperparams
        self.save_hyperparameters()

        # utils
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.drop_lstm = drop_lstm
        self.drop_linear = drop_linear
        self.linear1_size = linear1_size
        self.linear2_size = linear2_size

        # metrics
        self.metric_accuracy = torchmetrics.Accuracy()
        self.metric_f1 = torchmetrics.F1(num_classes=9, average="samples")

        # lstm layers
        self.lstm_positive_navec = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            dropout=self.drop_lstm,
            batch_first=True,
        )
        self.lstm_negative_navec = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            dropout=self.drop_lstm,
            batch_first=True,
        )
        self.lstm_positive_fasttext = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            dropout=self.drop_lstm,
            batch_first=True,
        )
        self.lstm_negative_fasttext = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            dropout=self.drop_lstm,
            batch_first=True,
        )
        self.lstm_positive_mean = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            dropout=self.drop_lstm,
            batch_first=True,
        )
        self.lstm_negative_mean = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            dropout=self.drop_lstm,
            batch_first=True,
        )

        # attention model
        self.attention_navec = AttentionModel(
            self.hidden_size, self.bidirectional, self.linear1_size, self.linear2_size
        )
        self.attention_fasttext = AttentionModel(
            self.hidden_size, self.bidirectional, self.linear1_size, self.linear2_size
        )
        self.attention_mean = AttentionModel(
            self.hidden_size, self.bidirectional, self.linear1_size, self.linear2_size
        )

        # linear layers
        self.linear1_metadata = nn.Linear(13, self.linear1_size)
        self.linear2_metadata = nn.Linear(self.linear1_size, self.linear2_size)
        self.linear_attention = nn.Linear(3 * self.linear2_size, self.linear2_size)
        self.linear_out = nn.Linear(2 * self.linear2_size, 9)

        # utils
        self.relu = nn.ReLU6(inplace=True)
        self.dropout_linear = nn.Dropout(self.drop_linear)

    def forward(
        self,
        positive_navec,
        negative_navec,
        positive_fasttext,
        negative_fasttext,
        positive_mean,
        negative_mean,
        metadata,
    ):

        # lstm outputs
        positive_navec_out, _ = self.lstm_positive_navec(positive_navec)
        negative_navec_out, _ = self.lstm_negative_navec(negative_navec)
        positive_fasttext_out, _ = self.lstm_positive_fasttext(positive_fasttext)
        negative_fasttext_out, _ = self.lstm_negative_fasttext(negative_fasttext)
        positive_mean_out, _ = self.lstm_positive_mean(positive_mean)
        negative_mean_out, _ = self.lstm_negative_mean(negative_mean)

        # Attention
        attention_navec = self.attention_navec(positive_navec_out, negative_navec_out)
        attention_fasttext = self.attention_navec(
            positive_fasttext_out, negative_fasttext_out
        )
        attention_mean = self.attention_navec(positive_mean_out, negative_mean_out)
        attention_out = torch.cat(
            (attention_navec, attention_fasttext, attention_mean), dim=-1
        )
        attention_out = self.dropout_linear(attention_out)
        attention_out = self.relu(self.linear_attention(attention_out))

        # metadata
        metadata = self.relu(self.linear1_metadata(metadata))
        metadata = self.dropout_linear(metadata)
        metadata_out = self.relu(self.linear2_metadata(metadata))

        # out
        out = torch.cat((attention_out, metadata_out), dim=-1)
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
        (
            positive_navec,
            negative_navec,
            positive_fasttext,
            negative_fasttext,
            positive_mean,
            negative_mean,
            metadata,
            target,
        ) = batch
        out = self(
            positive_navec,
            negative_navec,
            positive_fasttext,
            negative_fasttext,
            positive_mean,
            negative_mean,
            metadata,
        )
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
        (
            positive_navec,
            negative_navec,
            positive_fasttext,
            negative_fasttext,
            positive_mean,
            negative_mean,
            metadata,
            target,
        ) = batch
        out = self(
            positive_navec,
            negative_navec,
            positive_fasttext,
            negative_fasttext,
            positive_mean,
            negative_mean,
            metadata,
        )
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
            positive_navec, negative_navec, positive_fasttext, negative_fasttext, positive_mean, negative_mean, metadata = batch
        except:
            positive_navec, negative_navec, positive_fasttext, negative_fasttext, positive_mean, negative_mean, metadata, target = batch
        out = self(positive_navec, negative_navec, positive_fasttext, negative_fasttext, positive_mean, negative_mean, metadata)
        return out
