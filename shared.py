import nltk
from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter
from numpy import rec
from sklearn.metrics import accuracy_score, precision_score, recall_score

# natasha utils
segmenter = Segmenter()
emb = NewsEmbedding()
morph_vocab = MorphVocab()


def preprocessing(
    sent: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
):
    """
    Short preprocessing
    """
    # lowercase
    if lowercase:
        sent = sent.lower()

    # remove_punctuation
    if remove_punctuation:
        tokenizer = nltk.RegexpTokenizer(r"[а-я]+")
        sent = " ".join(tokenizer.tokenize(sent))

    # remove_stopwords
    if remove_stopwords:
        stopwords = nltk.corpus.stopwords.words("russian")
        sent = " ".join([w for w in sent.split() if w not in stopwords])

    # lemmatize
    if lemmatize:
        doc = Doc(sent)
        # Segmentation
        doc.segment(segmenter)

        # Morphology
        morph_tagger = NewsMorphTagger(emb)
        doc.tag_morph(morph_tagger)

        # Lemmatization
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        sent = " ".join([w.lemma for w in doc.tokens])

    return sent


def calculate_metrics(y_true, y_pred):
    # calculate metrcis
    acc = round(accuracy_score(y_true, y_pred), 3)
    pr = round(precision_score(y_true, y_pred, average="macro"), 3)
    rc = round(recall_score(y_true, y_pred, average="macro"), 3)
    # display scores
    print(f"Accuracy: {acc}, Precision: {pr}, Recall: {rc}")

    return acc, pr, rc
