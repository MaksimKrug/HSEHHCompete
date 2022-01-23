import nltk
from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter

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
