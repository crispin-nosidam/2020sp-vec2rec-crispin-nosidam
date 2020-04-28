from gensim.models.doc2vec import TaggedDocument


def read_corpus(df, token_only=False):
    for index, row in df.iterrows():
        if token_only:
            yield row
        yield TaggedDocument(row, [index])