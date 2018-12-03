import torch
from allennlp.commands.elmo import ElmoEmbedder


elmo_embedder = ElmoEmbedder(cuda_device=0) if torch.cuda.is_available() else ElmoEmbedder()


def embed_word(word):
    """
    :param word: str
    :return: embedding of word with length 1024
    """
    global elmo_embedder
    return elmo_embedder.embed_sentence([word])[2][0]


def embed_batch(sentences):
    """
    Embed a batch of sentences using elmo
    :param sentences: 2d array with shape (batch_size, length). Note that this requires all sentences are in the same length (must use padding for shorter sentences)
    :return: Tensor of embeddings with shape (batch_size, length, 1024) where 1024 is the size of elmo embedding
    """
    global elmo_embedder
    ret = elmo_embedder.embed_batch(sentences)
    embed_list = [torch.from_numpy(embed[2]) for embed in ret]
    return torch.stack(embed_list)
