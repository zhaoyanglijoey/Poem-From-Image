import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

def normalize(t):
    out = t / torch.norm(t, dim=-1, keepdim=True)
    return out


class DecoderRNN(nn.Module):
    """
    Example training file: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/train.py
    Example sampling file: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/sample.py
    """
    def __init__(self, embed_size, hidden_size, vocab_size, device, max_seq_length=70, sos_index=1):
        """
        Set the hyper-parameters and build the layers."
        :param embed_size: word embedding size
        :param hidden_size: hidden size of GRU. Make sure equal to size of image features
        :param vocab_size: size of vocabulary
        :param max_seq_length: the number of words at most
        :param sos_index: start of sentence: index (usually 1)
        """
        super(DecoderRNN, self).__init__()
        self.sos_index = sos_index
        self.device = device
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(0.3)
        self.linear.weight = self.embed.weight # tie weights

    def forward(self, features, poem_word_indices, lengths):
        """
        Decode image feature vectors and generates captions.
        :param features: image features. (batch_size, feature_size)
        :param poem_word_indices: indices of words in captions including <SOS> and <EOS>. (batch_size, max_length)
        :param lengths: lengths of captions including <SOS> and <EOS> (batch_size, )
        :return: Distribution. (words_in_batch, size_vocab)
        """
        """Decode image feature vectors and generates captions."""
        features = normalize(features)
        embeddings = self.embed(poem_word_indices)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # make sure image features size equal to GRU hidden_size
        hidden_states = features.unsqueeze(0)
        rnn_outputs, _ = self.rnn(packed, (hidden_states, hidden_states))
        outputs = self.linear(rnn_outputs[0])
        return outputs

    def sample(self, features):
        """
        Generate captions for given image features using greedy search.
        :param features: image features. (batch_size, feature_size)
        :return: contents of poem. (batch_size, max_seq_length)
        """
        features = normalize(features)
        batch_size = features.shape[0]
        sampled_ids = [torch.full((batch_size, ), 56, dtype=torch.long).to('cuda')]

        # use <sos> as init input
        start = torch.full((batch_size, 1), 56, dtype=torch.int).long().to(self.device)  # start symbol index is 1
        inputs = self.embed(start)  # inputs: (batch_size, 1, embed_size)

        # use img features as init hidden_states
        hidden_states = (features.unsqueeze(0), features.unsqueeze(0))  # add one dimension as num_layers * num_directions (which is 1)

        for i in range(self.max_seq_length):
            lstm_outputs, hidden_states = self.rnn(inputs, hidden_states)  # lstm_outputs: (batch_size, 1, hidden_size)
            outputs = self.linear(lstm_outputs.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


# class GRUDecoder(nn.Module):
#
#     def __init__(self, hidden_size, vocab_size):
#         super(GRUDecoder, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.vocab_size = vocab_size
#         self.max_words = 70   # contain the <S> and </S>
#         self.rnn_steps = self.max_words
#
#         self.embed_size = 1024  # size of elmo embedding
#
#         # elmo embedding function
#         self.elmo_embedder = embed_batch
#         # Or we can train our own embedding ?
#         # self.embed = nn.Embedding(self.vocab_size, self.image_feat_dim)
#
#         # GRU
#         self.gru = nn.GRU(self.embed_size, self.hidden_size, num_layers=1, batch_first=True)
#
#         # linear layer?
#         self.linear = nn.Linear(hidden_size, self.vocab_size)
#
#     def forward(self, features, poem, lengths):
#         """
#         Used for training
#         :param features: shape (batch_size, 1024)
#         :param poem: shape (batch_size, number_words, 1024)
#         :param lengths: (number_sentences) with each element being the the length of sentences (number_words)
#         :return:
#         """
#         # compute embedding of poem using elmo
#         embeddings = self.elmo_embedder(poem)
#         # concatenate features with poem features (each word)
#         embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
#         # throw to GRU
#         packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
#         hiddens, _ = self.gru(packed)
#         outputs = self.linear(hiddens[0])
#         return outputs
#
#     def sample(self, features, states=None):
#         """
#         Generate captions for given image features using greedy search.
#         :param features: shape (batch_size, 1024)
#         :param states: None
#         :return: indices of words
#         """
#         sampled_ids = []
#         inputs = features.unsqueeze(1)
#         for i in range(self.rnn_steps):
#             hiddens, states = self.gru(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
#             outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
#             _, predicted = outputs.max(1)                        # predicted: (batch_size)
#             sampled_ids.append(predicted)
#             inputs = self.elmo_embedder(predicted)                       # inputs: (batch_size, 1, embed_size)
#         sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
#         return sampled_ids
#
#
# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
#
#
# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=70):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         """
#
#         :param input: initial input token is the start-of-string <SOS>
#         :param hidden: use the feature of image as init hidden state
#         :param encoder_outputs:
#         :return:
#         """
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)