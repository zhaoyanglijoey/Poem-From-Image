import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, device, max_seq_length=20, sos_index=1):
        """
        Set the hyper-parameters and build the layers."
        :param embed_size:
        :param hidden_size: hidden size of GRU. Make sure equal to size of image features
        :param vocab_size:
        :param num_layers:
        :param max_seq_length:
        :param sos_index: start of sentence: index (usually 1)
        """
        super(DecoderRNN, self).__init__()
        self.sos_index = sos_index
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        self.device = device

    def forward(self, features, captions, lengths):
        """
        Decode image feature vectors and generates captions.
        :param features: image features. (batch_size, feature_size)
        :param captions: indices of words in captions including <SOS> and <EOS>. (batch_size, max_length)
        :param lengths: lengths of captions including <SOS> and <EOS> (batch_size, )
        :return: Distribution. (words_in_batch, size_vocab)
        """
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # make sure image features size equal to GRU hidden_size
        hidden_states = features.unsqueeze(0)
        lstm_outputs, _ = self.gru(packed, hidden_states)
        outputs = self.linear(lstm_outputs[0])
        return outputs

    def sample(self, features):
        """
        Generate captions for given image features using greedy search.
        :param features: image features. (batch_size, feature_size)
        :return:
        """
        sampled_ids = []
        batch_size = features.shape[0]

        # use <sos> as init input
        start = torch.full((batch_size, 1), self.sos_index, dtype=torch.int).long().to(self.device)  # start symbol index is 1
        inputs = self.embed(start)  # inputs: (batch_size, 1, embed_size)

        # use img features as init hidden_states
        hidden_states = features.unsqueeze(0)  # add one dimension as num_layers * num_directions (which is 1)

        for i in range(self.max_seg_length):
            lstm_outputs, hidden_states = self.gru(inputs, hidden_states)  # lstm_outputs: (batch_size, 1, hidden_size)
            outputs = self.linear(lstm_outputs.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

