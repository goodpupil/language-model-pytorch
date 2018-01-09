import torch.nn as nn
from torch.autograd import Variable
import torch

class CharCNN(nn.Module):
    """Convert a word to a tensor through CNN
    input: (N, nchar, nchardim)
    output: (N, ninp)
    """
    def __init__(self, nchar, nchardim, ninp):
        super(CharCNN, self).__init__()
        self.nchar = nchar
        self.ninp = ninp
        self.nchardim = nchardim
        self.conv3 = nn.Conv2d(1, ninp // 3, (3, nchardim))
        # should be (20, nchar-2)
        self.conv2 = nn.Conv2d(1, ninp // 3, (2, nchardim))
        # should be (30, nchar-1)
        self.conv4 = nn.Conv2d(1, ninp - self.conv2.out_channels - self.conv3.out_channels, (4, nchardim))
        # should be (40, nchar-3)
        self.pooling3 = nn.MaxPool2d((1, nchar-2))
        self.pooling2 = nn.MaxPool2d((1, nchar-1))
        self.pooling4 = nn.MaxPool2d((1, nchar-3))
        self.highway_affine = nn.Linear(ninp, ninp)
        self.highway_activation = nn.Tanh()
        self.init_weights()

    def forward(self, input: torch.FloatTensor):
        assert(input.size(1) == self.nchar)
        assert(input.size(2) == self.nchardim)
        cnn_input = input.view(input.size(0), 1, input.size(1), input.size(2))
        conv2_output = self.conv2(cnn_input).squeeze()
        conv3_output = self.conv3(cnn_input).squeeze()
        conv4_output = self.conv4(cnn_input).squeeze()
        # ^ (N, 40, nchar-3)
        #print(conv2_output.unsqueeze(1).shape)
        pooling2_result = self.pooling2(conv2_output.unsqueeze(1)).squeeze()
        pooling3_result = self.pooling3(conv3_output.unsqueeze(1)).squeeze()
        pooling4_result = self.pooling4(conv4_output.unsqueeze(1)).squeeze()
        before_highway = torch.cat((pooling2_result, pooling3_result, pooling4_result), 1)
        p = self.highway_activation(self.highway_affine(before_highway))
        return p



    def init_weights(self):
        self.conv2.weight.data.uniform_(-.1, .1)
        self.conv3.weight.data.uniform_(-.1, .1)
        self.conv4.weight.data.uniform_(-.1, .1)
        self.conv2.bias.data.fill_(0)
        self.conv3.bias.data.fill_(0)
        self.conv4.bias.data.fill_(0)
        self.highway_affine.weight.data.uniform_(-.1, .1)
        self.highway_affine.bias.data.fill_(0)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.
        ncharsize: number of the type of characters
        nmaxcharsize: maximal number of characters in a word
    """

    def __init__(self, rnn_type, ntoken, ncharsize, nmaxcharsize, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ninp = ninp
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.nmaxcharsize = nmaxcharsize
        self.char_encoder = nn.Embedding(ncharsize, 6)
        self.char_embedding_size = 12
        self.input_char_cnn = CharCNN(nmaxcharsize, 6, self.char_embedding_size)
        self.softmax = nn.Softmax(dim=2)
        # (N, nmaxcharsize, 30) -> (N, 12)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp + self.char_embedding_size, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp + self.char_embedding_size, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.char_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_word_embedding(self, word_vecs, input_char_vecs):
        # input: (N, seq)
        # input_char_vecs (N, seq, MAX_N_CHAR)
        # output: (N, seq, self.ninp + self.char_embedding_size)
        word_emb = self.drop(self.encoder(word_vecs))
        char_encoded = self.char_encoder(input_char_vecs.view(-1, self.nmaxcharsize)) # N, nmaxcharsize, 30
        char_embedded = self.drop(self.input_char_cnn(char_encoded)).view(
            input_char_vecs.size(0), input_char_vecs.size(1), -1) # N * seq, 200
        return self.softmax(torch.cat((word_emb, char_embedded), dim=2))



    def forward(self, input_pack, hidden):
        """
        :param input_pack: (input: (N), input_char_vecs: (N, nmaxcharsize))
        :param hidden:
        :return:
        """
        input, input_char_vecs = input_pack
        # input: (N, seq)
        # input_char_vecs (N, seq, MAX_N_CHAR)
        emb = self.get_word_embedding(input, input_char_vecs)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return self.softmax(decoded.view(output.size(0), output.size(1), decoded.size(1))), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
