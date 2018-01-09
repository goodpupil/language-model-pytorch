import sys
import model
import argparse
import data
import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--weight', type=str, default='out/checkpoint.t7')

args = parser.parse_args()

torch.manual_seed(args.seed)

corpus = torch.load('out/corpus') # type: data.Corpus
ntokens = len(corpus.dictionary)
ncharsize = len(corpus.dictionary.idx2char)
N_MAX_CHAR_SIZE = 32

model = model.RNNModel(args.model, ntokens, ncharsize, N_MAX_CHAR_SIZE,
                       args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

model.load_state_dict(torch.load(args.weight))

print('Model load done!')
while True:
    print('input sentence, split by space >', end='')
    ip = input()
    ip_split = ip.split()

    word_vecs = torch.LongTensor(1, len(ip_split))
    char_vecs = []
    for i, w in enumerate(ip_split):
        print(' > changing word ', w, ' to vector')
        word_vecs[0][i] = corpus.dictionary.word2idx[w]
        char_vec = torch.zeros(N_MAX_CHAR_SIZE).long()
        for j, c in enumerate(w):
            char_vec[j] = corpus.dictionary.char2idx[c] + 1
        char_vecs.append(char_vec)
    char_final_vecs = torch.cat(char_vecs).view(1, len(ip_split), N_MAX_CHAR_SIZE)
    if args.cuda:
        word_vecs = word_vecs.cuda()
        char_final_vecs = char_final_vecs.cuda()

    print('Input word vec:')
    print(word_vecs)
    print('Input char vec:')
    print(char_final_vecs)

    hidden = model.init_hidden(1)
    output, _ = model((Variable(word_vecs), Variable(char_final_vecs)), hidden)
    print('Output:')
    output = output.data.squeeze()
    max_conf, output_idx = torch.max(output, dim=1)
    print(output_idx)
    print('Max confidence:', max_conf)
    for word_idx in output_idx:
        print(corpus.dictionary.idx2word[word_idx], ' ', end='')
    print('')

