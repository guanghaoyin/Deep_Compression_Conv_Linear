import argparse
from net.models import LeNet,LeNet_5
import torch

from net.huffmancoding import huffman_encode_model,huffman_decode_model
import util

parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
parser.add_argument('model', type=str, help='saved quantized model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

model = torch.load(args.model)
huffman_encode_model(model)

model = LeNet_5(mask=True).to(device)
huffman_decode_model(model)
util.test(model, use_cuda)
