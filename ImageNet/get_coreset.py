import os
import torch
import argparse
import re
import numpy as np

parser = argparse.ArgumentParser('Coreset Selection for ImageNet')
parser.add_argument('--input', type=str, required=True, metavar='FILE',
                    help='path to input predicted probability folder')
parser.add_argument('--dataset_file_list', type=str, required=True, metavar='FILE',
                    help='path to relevant dataset file list')
parser.add_argument('--fraction', type=float, required=True, help='path to input predicted probability folder')
parser.add_argument('--window', type=int, required=True, help='window length')
parser.add_argument('--output', type=str, required=True, metavar='FILE', help='path to output folder')

args, unparsed = parser.parse_known_args()

with open(args.dataset_file_list, 'r') as f:
    nums = len(f.readlines())

pred_prob_his = torch.zeros(nums, args.window).double()

pred_prob_history_files = os.listdir(f'{args.input}/pred_prob_history')
pred_prob_history_files.sort(key=lambda x: int(re.findall("\d+", x)[0]))

uncertainty_his = []

for (idx, pred_prob_file) in enumerate(pred_prob_history_files):
    pred_prob = torch.from_numpy(np.loadtxt(f'{args.input}/pred_prob_history/{pred_prob_file}'))
    indices = pred_prob.nonzero().squeeze().cpu()

    pred_prob_his[indices] = torch.cat((pred_prob_his[indices, 1:], torch.unsqueeze(pred_prob[indices], 1).cpu()),
                                       dim=1)
    if idx >= args.window - 1 and idx < len(pred_prob_history_files) - 1:
        uncertainty_his.append((torch.std(pred_prob_his, dim=1) * 10).detach().numpy())

dynamic_uncertainty = np.mean(np.array(uncertainty_his), axis=0)

dyn_unc_rank = np.argsort(dynamic_uncertainty)

keep = set(dyn_unc_rank[-int(nums * args.fraction):])

with open(args.output, 'w') as g:
    with open(args.dataset_file_list, 'r') as f:
        for (idx, line) in enumerate(f):
            if idx in keep:
                g.write(line)

print(f'select {args.fraction * 100}% ({len(keep)}) of {args.dataset_file_list}({nums})')
