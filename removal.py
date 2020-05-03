import argparse
import os
import torch
from mask_models import E1, E2, D_A, D_B_removal
from mask_utils import load_model_for_eval, load_model_for_eval_pretrained, removal


def eval(args):

    if args.gpu > -1:
        torch.cuda.set_device(args.gpu)

    e1 = E1(args.sep, args.resize // 64)
    e2 = E2(args.sep, args.resize // 64)
    d_a = D_A(args.resize // 64)
    d_b = D_B_removal(args.resize // 64)

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        d_a = d_a.cuda()
        d_b = d_b.cuda()

    if args.load != '':
        save_file = os.path.join(args.load, args.check)
        if not args.old_model:
            _iter = load_model_for_eval(save_file, e1, e2, d_a, d_b)
        else:
            _iter = load_model_for_eval_pretrained(save_file, e1, e2, d_a, d_b)

    e1 = e1.eval()
    e2 = e2.eval()
    d_a = d_a.eval()
    d_b = d_b.eval()

    if not os.path.exists(args.out) and args.out != "":
        os.mkdir(args.out)

    removal(args, e1, e2, d_a, d_b)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--check', default='checkpoint')
    parser.add_argument('--eval_folder', default='')
    parser.add_argument('--load', default='')
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--num_display', type=int, default=6)
    parser.add_argument('--amount', type=int, default=64)
    parser.add_argument('--ext', default='.png')
    parser.add_argument('--old_model', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=-1)

    args = parser.parse_args()

    eval(args)
