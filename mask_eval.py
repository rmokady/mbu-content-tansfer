import argparse
import os
import torch
from mask_models import E1, E2, D_A, D_B
from mask_utils import save_imgs, load_model_for_eval


def eval(args):
    e1 = E1(args.sep, args.resize // 64)
    e2 = E2(args.sep, args.resize // 64)
    d_a = D_A(args.resize // 64)
    d_b = D_B(args.resize // 64)

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        d_a = d_a.cuda()
        d_b = d_b.cuda()

    if args.load != '':
        save_file = os.path.join(args.load, args.check)
        _iter = load_model_for_eval(save_file, e1, e2, d_a, d_b)

    e1 = e1.eval()
    e2 = e2.eval()
    d_a = d_a.eval()
    d_b = d_b.eval()

    if not os.path.exists(args.out) and args.out != "":
        os.mkdir(args.out)

    save_imgs(args, e1, e2, d_a, d_b, _iter)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--check', default='checkpoint')
    parser.add_argument('--load', default='')
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--crop', type=int, default=178)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--num_display', type=int, default=6)

    args = parser.parse_args()

    eval(args)
