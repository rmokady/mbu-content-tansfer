import os
import sys
import torch
from torch import nn
from torch import optim
import torchvision
from mask_models import E1, E2, D_A, Disc, D_B
from mask_utils import save_imgs, save_model, load_model, CustomDataset
import argparse
import time



def train(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if args.gpu > -1:
        torch.cuda.set_device(args.gpu)

    print("Alpha1 is " + str(args.alpha1))
    print("Alpha2 is " + str(args.alpha2))
    print("Beta1 is " + str(args.beta1))
    print("Beta2 is " + str(args.beta2))
    print("Gama is " + str(args.gama))
    print("Delta is " + str(args.delta))
    print("discweight is " + str(args.discweight))

    _iter = 0

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.resize, args.resize)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    domA_train = CustomDataset(os.path.join(args.root, 'trainA.txt'), transform=transform)
    domB_train = CustomDataset(os.path.join(args.root, 'trainB.txt'), transform=transform)

    A_label = torch.full((args.bs,), 1)
    B_label = torch.full((args.bs,), 0)
    B_separate = torch.full((args.bs, args.sep * (args.resize // 64) * (args.resize // 64)), 0)

    e1 = E1(args.sep, args.resize // 64)
    e2 = E2(args.sep, args.resize // 64)
    d_a = D_A(args.resize // 64)
    disc = Disc(args.sep, args.resize // 64)
    d_b = D_B(args.resize // 64)

    mse = nn.MSELoss()
    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        d_a = d_a.cuda()
        d_b = d_b.cuda()
        disc = disc.cuda()
        A_label = A_label.cuda()
        B_label = B_label.cuda()
        B_separate = B_separate.cuda()
        mse = mse.cuda()
        bce = bce.cuda()
        l1 = l1.cuda()

    ae_params = list(e1.parameters()) + list(e2.parameters()) + list(d_a.parameters()) + list(d_b.parameters())
    ae_optimizer = optim.Adam(ae_params, lr=args.lr, betas=(0.5, 0.999))

    disc_params = disc.parameters()
    disc_optimizer = optim.Adam(disc_params, lr=args.disclr, betas=(0.5, 0.999))

    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = load_model(save_file, e1, e2, d_a, ae_optimizer, disc, disc_optimizer)

    e1 = e1.train()
    e2 = e2.train()
    d_a = d_a.train()
    d_b = d_b.train()
    disc = disc.train()

    print('Started training...')
    while True:

        domA_loader = torch.utils.data.DataLoader(dataset=domA_train, batch_size=args.bs, shuffle=True)
        domB_loader = torch.utils.data.DataLoader(dataset=domB_train, batch_size=args.bs, shuffle=True)

        if _iter >= args.iters:
            break

        for domA_img, domB_img in zip(domA_loader, domB_loader):
            if domA_img.size(0) != args.bs or domB_img.size(0) != args.bs:
                break

            if torch.cuda.is_available():
                domA_img = domA_img.cuda()
                domB_img = domB_img.cuda()
            else:
                domA_img = domA_img
                domB_img = domB_img

            domA_img = domA_img.view((-1, 3, args.resize, args.resize))
            domB_img = domB_img.view((-1, 3, args.resize, args.resize))

            ae_optimizer.zero_grad()

            A_common = e1(domA_img)
            A_separate = e2(domA_img)
            A_encoding = torch.cat([A_common, A_separate], dim=1)
            A_shaved_encoding = torch.cat([A_common, B_separate], dim=1)

            B_common = e1(domB_img)
            B_encoding = torch.cat([B_common, B_separate], dim=1)

            A_decoding, _ = d_b(A_encoding, d_a(A_shaved_encoding))
            B_decoding = d_a(B_encoding)

            #Reconstruction
            loss = args.gama * l1(A_decoding, domA_img) + args.delta * l1(B_decoding, domB_img)

            C_encoding = torch.cat([B_common, A_separate], dim=1)
            C_decoding, _ = d_b(C_encoding, domB_img)

            B_rec, _ = d_b(B_encoding, domB_img)
            A_rec, _ = d_b(A_encoding, domA_img)

            e1_b = e1(C_decoding)
            e2_a = e2(C_decoding)

            #Cycle loss
            loss += args.beta1 * mse(e1_b, B_common) + args.beta2 * mse(e2_a, A_separate)

            #Reconstruction 2
            mask_loss = args.alpha1 * l1(A_rec, domA_img) + args.alpha2 * l1(B_rec, domB_img)
            loss += mask_loss

            if args.discweight > 0:
                preds_A = disc(A_common)
                preds_B = disc(B_common)
                loss += args.discweight * (bce(preds_A, B_label) + bce(preds_B, B_label))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, 5)
            ae_optimizer.step()

            if args.discweight > 0:
                disc_optimizer.zero_grad()

                A_common = e1(domA_img)
                B_common = e1(domB_img)

                disc_A = disc(A_common)
                disc_B = disc(B_common)
                loss = bce(disc_A, A_label) + bce(disc_B, B_label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 5)
                disc_optimizer.step()

            if _iter % args.progress_iter == 0:
                print('Outfile: %s <<>> Iteration %d' % (args.out, _iter))
                sys.stdout.flush()

            if _iter % args.display_iter == 0 and _iter > 0:
                e1 = e1.eval()
                e2 = e2.eval()
                d_a = d_a.eval()
                d_b = d_b.eval()

                save_imgs(args, e1, e2, d_a, d_b, _iter)

                e1 = e1.train()
                e2 = e2.train()
                d_a = d_a.train()
                d_b = d_b.train()

            if _iter % args.save_iter == 0 and _iter > 0:
                save_file = os.path.join(args.out, 'checkpoint')
                save_model(save_file, e1, e2, d_a, d_b, ae_optimizer, disc, disc_optimizer, _iter)

            _iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=150000)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--discweight', type=float, default=0.005)
    parser.add_argument('--disclr', type=float, default=0.0002)
    parser.add_argument('--progress_iter', type=int, default=2500)
    parser.add_argument('--display_iter', type=int, default=10000)
    parser.add_argument('--save_iter', type=int, default=25000)
    parser.add_argument('--load', default='')
    parser.add_argument('--num_display', type=int, default=6)
    parser.add_argument('--alpha1', type=float, default=0.7)
    parser.add_argument('--alpha2', type=float, default=0.7)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.001)
    parser.add_argument('--gama', type=float, default=7.0)
    parser.add_argument('--delta', type=float, default=5.0)
    parser.add_argument('--gpu', type=int, default=-1)

    args = parser.parse_args()

    train(args)
