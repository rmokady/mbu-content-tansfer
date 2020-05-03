import os
import torch
import torch.utils.data as data
import torchvision
import torchvision.utils as vutils
from PIL import Image


def save_imgs(args, e1, e2, d_a, d_b, iters):
    test_domA, test_domB = get_test_imgs(args)
    exps = []
    exps2 = []
    exps3 = []
    exps4 = []

    for i in range(args.num_display):
        with torch.no_grad():
            if i == 0:
                filler = test_domB[i].unsqueeze(0).clone()
                exps.append(filler.fill_(0))
                exps2.append(filler.fill_(0))
                exps3.append(filler.fill_(0))
                exps4.append(filler.fill_(0))

            exps.append(test_domB[i].unsqueeze(0))
            exps2.append(test_domB[i].unsqueeze(0))
            exps3.append(test_domB[i].unsqueeze(0))
            exps4.append(test_domB[i].unsqueeze(0))

    for i in range(args.num_display):
        exps.append(test_domA[i].unsqueeze(0))
        exps2.append(test_domA[i].unsqueeze(0))
        exps3.append(test_domA[i].unsqueeze(0))
        exps4.append(test_domA[i].unsqueeze(0))
        separate_A = e2(test_domA[i].unsqueeze(0))
        common_A = e1(test_domA[i].unsqueeze(0))
        for j in range(args.num_display):
            with torch.no_grad():
                common_B = e1(test_domB[j].unsqueeze(0))
                BA_encoding = torch.cat([common_B, separate_A], dim=1)
                temp_decoding = d_a(BA_encoding)
                BA_decoding, mask = d_b(BA_encoding, test_domB[j])
                AA_encoding = torch.cat([common_A, separate_A], dim=1)
                AA_decoding, mask2 = d_b(AA_encoding, test_domA[j])
                A_decoding = d_a(AA_encoding)
                exps.append(BA_decoding)
                exps2.append(mask)
                exps4.append(mask2)
                if j % 2 == i % 2:
                    exps3.append(temp_decoding)
                else:
                    exps3.append(A_decoding)


    with torch.no_grad():
        exps = torch.cat(exps, 0)
        exps2 = torch.cat(exps2, 0)
        exps3 = torch.cat(exps3, 0)
        exps4 = torch.cat(exps4, 0)

    vutils.save_image(exps,
                      '%s/experiments_%06d.png' % (args.out, iters),
                      normalize=True, nrow=args.num_display + 1)
    vutils.save_image(exps2,
                      '%s/masks_%06d.png' % (args.out, iters),
                      normalize=True, nrow=args.num_display + 1)
    vutils.save_image(exps3,
                      '%s/d_a_%06d.png' % (args.out, iters),
                      normalize=True, nrow=args.num_display + 1)
    vutils.save_image(exps4,
                      '%s/segmentation_%06d.png' % (args.out, iters),
                      normalize=True, nrow=args.num_display + 1)


def removal(args, e1, e2, d_a, d_b):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.resize, args.resize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.eval_folder != '':

        class Faces(data.Dataset):
            """Faces."""

            def __init__(self, root_dir, transform, size, ext):
                self.root_dir = root_dir
                self.transform = transform
                self.size = size
                self.ext = ext
                self.files = [f for f in os.listdir(root_dir) if f.endswith(ext)]

            def __len__(self):
                return self.size  # number of images

            def __getitem__(self, idx):
                img_name = os.path.join(self.root_dir, self.files[idx])
                image = Image.open(img_name)
                sample = self.transform(image)
                return sample

        test_data = Faces(args.eval_folder, transform, args.amount, args.ext)
        domA_test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.bs, shuffle=False)
    else:
        domA_test = CustomDataset(os.path.join(args.root, 'testA.txt'), transform=transform)
        domA_test_loader = torch.utils.data.DataLoader(domA_test, batch_size=args.bs, shuffle=False)

    cnt = 0
    for test_domA in domA_test_loader:
        if torch.cuda.is_available():
            test_domA = test_domA.cuda()
        else:
            test_domA = test_domA

        test_domA = test_domA.view((-1, 3, args.resize, args.resize))
        for i in range(args.bs):
            separate_A = e2(test_domA[i].unsqueeze(0))
            common_A = e1(test_domA[i].unsqueeze(0))
            A_encoding = torch.cat([common_A, separate_A], dim=1)
            A_decoding = d_a(A_encoding)
            BA_decoding, mask = d_b(A_encoding, test_domA[i], A_decoding, args.threshold)

            exps = torch.cat([test_domA[i].unsqueeze(0), BA_decoding], 0)
            vutils.save_image(exps, '%s/%0d.png' % (args.out, cnt), normalize=True)
            print(cnt)
            cnt += 1
            if cnt == args.amount:
                break

def get_test_imgs(args):


    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.resize, args.resize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    domA_test = CustomDataset(os.path.join(args.root, 'testA.txt'), transform=transform)
    domB_test = CustomDataset(os.path.join(args.root, 'testB.txt'), transform=transform)

    domA_test_loader = torch.utils.data.DataLoader(domA_test, batch_size=64,
                                                   shuffle=False, num_workers=0)
    domB_test_loader = torch.utils.data.DataLoader(domB_test, batch_size=64,
                                                   shuffle=False, num_workers=0)

    for domA_img in domA_test_loader:
        if torch.cuda.is_available():
            domA_img = domA_img.cuda()
        domA_img = domA_img.view((-1, 3, args.resize, args.resize))
        domA_img = domA_img[:]
        break

    for domB_img in domB_test_loader:
        if torch.cuda.is_available():
            domB_img = domB_img.cuda()
        domB_img = domB_img.view((-1, 3, args.resize, args.resize))
        domB_img = domB_img[:]
        break

    return domA_img, domB_img


def save_model(out_file, e1, e2, d_a, d_b, ae_opt, disc, disc_opt, iters):
    state = {
        'e1': e1.state_dict(),
        'e2': e2.state_dict(),
        'd_a': d_a.state_dict(),
        'd_b': d_b.state_dict(),
        'ae_opt': ae_opt.state_dict(),
        'disc': disc.state_dict(),
        'disc_opt': disc_opt.state_dict(),
        'iters': iters
    }
    torch.save(state, out_file)
    return


def load_model(load_path, e1, e2, d_a, d_b, ae_opt, disc, disc_opt):
    state = torch.load(load_path)
    e1.load_state_dict(state['e1'])
    e2.load_state_dict(state['e2'])
    d_a.load_state_dict(state['d_a'])
    d_b.load_state_dict(state['d_b'])
    ae_opt.load_state_dict(state['ae_opt'])
    disc.load_state_dict(state['disc'])
    disc_opt.load_state_dict(state['disc_opt'])
    return state['iters']


def load_model_for_eval(load_path, e1, e2, d_a, d_b):
    state = torch.load(load_path)
    e1.load_state_dict(state['e1'])
    e2.load_state_dict(state['e2'])
    d_a.load_state_dict(state['d_a'])
    d_b.load_state_dict(state['d_b'])
    return state['iters']


def load_model_for_eval_pretrained(load_path, e1, e2, d_a, d_b):
    state = torch.load(load_path)
    e1.load_state_dict(state['e1'])
    e2.load_state_dict(state['e2'])
    d_a.load_state_dict(state['decoder'])
    d_b.load_state_dict(state['mustacher'])
    return state['iters']


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def default_loader(path):
    return Image.open(path).convert('RGB')



class CustomDataset(data.Dataset):
    def __init__(self, path, transform=None, return_paths=False,
                 loader=default_loader):
        super(CustomDataset, self).__init__()

        with open(path) as f:
            imgs = [s.replace('\n', '') for s in f.readlines()]

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + path + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    pass
