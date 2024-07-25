from __future__ import print_function
import argparse
import torch.optim as optim
from utils.log import logger
from utils.dataset import *
from tqdm import tqdm

from utils.train_utli import *
from test import test


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


def get_parsers():
    parser = argparse.ArgumentParser('FCA_FGVC', add_help=False)
    parser.add_argument('--epochs', type=int, default=300, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for training")
    parser.add_argument('--resume', type=str, default="", help="resume from saved model path")
    parser.add_argument('--dataset_name', type=str, default="cub", help="dataset name")
    parser.add_argument('--backbone', type=str, default="resnet50", help="backbone")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--hp', nargs='+')
    args, _ = parser.parse_known_args()
    return args


def train():
    # all parsers
    args = get_parsers()
    epochs = args.epochs
    batch_size = args.batch_size

    # load Data
    dataset_name = args.dataset_name
    classes_num, train_set, test_set = get_dataset(dataset_name)
    num_workers = 12 if torch.cuda.is_available() else 0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    exp_dir = dataset_name + '_' + args.backbone + '_FCA'
    log_name = '/train.txt'

    # load model
    net, num_ftrs = get_backbone(args.backbone, False)
    if net is not None:
        net = FCA(net, 512, num_ftrs, classes_num)
        net.load_state_dict(torch.load(args.resume))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        net = net.to(device)
        netp = torch.nn.DataParallel(net)
    else:
        device = torch.device('cpu')
        netp = net

    # optimizer
    deep_paras = [para for name, para in net.named_parameters() if "backbone" not in name]
    optimizer = optim.SGD(
        [{'params': deep_paras},
         {'params': net.backbone.parameters(), 'lr': args.lr / 10.0}],
        lr=args.lr, momentum=0.9, weight_decay=5e-4)

    hype_par = args.hp
    acc1_flag = False
    os.makedirs(f'./temp/feature/{dataset_name}', exist_ok=True)

    # train
    for epoch in range(1, epochs + 1):
        print('\nEpoch %d' % epoch)
        progress_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch}/{epochs}', total=len(train_loader),
                            leave=False)
        # lr set
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, epochs, args.lr)
        optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch, epochs, args.lr / 10.0)

        net.train()
        torch.autograd.set_detect_anomaly(True)
        num_correct = [0] * 5
        # Train
        for _, (inputs, targets) in progress_bar:
            if inputs.shape[0] < batch_size:
                continue
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            y1, y2, y3, y4, f1, f2, f3, f4, y1e, y2e, y3e, y4e, f1e, f2e, f3e, f4e, f1a = netp(inputs, hype_par, True, acc1_flag)

            # S1
            loss1 = smooth_CE(y1, targets, 0.8) * 1
            reward_loss1 = loss_rew(y1e.data, targets.data, f1, f1e)
            loss1t = loss1 + reward_loss1

            # S2
            loss2 = smooth_CE(y2, targets, 0.85) * 1
            reward_loss2 = loss_rew(y2e.data, targets.data, f2, f2e)
            loss2t = loss2 + reward_loss2

            # S3
            loss3 = smooth_CE(y3, targets, 0.9) * 1
            reward_loss3 = loss_rew(y3e.data, targets.data, f3, f3e)
            loss3t = loss3 + reward_loss3

            # S4
            loss4 = smooth_CE(y4, targets, 1) * 1
            reward_loss4 = loss_rew(y4e.data, targets.data, f4, f4e)
            loss4t = loss4 + reward_loss4
            loss = loss1t + loss2t + loss3t + loss4t
            loss.backward()
            optimizer.step()

            _, p1 = torch.max(y1.data, 1)
            _, p2 = torch.max(y2.data, 1)
            _, p3 = torch.max(y3.data, 1)
            _, p4 = torch.max(y4.data, 1)
            _, p5 = torch.max(y4e.data, 1)

            num_correct[0] += p1.eq(targets.data).cpu().sum()
            num_correct[1] += p2.eq(targets.data).cpu().sum()
            num_correct[2] += p3.eq(targets.data).cpu().sum()
            num_correct[3] += p4.eq(targets.data).cpu().sum()
            num_correct[4] += p5.eq(targets.data).cpu().sum()

        # result
        total = len(train_set)
        acc1 = 100. * float(num_correct[0]) / total
        acc2 = 100. * float(num_correct[1]) / total
        acc3 = 100. * float(num_correct[2]) / total
        acc4 = 100. * float(num_correct[3]) / total
        acc5 = 100. * float(num_correct[4]) / total
        if acc1_flag:
            var_feature(dataset_name, classes_num)
        acc1_flag = acc1 > 85

        progress_bar.close()

        result_str = 'train %d | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f  | acc5 = %.5f ' % (
            epoch, acc1, acc2, acc3, acc4, acc5)
        logger(exp_dir, log_name, result_str)

        torch.save(net.state_dict(), './{}/checkpoint.pth'.format(exp_dir))


if __name__ == "__main__":
    train()
