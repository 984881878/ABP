import os
import torch
import argparse
import torch.nn as nn
from torch.optim import SGD
from util.utils import correct, name2model_and_loader
from torch.utils.tensorboard import SummaryWriter


def train(modelname, epoch, sub_dir, lr, momentum, weight, train_batch_size, eval_batch_size):
    model, loader, is_abp = name2model_and_loader(modelname)
    # model = model()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    train_criterion = nn.CrossEntropyLoss(reduction='mean')
    test_criterion = nn.CrossEntropyLoss(reduction='sum')
    train_loader, val_loader = loader(train_batch_size, eval_batch_size)

    logdir = f'log/{modelname}/{sub_dir}'
    if os.path.exists(logdir):
        raise ValueError(f'{logdir} already exists, please reassign value for --log [str]')
    writer = SummaryWriter(logdir)
    # if torch.cuda.is_available():
    model.cuda()
    best_model_acc = 0
    # best_model_path = None
    for e in range(epoch):
        model.train()
        for iteration, (img, label) in enumerate(train_loader):
            img = img.cuda().requires_grad_(True)
            label = label.cuda()

            pred = model(img)
            loss = train_criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                # loss_input_grad = img.grad.norm()
                loss_input_grad = (0.5 * (img.grad - 0) ** 2).sum()     # expect the grad converge to zero
            if is_abp:
                model.another_backward(img.grad)
                model.update_grad(weight)
            optimizer.step()

            if not iteration % 20:
                writer.add_scalar('Loss_input/train', loss_input_grad.item(), round((e * len(train_loader) + iteration) / 20))
                writer.add_scalar('Loss/train', loss.item(), round((e * len(train_loader) + iteration) / 20))
                # print('Train: Epoch {}, Iteration: {}, Loss {:.4f}'.format(e + 1, iteration, loss.item()))

        # eval
        model.eval()
        correct_nums = 0
        losses = 0
        with torch.no_grad():
            for iteration, (img, label) in enumerate(val_loader):
                img = img.cuda()
                label = label.cuda()
                pred = model(img)
                losses += test_criterion(pred, label).cpu()
                correct_nums += correct(pred, label).cpu()
        acc = correct_nums / len(val_loader.dataset)
        losses /= len(val_loader.dataset)
        writer.add_scalar('Loss/eval', losses, e)
        writer.add_scalar('Accuracy/eval', acc, e)
        print('Epoch [{}], Evaluate loss {:.4f}'.format(e + 1, losses))
        print('Epoch [{}], Evaluate accuracy {:.4f}'.format(e + 1, acc))
        if best_model_acc < acc:
            best_model_acc = acc
            # uncomment following code to save the best checkpoint
            # if best_model_path is not None:
            #     os.remove(best_model_path)
            # best_model_path = f'log/{model.__name__}/{sub_dir}/acc-{best_model_acc:.4f}-Epoch{e+1}.pth'
            # torch.save(model, best_model_path)
    writer.flush()
    print(f'{modelname}-{sub_dir} best accuracy: {best_model_acc}')
    return best_model_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.09)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight', type=float, default=0.2, help='balance importance between grad and another grad')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=500)
    args = parser.parse_args()

    if args.log is None:
        raise ValueError('--log [str] is must.')

    print(args)

    train(
        args.model, args.epoch, args.log, args.lr, args.momentum, args.weight,
        args.train_batch_size, args.eval_batch_size
    )

