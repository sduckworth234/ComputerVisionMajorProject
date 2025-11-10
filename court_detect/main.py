from dataset import courtDataset
import torch
import torch.nn as nn
from base_trainer import train
from base_validator import val
import os
from tensorboardX import SummaryWriter
from tracknet import BallTrackerNet
import argparse
from torch.optim import lr_scheduler

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--exp_id', type=str, default='default', help='path to saving results')
    parser.add_argument('--num_epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='number of steps per one epoch')
    parser.add_argument('--train_subset', type=int, default=None, help='limit train dataset to the first N samples')
    parser.add_argument('--val_subset', type=int, default=None, help='limit validation dataset to the first N samples')
    args = parser.parse_args()
    
    train_dataset = courtDataset('train', max_samples=args.train_subset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    val_dataset = courtDataset('val', max_samples=args.val_subset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model = BallTrackerNet(out_channels=15)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    exps_path = './exps/{}'.format(args.exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=0)

    val_best_accuracy = 0
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.steps_per_epoch)
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)

        if (epoch > 0) & (epoch % args.val_intervals == 0):
            val_loss, tp, fp, fn, tn, precision, accuracy = val(model, val_loader, criterion, device, epoch)
            print('val loss = {}'.format(val_loss))
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/tp', tp, epoch)
            log_writer.add_scalar('Val/fp', fp, epoch)
            log_writer.add_scalar('Val/fn', fn, epoch)
            log_writer.add_scalar('Val/tn', tn, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/accuracy', accuracy, epoch)
            if accuracy > val_best_accuracy:
                val_best_accuracy = accuracy
                torch.save(model.state_dict(), model_best_path)     
            torch.save(model.state_dict(), model_last_path)


