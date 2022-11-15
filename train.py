import torch 
import numpy as np
import os
from tqdm import tqdm, trange
import argparse


from model import Glow
from utils import AddNoise, get_optimizer, NLLLoss

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import DataParallel
from torch.optim import lr_scheduler



def one_step_training(inputs, optimizer, model, criterion):
    optimizer.zero_grad()
    inputs = inputs.cuda()
    sldj  = torch.zeros(inputs.size(0), 
                            device=inputs.device)
    outputs, sldj = model(inputs, sldj, False)
    nll, bpd  = criterion(outputs, sldj)
    nll.backward()
    optimizer.step()
    return bpd.detach().cpu().numpy()

def one_step_eval(inputs, model, criterion):
    inputs = inputs.cuda()
    sldj = torch.zeros(inputs.size(0), 
                            device=inputs.device)
    outputs, sldj = model(inputs, sldj, False)
    nll, bpd = criterion(outputs, sldj)
    return bpd.detach().cpu().numpy()*inputs.shape[0]

def save_model(model):
    state = {'model_state_dict': model.state_dict()}
    torch.save(state, os.path.join('checkpoint', 'bestmodel.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=256,
                      help="The batch size to use for training.")
    parser.add_argument("--optimizer", type=str, default='adam',
                        help="Optimizer to use for training.")
    parser.add_argument("--gamma", type=float, default=0.5,
                      help="Gamma value to use for learning rate schedule.")
    parser.add_argument("--lr", type=float, default=0.001,
                      help="The learning rate to use for training.")
    parser.add_argument("--wd", type=float, default=0,
                      help="Weight decay to use for training.")
    parser.add_argument("--scheduler_decay", type=str, default='150-300-450',
                      help="Decay to use for the learning rate schedule.")
    parser.add_argument("--num_channels", type=int, default=16,
                      help="Number of channels to use in the model.")
    parser.add_argument("--num_features", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=5,
                      help="Depth of the model.")
    parser.add_argument("--batchnorm", default=False, action='store_false')
    parser.add_argument("--actnorm", default=True, action='store_true')
    parser.add_argument("--activation", type=str, default='relu') 

    args = parser.parse_args()



    # Data Pipeline
    print('Dataset loading...')
    x_dim = (1, 32, 32)
    args.num_levels = int(np.log2(x_dim[1]))-1
    transform = transforms.Compose([
                                transforms.Pad(padding=2),
                                transforms.ToTensor(),
                                AddNoise()])

    train_dataset = MNIST(root='data/mnist/', 
                                   train=True, transform=transform, download=True)
    test_dataset = MNIST(root='data/mnist/', 
                                  train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    # Model Pipeline
    model = Glow(in_channels = 1,
                 num_channels = args.num_channels,
                 num_levels = args.num_levels, 
                 num_steps = args.num_steps,
                 params = args).cuda()

    # model = DataParallel(model).cuda()
    print('Model loaded.')
    print(model)
    # Optimizer 
    optimizer = get_optimizer(args, model)

    milestones = list(map(int, args.scheduler_decay.split('-')))
    scheduler = lr_scheduler.MultiStepLR(
      optimizer, milestones=milestones, gamma=args.gamma)


    # define loss
    criterion = NLLLoss()

    print('Start Training :')
    best_bpd = np.inf
    with trange(args.epochs, desc="Epoch", unit="epoch") as te:
        for epoch in te:
            model.train()
            with tqdm(train_loader, desc="Training", unit="batch", leave=False) as bt:
                for n_batch, (inputs, label) in enumerate(bt):
                    bpd = one_step_training(inputs, optimizer, model, criterion)
                    bt.set_postfix(bpd=bpd)
            model.eval()
            bpd_eval = 0
            with tqdm(test_loader, desc= "Evaluation",  leave=False) as bte:
                for n_batch, (inputs, label) in enumerate(bte):
                    bpd_eval += one_step_eval(inputs, model, criterion)
                    bte.set_postfix(bpd_eval=bpd_eval/((n_batch+1)*args.batch_size))
            bpd_eval /= 10000
            if bpd_eval <= best_bpd:
                best_bpd = bpd_eval
                save_model(model)
            te.set_postfix(bpd_eval=bpd_eval, best_bpd=best_bpd)
            
    print('Training done')

        


