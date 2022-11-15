import torch 
import numpy as np
from tqdm import tqdm, trange
import os
import argparse


from model import Glow, Discriminator
from utils import AddNoise, get_optimizer, NLLLoss, weights_init_normal

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn import DataParallel
from torch.optim import lr_scheduler



def one_step_training(inputs, samples, optimizer, D, criterion):
    optimizer.zero_grad()
    inputs = inputs.cuda(non_blocking=True)
    valid = torch.Tensor(inputs.shape[0], 1).fill_(1.0).cuda()
    fake = torch.Tensor(inputs.shape[0], 1).fill_(0.0).cuda()
    Dxr = D(inputs)
    Dxf = D(samples)
    real_loss = criterion(Dxr, valid)
    fake_loss = criterion(Dxf, fake)
    loss = (real_loss+fake_loss)/2
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().numpy()

def one_step_eval(inputs, samples, D, criterion):
    inputs = inputs.cuda(non_blocking=True)
    valid = torch.Tensor(inputs.shape[0], 1).fill_(1.0).cuda()
    fake = torch.Tensor(inputs.shape[0], 1).fill_(0.0).cuda()
    Dxr = D(inputs)
    Dxf = D(samples)
    real_loss = criterion(Dxr, valid)
    fake_loss = criterion(Dxf, fake)
    loss = (real_loss+fake_loss)/2
    return real_loss.detach().cpu().numpy(), fake_loss.detach().cpu().numpy(), loss.detach().cpu().numpy()

def generate_data(batch_size, model, x_dim):
    with torch.no_grad():
        sample = torch.randn(batch_size,
                          x_dim[1]*x_dim[2],
                          1,
                          1).cuda()
        x, _ = model(sample, None, True)
    return x


def load_model_F(model):
    state = torch.load(os.path.join('checkpoint', 'bestmodel.pth'))
    model.load_state_dict(state['model_state_dict'])
    return model

def save_model_D(model):
    state = {'model_state_dict': model.state_dict()}
    torch.save(state, os.path.join('checkpoint', 'bestmodel_D.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64,
                      help="The batch size to use for training.")
    parser.add_argument("--optimizer", type=str, default='adam',
                        help="Optimizer to use for training.")
    parser.add_argument("--gamma", type=float, default=0.5,
                      help="Gamma value to use for learning rate schedule.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--wd", type=float, default=0,
                      help="Weight decay to use for training.")
    parser.add_argument("--scheduler_decay", type=str, default='100-150',
                      help="Decay to use for the learning rate schedule.")
    parser.add_argument("--frequency_log_steps", type=int, default=5,
                      help="Print log for every step.")
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--num_levels", type=int, default=0,
                      help="Number of downsizing to use in the model.")
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


    # Model Pipeline
    x_dim = (1, 32, 32)
    args.num_levels = int(np.log2(x_dim[1]))-1


    print('Model Loading...')
    # Model Pipeline
    model = Glow(in_channels = 1,
                 num_channels = args.num_channels,
                 num_levels = args.num_levels, 
                 num_steps = args.num_steps,
                 params = args).cuda()
    model = load_model_F(model)
    model = DataParallel(model).cuda()
    model.eval()

    D = Discriminator()
    D = DataParallel(D).cuda()
    D.apply(weights_init_normal)
    print('Model loaded.')

    # Optimizer 
    optimizer = get_optimizer(args, D)

    milestones = list(map(int, args.scheduler_decay.split('-')))
    scheduler = lr_scheduler.MultiStepLR(
      optimizer, milestones=milestones, gamma=args.gamma)


    # define loss
    criterion = torch.nn.BCELoss()


    print('Start Training :')

    best_loss = np.inf
    with trange(args.epochs, desc="Epoch", unit="epoch") as te:
        for epoch in te:
            D.train()
            te.set_description(f'loss eval: {loss_eval : .2f} \t best loss: {best_loss : .2f}')
            with tqdm(train_loader, desc="Training", unit="batch", leave=False) as bt:
                for n_batch, (inputs, label) in enumerate(bt):
                    samples = generate_data(args.batch_size, model, x_dim)
                    loss = one_step_training(inputs, samples, optimizer, D, criterion)
                    bt.set_postfix(loss=loss)
            D.eval()
            loss_eval = 0 
            real_loss_eval = 0
            fake_loss_eval = 0
            with tqdm(test_loader, desc= "Evaluation",  leave=False) as bte:
                for n_batch, (inputs, label) in enumerate(bte):
                    samples = generate_data(args.batch_size, model, x_dim)
                    real_loss_eval_i, fake_loss_eval_i, loss_eval_i = one_step_eval(inputs, samples, D, criterion)
                    loss_eval += loss_eval_i 
                    real_loss_eval += real_loss_eval_i
                    fake_loss_eval += fake_loss_eval_i
                    bte.set_postfix(Real=real_loss_eval/((n_batch+1)*args.batch_size), 
                                    Fake=fake_loss_eval/(((n_batch+1)*args.batch_size)))
            loss_eval /= 10000
            if loss_eval <= best_loss:
                best_loss = loss_eval
                save_model_D(D)
            te.set_postfix(loss_eval=loss_eval, best_loss=best_loss)
            
    print('Training done')

        


