import os
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class LoadData:
    def __init__(self, smoothing, horizon):
        # load data and do seperate train and test set
        X_density = glob.glob("./AEData_64/ac_density_*_smoothing_{}.npy".format(smoothing))
        X_complexity = glob.glob("./AEData_64/ev_density_*_smoothing_{}.npy".format(smoothing))
        Y = glob.glob("./AEData_32/ev_2class_density_*.npy")

        if len(X_density) != len(X_complexity):
            print("Input Dimension Mismatch! Quit!")
            return
        elif len(X_density) != len(Y):
            print("Input and Output Dimension Mismatch! Quit!")
            return

        for idx in range(len(X_density)):
            if idx == 0:
                self.X_density = np.load(X_density[idx])[None, ...]
                self.X_complexity = np.load(X_complexity[idx])[None, ...]
                self.Y = np.load(Y[idx])[None, ...]
            else:
                self.X_density = np.concatenate((self.X_density, np.load(X_density[idx])[None, ...]), axis=0)
                self.X_complexity = np.concatenate((self.X_complexity, np.load(X_complexity[idx])[None, ...]), axis=0)
                self.Y = np.concatenate((self.Y, np.load(Y[idx])[None, ...]), axis=0)

        # concatenate the raw data as X and Y
        # self.X: 27x240x64x64x2
        # self.Y: 27x240x64x64x1
        self.X = np.concatenate((self.X_density[..., np.newaxis], self.X_complexity[..., np.newaxis]), axis=-1)
        self.Y = self.Y[..., np.newaxis]

        # shift data based on time horizon
        shift = int(float(horizon) / 60.0)
        if shift == 0:
            pass
        else:
            self.X, self.Y = self.X[:, :-shift, ...], self.Y[:, shift:, ...]

        # scale the data for better training result, won't impact optimal
        #self.X, self.Y = self.X*100, self.Y*100
        self.Y = self.Y*100

    def dataloader(self, bs, ratio):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=ratio, shuffle=True)
        X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
        Y_train, Y_test = torch.Tensor(Y_train), torch.Tensor(Y_test)

        trainloader = DataLoader(TensorDataset(X_train, Y_train), batch_size=bs, shuffle=True, num_workers=0)
        testloader = DataLoader(TensorDataset(X_test, Y_test), batch_size=bs, shuffle=True, num_workers=0)
        return trainloader, testloader


class AutoEncoder(nn.Module):
    def __init__(self, nf):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(2, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(nf * 8, nf * 2, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nf * 2, nf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(nf * 2, 1, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(nf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(nf, 1, 4, 2, 1, bias=False),
            #nn.Tanh()
            #nn.ReLU()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    smoothing = 4
    horizon = 60
    nf = 8

    # # run with shell inputs
    # parser = argparse.ArgumentParser(description='Code Inputs.')
    # parser.add_argument('--smoothing', type=int, nargs='+', metavar='SMOOTHING')
    # parser.add_argument('--horizon', type=int, nargs='+', metavar='HORIZON')
    # parser.add_argument('--nf', type=int, nargs='+', metavar='NF')
    # args = parser.parse_args()
    # smoothing = args.smoothing[0]
    # horizon = args.horizon[0]
    # nf = args.nf[0]

    batch_size = 1
    test_ratio = 0.05
    epochs = 2000

    # save path
    save_path = './AE_in_64x64x2_out_32x32x1/Epochs_{}_Horizon_{}_Smooth_{}_nf_{}'.format(epochs, horizon, smoothing, nf)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    trainloader, testloader = LoadData(smoothing, horizon).dataloader(batch_size, test_ratio)

    # load computational graph
    net = AutoEncoder(nf)
    if torch.cuda.is_available():
        GPU = 1
        print('Training on GPU')
        net = net.cuda()
    print(net)

    # setup paramaters
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.00001)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.001)
    criterion = nn.MSELoss()

    # training and save results
    # training and save results
    trainloss = []
    for epoch in range(epochs):
        training_loss = 0

        for X_train, Y_train in trainloader:
            X_train = X_train.view(-1, X_train.shape[-1], X_train.shape[-3], X_train.shape[-2])
            Y_train = Y_train.view(-1, Y_train.shape[-1], Y_train.shape[-3], Y_train.shape[-2])

            if GPU:
                X_train, Y_train = Variable(X_train).cuda(), Variable(Y_train).cuda()

            optimizer.zero_grad()
            X_pred = net(X_train)
            loss = criterion(X_pred, Y_train)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() / X_train.shape[0]

        print('Epoch: {}/{} \t Mean Square Error Loss: {}'.format(epoch + 1, epochs, training_loss))
        trainloss.append(training_loss)

        # with torch.no_grad():
        #     # test dataset
        #     X_test, Y_test = next(iter(testloader))
        #     X_test = X_test.view(-1, X_test.shape[-1], X_test.shape[-3], X_test.shape[-2])
        #     Y_test = Y_test.view(-1, Y_test.shape[-1], Y_test.shape[-3], Y_test.shape[-2])
        #     if GPU:
        #         X_test, Y_test = Variable(X_test).cuda(), Variable(Y_test).cuda()
        #
        #     X_pred_test = net(X_test)
        #     testing_loss = criterion(X_pred_test, Y_test) / X_test.shape[0]
        # testloss.append(testing_loss)
        # print('Epoch: {}/{} \t TrainLoss: {} \t TestLoss: {}'.format(epoch + 1, epochs, training_loss, testing_loss))
    # np.save('{}/Testloss.npy'.format(save_path), np.asarray(testloss))

    np.save('{}/Trainloss.npy'.format(save_path), np.asarray(trainloss))

    # save trained model as pth
    torch.save(net.state_dict(), '{}/AE.pth'.format(save_path))

    # visualization
    plt.figure(figsize=(10, 5))
    plt.title("MSE During Training")
    plt.plot(trainloss[100:], label="TrainLoss, nf={}".format(nf))
    #plt.plot(testloss[100:], label="TestLoss, nf={}".format(nf))
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig('{}/Loss.png'.format(save_path), dpi=300)

    # test on testing dataset
    X_test, Y_test = next(iter(testloader))
    X_test = X_test.view(-1, X_test.shape[-1], X_test.shape[-3], X_test.shape[-2])
    Y_test = Y_test.view(-1, Y_test.shape[-1], Y_test.shape[-3], Y_test.shape[-2])
    if GPU:
        X_test = Variable(X_test).cuda()
        Y_pred = net(X_test)

    # save test images
    for num in range(Y_test.shape[0]):
        fig = plt.figure()
        plot = fig.add_subplot(1, 2, 1)
        plot.set_title('Original Image')
        imgplot = plt.imshow(Y_test[num, 0, :, :].cpu(), origin='lower', cmap='Blues')

        plot = fig.add_subplot(1, 2, 2)
        plot.set_title('Generated Image')
        imgplot = plt.imshow(Y_pred[num, 0, :, :].cpu().detach(), origin='lower', cmap='Blues')

        plt.savefig('{}/AE_{}.png'.format(save_path, num), dpi=300)
        plt.close(fig)
