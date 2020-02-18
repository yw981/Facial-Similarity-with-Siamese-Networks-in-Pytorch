import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from net import SiameseNetwork
from torch import optim
import torch.nn.functional as F
from dataset import SiameseNetworkDataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


class Config():
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    train_batch_size = 64
    train_number_epochs = 10


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False)

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=0,
                              batch_size=Config.train_batch_size)

# net = SiameseNetwork().cuda()
net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        # img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
# show_plot(counter, loss_history)

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False)

test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

for i in range(10):
    _, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    # output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
    output1, output2 = net(Variable(x0), Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    print('{} {} Dissimilarity: {:.2f}'.format(i, label2, euclidean_distance.item()))
    # imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
