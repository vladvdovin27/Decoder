from Decoder import Decoder
from torch import nn
import torch
from torch.utils.data import DataLoader
from dataset import VecDataset
import torchvision


def train_epoch(decoder, device, dataloader, loss_fn, optimizer):
    decoder.train()
    l = []

    for data, target in dataloader:
        target = torch.cuda.FloatTensor(target)
        data = data.to(device)

        decoded_data = decoder(data)
       
        loss = loss_fn(decoded_data, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('\t partial train loss (single batch): %f' % (loss.data))
        l.append(loss.data)
    
    print(sum(l) / len(l))


if __name__ == "__main__":
    print(torchvision.__version__)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Decoder()
    print(model)

    loss_fn = nn.MSELoss()
    lr = 0.01
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)

    # device = 'cpu'
    model.to(device)

    print('Dataset')
    dataset = VecDataset()
    print('Dataset ready')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    epochs = 200

    print("Learn")
    for _ in range(epochs):
        train_epoch(model, device, dataloader, loss_fn, optim)

    torch.save(model.state_dict(), 'decoder.pt')
