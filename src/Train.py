import torch
from tqdm import tqdm

def train(model,optimizer,loss_fn,loader,device):
    epoch_loss =0.0
    model.train()
    
    for x,y in tqdm(loader):
        x = x.to(device,dtype = torch.float32)
        y = y.to(device, dtype = torch.float32)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
    
    return epoch_loss/len(loader)

def valid(model,loader,loss_fn,device):
    epoch_loss = 0.0
    model.eval()
    
    for x,y in loader:
        x = x.to(device,dtype = torch.float32)
        y = y.to(device, dtype = torch.float32)
        pred = model(x)
        loss = loss_fn(pred,y)
        epoch_loss +=loss.item()
    return epoch_loss/len(loader)

