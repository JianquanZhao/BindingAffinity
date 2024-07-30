import torch
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from   torch.nn  import KLDivLoss
import torch.nn.functional as F

def mpnn_train(model,dataloader,optimizer,criterion, args,epoch_idx):
    model.train()
    predlist = []
    truelist = []
    names = []
    epoch_loss = 0.0
    for batch_id,data in enumerate(dataloader):
        y_h = data.y.unsqueeze(-1)
        y =  model(data,data.x)
        loss = criterion(y,y_h)
        predlist += y.squeeze(-1).cpu().tolist()
        truelist += data.y.cpu().tolist()
        names += data.name
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (batch_id+1)
    return model,names,predlist,truelist,epoch_loss

def gcn_predict(model,dataloader,criterion,args):
    model.eval()
    epoch_loss = 0.0
    predlist = []
    truelist = []
    names = []
    for batch_id,data in enumerate(dataloader):
        y =  model(data,data.x)
        y_h = data.y.unsqueeze(-1)
        mse = criterion(y,y_h)
        predlist += y.squeeze(-1).cpu().tolist()
        truelist += y_h.squeeze(-1).cpu().tolist()
        names += data.name
        epoch_loss += mse.detach().item()
    epoch_loss /= (batch_id+1)
    return names,predlist,truelist,epoch_loss

def cnn_train(model,dataloader,optimizer,criterion,args):
    model.train()
    predlist = []
    truelist = []
    epoch_loss = 0.0
    for batch_id,data in enumerate(dataloader):
        x1,x2,y = data[0],data[1],data[2]
        # torch.zeros().to(non_blocking=True)
        pre,_ =  model(x1,x2)
        label = y.unsqueeze(-1).to(torch.float32).to(args.device)
        loss = criterion(pre,label)
        predlist += pre.squeeze(-1).cpu().tolist()
        truelist += label.squeeze(-1).cpu().tolist()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return model,predlist,truelist,epoch_loss

def cnn_predict(model,dataloader,criterion,args):
    model.eval()
    epoch_loss=0
    predlist = []
    truelist = []
    names = []
    for batch_id,data in enumerate(dataloader):
        x1,x2,y,name = data[0],data[1],data[2],data[3]
        pre,att =  model(x1,x2)
        label = y.unsqueeze(-1).to(torch.float32).to(args.device)
        loss = criterion(pre,label)
        predlist += pre.squeeze(-1).cpu().tolist()
        truelist += label.squeeze(-1).cpu().tolist()
        names += name
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return names,predlist,truelist,epoch_loss,att


def transformer_train(model,dataloader,optimizer,criterion, args):
    model.train()
    predlist = []
    truelist = []
    names = []
    epoch_loss = 0.0
    for batch_id,graph in enumerate(dataloader):
        y =  model(graph[0],graph[3])
        y_h = graph[1].to(args.device)
        loss = criterion(y,y_h)
        predlist += y.squeeze(-1).cpu().tolist()
        truelist += y_h.squeeze(-1).cpu().tolist()
        names += graph[2]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (batch_id+1)
    return model,names,predlist,truelist,epoch_loss

def transformer_predict(model,dataloader,criterion,args):
    model.eval()
    epoch_loss = 0.0
    predlist = []
    truelist = []
    names = []
    for batch_id,graph in enumerate(dataloader):
        y =  model(graph[0],graph[3])
        y_h = graph[1].to(args.device)
        loss = criterion(y,y_h)
        predlist += y.squeeze(-1).cpu().tolist()
        truelist += y_h.squeeze(-1).cpu().tolist()
        names += graph[2]
        epoch_loss += loss.detach().item()
    epoch_loss /= (batch_id+1)
    return names,predlist,truelist,epoch_loss

def skempi_mpnn_train(model,dataloader,optimizer,criterion, args,epoch_idx):
    model.train()
    dg_predlist = []
    dg_truelist = []
    ddg_predlist = []
    ddg_truedlist = []
    names = []
    epoch_loss = 0.0
    for batch_id,data in enumerate(dataloader):
        y =  model(data,data.x) #预测突变型dg
        y_h = data.y.unsqueeze(-1) #ddg label
        origin = data.origin.unsqueeze(-1) #野生型dg
        mut_label = data.mutlabel.unsqueeze(-1) #突变型true label
        ddg = y - origin #相减得到预测ddg pred
        loss = criterion(ddg,y_h)
        dg_predlist += y.squeeze(-1).cpu().tolist()
        dg_truelist += mut_label.squeeze(-1).cpu().tolist()
        ddg_predlist += ddg.squeeze(-1).cpu().tolist()
        ddg_truedlist += data.y.cpu().tolist()
        names += data.name
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (batch_id+1)
    return model,names,dg_predlist,dg_truelist,epoch_loss,ddg_predlist,ddg_truedlist

def skempi_mpnn_predict(model,dataloader,criterion,args):
    model.eval()
    dg_predlist = []
    dg_truelist = []
    ddg_predlist = []
    ddg_truedlist = []
    names = []
    epoch_loss = 0.0
    for batch_id,data in enumerate(dataloader):
        y =  model(data,data.x) #预测突变型dg
        y_h = data.y.unsqueeze(-1) #ddg label
        origin = data.origin.unsqueeze(-1) #野生型dg
        mut_label = data.mutlabel.unsqueeze(-1) #突变型true label
        ddg = y - origin #相减得到预测ddg pred
        loss = criterion(ddg,y_h)
        dg_predlist += y.squeeze(-1).cpu().tolist()
        dg_truelist += mut_label.squeeze(-1).cpu().tolist()
        ddg_predlist += ddg.squeeze(-1).cpu().tolist()
        ddg_truedlist += data.y.cpu().tolist()
        names += data.name
        epoch_loss += loss.detach().item()
    epoch_loss /= (batch_id+1)
    return names,dg_predlist,dg_truelist,epoch_loss,ddg_predlist,ddg_truedlist