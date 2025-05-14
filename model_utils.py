import torch
import torch.nn as nn
from utils import acc, showConfMatrix, logMetrics
import tqdm


class EmbedHead(torch.nn.Module):
    def __init__(self,embedDim,inDims, numLayers:int = 4,dropout:float = 0.3):
        super(EmbedHead, self).__init__()
        layers = [nn.Linear(inDims,embedDim)]
        for i in range(1,numLayers):
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(embedDim*i,embedDim*(i+1)))
        
        self.prevEmbed = torch.nn.Sequential(*layers)
        self.currEmbed = torch.nn.Sequential(*layers)    
    def forward(self,x1,x2):
        x1 = self.prevEmbed(x1)
        x2 = self.currEmbed(x2)

        #Kinda like attention
        x = x1@x2.permute(0,2,1)
        return x

class SHModel(nn.Module):
    def __init__(self):
        super(SHModel, self).__init__()
        self.sh1 = nn.Softmax(dim = -2)
        self.sh2 = nn.Softmax(dim = -1)
        
    def forward(self,x):
        x = x - self.sh1(x)
        x = x - self.sh2(x)
        return x
    
class DistModel(torch.nn.Module):
    def __init__(self,embedDim,inDims,smoothres:int = 1):
        super(DistModel, self).__init__()
        self.ehead1 = EmbedHead(embedDim=embedDim,inDims=inDims)
        self.ehead2 = EmbedHead(embedDim=embedDim,inDims=inDims)
        smoothLayers = [*[SHModel()]*smoothres]
        self.smoothing =  torch.nn.Sequential(*smoothLayers)
        self.act = nn.Softmax(dim = -2)

    def forward(self,x1,x2):
        xe1 = self.ehead1(x1,x2)
        #xe2 = self.ehead2(x1,x2)
        x = xe1 #+ xe2
        x = self.smoothing(x)
        x = self.act(x)
        return x

class Trainer():
    def __init__(self,targetLayer,device):
        self.targetLayer = targetLayer
        self.device = device
    

    def trainStep(self,model,batch,target, isTrain:bool = True):
        x_curr = batch[:,:,self.targetLayer].to(self.device)
        x_prev = batch[:,:,self.targetLayer+1].to(self.device)
        output = model(x_curr,x_prev)
        loss = self.criterion(output,target)
        if isTrain:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return output, loss.item()
    
    def step(self,model,batch,loobObj,train:bool = True):
        target = torch.eye(batch.shape[1]).repeat((batch.shape[0],1,1)).to(self.device)
        output, loss = self.trainStep(model = model,batch=batch,target=target,isTrain=train)
        stepAcc = acc(output,target,axi=1)
        stepAcc2 = acc(output,target,axi=2)
        #fig = showConfMatrix(output.detach().cpu().numpy())
        loobObj.set_postfix({'Accuracy':f'{stepAcc:.4f}','Accuracy2':f'{stepAcc2:.4f}','Loss':f'{loss:.4f}'})
        return loss,stepAcc,stepAcc2
    
    def train(self,model,loader,numEpochs,valLoader,writer,optimizer,criterion):
        steps = len(loader)
        valsteps = len(valLoader)
        
        self.optimizer = optimizer(model.parameters())
        self.criterion = criterion()
        for epoch in range(numEpochs):
            epochLoss,epochAcc,epochAcc2 = 0.,0.,0.
            valepochLoss, valepochacc,valepochacc2 = 0.,0.,0.
            
            pbar = tqdm.tqdm(loader,desc=f'Epoch {epoch}/{numEpochs}',colour='green')
            model.train()
            for batch in pbar:
                loss, stepAcc, stepAcc2 =self.step(model=model,batch=batch,loobObj=pbar,train=True)
                epochLoss += loss
                epochAcc += stepAcc
                epochAcc2 += stepAcc2
            #output = self.trainStep(model = model, batch = batch, target =torch.eye(batch.shape[1]).repeat((batch.shape[0],1,1)).to(self.device), isTran = False)
            #fig = showConfMatrix(output.detach().cpu().numpy())
            logMetrics(epochAcc/steps,epochAcc2/steps,epochLoss/steps,fig=None,e=epoch,writer = writer)

            valPbar = tqdm.tqdm(valLoader,desc=f'Validation Epoch {epoch}/{numEpochs}',colour='green')
            model.eval()
            for batch in valPbar:
                valloss, valstepAcc, valstepAcc2 = self.step(model=model,batch=batch,loobObj=valPbar,train=False)
                valepochLoss += valloss
                valepochacc += valstepAcc
                valepochacc2 += valstepAcc2
            #output = self.trainStep(model = model, batch = batch, target =torch.eye(batch.shape[1]).repeat((batch.shape[0],1,1)).to(self.device), isTran = False)
            #fig = showConfMatrix(output.detach().cpu().numpy())
            logMetrics(valepochacc/valsteps,valepochacc2/valsteps,valepochLoss/valsteps,fig=None,e=epoch,writer=writer,c = 'Validation') 