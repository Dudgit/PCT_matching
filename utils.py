import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def showConfMatrix(res):
    fig = plt.figure()
    plt.imshow(res[0])
    plt.colorbar()
    return fig

@torch.no_grad()
def acc(preds,target,axi):
    f = lambda x: torch.argmax(x,axis = axi)
    y_hat = f(preds)
    y_target = f(target)
    return y_hat.view(-1).eq(y_target.view(-1)).sum()/y_target.numel()


def logMetrics(acc,acc2,loss,fig,e,writer,c:str = 'Train'):
    writer.add_scalar(c+'/Loss',loss,e)
    writer.add_scalar(c+'/Accurcay/Horizontal',acc,e)
    writer.add_scalar(c+'/Accurcay/Vertical',acc2,e)
    if fig:
        writer.add_figure(c+'/Probabilty',fig,e)
        fig.close()
