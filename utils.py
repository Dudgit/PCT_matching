import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def showConfMatrix(res):
    fig = plt.figure()
    avgd = res.mean(axis=0)
    plt.imshow(avgd, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    return fig


@torch.no_grad()
def showConfMatrix2(res):
    fig = plt.figure()
    avgd = res.mean(axis=0)
    im = plt.imshow(avgd, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    # Add values on grid
    for i in range(avgd.shape[0]):
        for j in range(avgd.shape[1]):
            plt.text(j, i, f"{avgd[i, j]:.2f}", ha="center", va="center", color="w")
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
