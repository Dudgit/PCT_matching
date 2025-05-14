import torch
from omegaconf import OmegaConf
from data import pctDataset
from model_utils import DistModel
from model_utils import Trainer
from torch.utils.tensorboard import SummaryWriter
import argparse

ROOT_TO_DATA = '/home/bdudas/PCT_tracking/data'
def getLoader(conf):
    trainDataset = pctDataset(ROOT_TO_DATA +"/train",**conf.LoaderParams)
    trainLoader = torch.utils.data.DataLoader(trainDataset,batch_size=conf.TrainingParams.batch_size,shuffle=True)
    valDataset = pctDataset(ROOT_TO_DATA +"/test",**conf.ValLoaderParams)
    valLoader = torch.utils.data.DataLoader(valDataset,batch_size=conf.TrainingParams.batch_size)
    return trainLoader, valLoader

def main(conf,args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    trainLoader, valLoader = getLoader(conf)
    model = DistModel(**conf.ModelParams).to(device)
    myTrainer = Trainer(targetLayer=conf.TrainingParams.targetLayer,device=device)


    writer = SummaryWriter(comment= "_"+ args.comment)
    sample = next(iter(trainLoader))
    writer.add_graph(model, (sample[:,:,0].to(device),sample[:,:,1].to(device)))
    #model = torch.compile(model)
    
    with open(f'{writer.log_dir}/config.yaml', 'w') as f:
        OmegaConf.save(conf, f)
        
    myTrainer.train(model = model, loader = trainLoader,numEpochs=conf.TrainingParams.epochs,valLoader = valLoader,
                    writer=writer,optimizer=torch.optim.Adam,criterion=torch.nn.CrossEntropyLoss)


if __name__ == '__main__':
    conf = OmegaConf.load("config.yaml")
    conf.ValLoaderParams = conf.LoaderParams
    conf.ValLoaderParams.numWPTS = 2000
    conf.ModelParams.inDims = conf.LoaderParams.dims

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g','--gpu',type=int,default=0)
    argparser.add_argument('-c','--comment',type=str,default='')
    args = argparser.parse_args()

    conf.deviceNum = args.gpu
    #for sres in range(1,4):
    #args.comment = args.comment + f"_smoothRes{sres}"
    #conf.ModelParams.smoothres = sres
    main(conf,args)