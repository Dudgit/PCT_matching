import numpy as np
from glob import glob

ROOT_TO_DATA = '/home/bdudas/PCT_tracking/data'
NUM_WPTS = 10_000

class pctDataset():
    def __init__(self,rootData:str = ROOT_TO_DATA,numWPTS:int = NUM_WPTS,ParticleNumber:int = 100,norm = True,dims:int = 3):
        self.ParticleNumber = ParticleNumber
        self.norm = norm
        self.dims = dims
        wpt_dirs = glob(rootData + f'/wpt_*')
        self.wpts = [wpt_dir.split('wpt_')[-1] for wpt_dir in wpt_dirs]
        self.allPath = np.array([f'{wpt_dir}/{i}.npy' for wpt_dir in wpt_dirs for i in range(numWPTS)])

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.allPath)
    
    def __normalize__(self, inpData):
        norm = lambda x: (x - x.min()) / (x.max() - x.min())
        return norm(inpData)
    
    def __getitem__(self, idx):
        path = self.allPath[idx]
        data = np.load(path)[:self.ParticleNumber,:,:self.dims]
        data = self.__normalize__(data) if self.norm else data
        return data