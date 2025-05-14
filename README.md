# PCT Matching
This project is made for the Bergen pCT collaboration.  
  
The aim of this project to create a particle matching system for the Proton Computed Tomography detector system.  

## Project description
Proton Computed Tomography(pCT) aims to create a detector system for Hadron theraphy (this is a radiation based cancer treatment). But this is a charged particle tracking, which needs to be evaluated in real time for medical purposes. This is why the traditional algorithms are not used, because they are much slower, while deep neural nets can run much faster compared to them.  

  
My aim is to create a matching neural network, which will create a particle connection probability matrix. Our model will create $N \times d_{emb}$ matricies from the initial input data, where $N$ is the number of particles and $d_{emb}$ is the embedding dimension. If we create 2 matricies one corresponding for a detector layer $L$ and the other is to $L-1$ we can multiply this two and the result of the multiplication will be how close they are to each other. It's important to note, that it is because the input is normalized. We apply a softmax function to obtain the connection probabilities. The intuiton for this model is coming from the famous attention mechanism.
  
The next step which I'm currently working on, is that it is possible that matching algorithm will find more than 1 most likely match for a particle hit. I'm trying to translate the Sinkhorn matching algorithm into some kind of deep neural network structure or, something similar to that, which will solve this problem. The smoothlayers aiming to do this, since if I renormalize (between 0-1 not standardize) the rows and the columns in an alternating fashion I will make this matrix to have 1 to 1 matching.

## data.py
A simple data loader that will load files for training. Data is coming from [GATE](http://www.opengatecollaboration.org/) simulations. The simulations were prepared for the pCT detector system.

## config.yaml
Configuration file to make it easier to handle multiple runs. Since the [Wigner Scientific Computational laboratory](https://wsclab.wigner.hu/en) is part of the research there are multiple experiments running paralelly, that is why I use configuration yamls.

## model_utils.py
This file contains the model definitions and a trainer definiton. I might change the trainer to pytorch lightning trainer but for convinience I wrote my own.

## utils.py
Some function to be able to monitor training better.