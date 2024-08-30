import os
import sys
import argparse
import scipy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from caesar_lnd_dataset import CAESARDataset

# system setting
device = torch.device('cpu')
dtype = torch.float32

# out feature dim
C_out = 1

# model
C_in = 355
k_eig = 128
    
# training setting
train = True if int(sys.argv[2])==1 else False
n_epoch = 20
lr = 1e-3
decay_every = 3
decay_rate = 0.5
method = "nuc"


# target lnd index
lnd = int(sys.argv[1])
print("Target landmark: ", lnd, train)

num_shapes = 2 # number of shapes in dataset

# testing setting
num_test = 1 # number of testing shapes

# paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "../data", "op_cache")
lnd_test_path = "model/faust_scan_remeshed/lnd_{}.pth".format(lnd)
pretrain_path = os.path.join(base_path, lnd_test_path)
lnd_save_path = "model/faust_scan_remeshed/lnd_{}.pth".format(lnd)
model_save_path = os.path.join(base_path, lnd_save_path)
dataset_path = os.path.join(base_path, "../../data/faust_scan_remeshed")
testing_path = os.path.join(base_path, "../../data/test")

# load training dataset
if train:
    train_dataset = CAESARDataset(dataset_path, train=True, k_eig=k_eig, lnd=lnd, num_shapes=num_shapes)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# load testing dataset    
if not train:
    test_dataset = CAESARDataset(testing_path, train=False, k_eig=k_eig, lnd=lnd, num_shapes=num_test)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=True)

    
# create the model
class LndNet(torch.nn.Module):
    def __init__(self, C_in, C_out, C_width, N_block, last_activation, outputs_at, dropout, lastfc):
        super(LndNet, self).__init__()
        self.net = torch.nn.Sequential()
        self.diffnet = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=C_out,
                                          C_width=C_width, 
                                          N_block=N_block, 
                                          diffusion_method='spectral',
                                          with_gradient_features=True,
                                          last_activation=last_activation,
                                          outputs_at=outputs_at, 
                                          dropout=dropout)
        self.net.add_module("diffnet", self.diffnet)
        self.lastfc = lastfc
        if lastfc:
            self.fc = torch.nn.Linear(C_out, 1)
            self.net.add_module("fc", self.fc)  
            
    def forward(self, shots, mass, L, evals, evecs, gradX, gradY):
        refined_shot = self.diffnet(shots, mass, L=L, evals=evals, evecs=evecs, 
                             gradX=gradX, gradY=gradY)
        if self.lastfc:
            return self.fc(refined_shot)
        else:
            return refined_shot
        
        
model = LndNet(C_in=C_in, C_out=C_out, C_width=128, N_block=4,
               last_activation=None, outputs_at='vertices', dropout=True,
               lastfc=True)

# load pre-trained model
if not train:
    print("Loading pre-trained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def correlation_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    
    return torch.sum(vx*vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

def train_nuc(epoch):
    
    # implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    # set model to 'train' mode
    model.train()
    optimizer.zero_grad()

    for data in tqdm(train_loader):
        verts = data["verts"].to(device)
        faces = data["faces"].to(device)
        frames = data["frame"].to(device)
        mass = data["massv"].to(device)
        L = data["L"].to(device)
        evals = data["evals"].to(device)
        evecs = data["evecs"].to(device)
        gradX = data["gradX"].to(device)
        gradY = data["gradY"].to(device)
        shots = data["SHOT"].to(device)
        lnd = data["lnd"].to(device).long()
        idx = data["tarlnd"]
        
        # refined shot by net
        feats = torch.Tensor(np.hstack((shots, verts))).float().to(device)
        shot_refined = model(feats, mass, L=L, evals=evals, evecs=evecs, 
                             gradX=gradX, gradY=gradY)
        scipy.io.savemat("../../vis/res/p.mat", {"p": shot_refined.cpu().detach().numpy()})
        # compute euclidean distances
        lnd_vert = verts[lnd[idx]]
        ecl_dist = torch.unsqueeze(torch.log(torch.norm(verts - lnd_vert, p=2, dim=1)+1e-15), 1)
        geo_dist = torch.unsqueeze(torch.norm(shots - shots[lnd[idx]], p=2, dim=1), 1)
        similarity = 0.05*(geo_dist)+0.95*ecl_dist
        scipy.io.savemat("../../vis/res/d.mat", {"p": similarity.cpu().detach().numpy()})
        
        
        # neighbour points
        neighbour_vert = np.argsort(np.squeeze(ecl_dist.cpu().detach().numpy()))[:20]
        
        # evaluate loss
        loss = -correlation_loss(shot_refined, similarity)\
            +0.001*torch.std(shot_refined[neighbour_vert])
        loss.backward()
        
        # step optimizer
        optimizer.step()
        optimizer.zero_grad()
        
    return loss

def train():
    print("Training...")
    for epoch in range(n_epoch):
        loss = train_nuc(epoch)
        print("Epoch {} - Loss {}".format(epoch, loss.data))
            
    torch.save(model.state_dict(), model_save_path)
    print(" ==> saving model to " + model_save_path)
    
def test():
    model.eval()
    result = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            verts = data["verts"].to(device)
            mass = data["massv"].to(device)
            L = data["L"].to(device)
            evals = data["evals"].to(device)
            evecs = data["evecs"].to(device)
            gradX = data["gradX"].to(device)
            gradY = data["gradY"].to(device)
            shots = data["SHOT"].to(device)
            lnd = data["lnd"].to(device).long()
            idx = data["tarlnd"]
            name = data["name"]
            
            
            feats = torch.Tensor(np.hstack((shots, verts))).float().to(device)
            shot_refined = model(feats, mass, L=L,
                                 evals=evals, evecs=evecs,
                                 gradX=gradX, gradY=gradY)
            lnd_vert = verts[lnd[idx]]
            ecl_dist = torch.unsqueeze(torch.norm(verts - lnd_vert, p=2, dim=1), 1)
            
            result.append([verts.cpu().detach().numpy(),
                           shot_refined.cpu().detach().numpy(),
                           ecl_dist.cpu().detach().numpy(),
                           name])
    return result
    
if __name__ == '__main__':
    if int(sys.argv[2])==1:
        train()
    else:
        res = test()

        for i in range(0,num_test):
            v = np.reshape(res[i][0], (len(res[i][0]), 3))
            p = res[i][1]
            g = res[i][2]
            scipy.io.savemat("../../vis/res/faust_scan_remeshed/res_{}_{}".format(lnd, res[i][-1]), {"p": p})
            print(f"Point-wise potential values for mesh {res[i][-1]} are saved")