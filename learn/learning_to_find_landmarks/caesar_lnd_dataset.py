import os
import sys
import numpy as np

import scipy
import scipy.io
from plyfile import PlyData

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))
import diffusion_net

def pp_to_array(pp):
    file = open(pp, 'r')
    file.readline()
    file.readline()
    verts = []
    while True:
        l = file.readline().strip().split(' ')
        if l[0] == '</PickedPoints>':
            break
        v = [l[1].split('"')[1], l[2].split('"')[1], l[3].split('"')[1]]
        verts.append(v)
    return verts

def euclidean_dist_squared(lnds, verts):
    return scipy.spatial.distance_matrix(lnds, verts)

def find_min_dis_vert(lnds, verts):
    dist = np.sqrt(euclidean_dist_squared(lnds, verts))
    dist[np.isnan(dist)] = np.inf
    return np.argmin(dist, axis=1)

def read_ply_file(ply_file):
    ply_file = '{}.ply'.format(ply_file)
    ply_data = PlyData.read(ply_file)
    verts_data = ply_data['vertex'].data
    verts = np.zeros((verts_data.size, 3))
    verts[:, 0] = verts_data['x']
    verts[:, 1] = verts_data['y']
    verts[:, 2] = verts_data['z']  
    return verts

def input_to_batch_with_lnd(root_dir, mat_dict, name):
    dict_out = dict()

    for attr in ["vert", "triv", "SHOT"]:
        if mat_dict[attr][0].dtype.kind in np.typecodes["AllInteger"]:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.int32)
        else:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.float32)
            
    lnd = np.array(pp_to_array(os.path.join(root_dir, 'lnd/{}.pp').format(name[:-4]))).astype('float')
    dict_out["lnd_idx"] = find_min_dis_vert(lnd, read_ply_file(os.path.join(root_dir, 'ply/{}').format(name[:-4])))
    return dict_out

class CAESARDataset(Dataset):
    """
    CAESAR dataset with landmark information
    
    """
    def __init__(self, root_dir, train, k_eig, lnd, num_shapes, use_cache=True, op_cache_dir=None):
        
        self.train = train  # bool
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.num_shapes = num_shapes
        self.lnd_idx = lnd # index of target landmark
        
        self.verts_list = []
        self.faces_list = []
        self.shot_list = []
        self.lnd_list = []
        self.name_list = []
        
        self._init_data()
        
    def _get_caesar_file(self, folder_path):
        faust_files = [f for f in os.listdir(os.path.join(folder_path, 'processed'))
                       if os.path.isfile(os.path.join(os.path.join(folder_path, 'processed'), f))]
        faust_files.sort()
        return faust_files
    
    def _get_index(self, i):
        return i
        
    def _init_data(self):
        shape_names = self._get_caesar_file(self.root_dir)
        sample_shapes = np.random.choice(np.arange(len(shape_names)), self.num_shapes, replace=False)
        
        for i in range(self.num_shapes):
            file_name = os.path.join(os.path.join(self.root_dir, 'processed'), shape_names[self._get_index(i)])
            shape_name = shape_names[self._get_index(i)]
            load_data = scipy.io.loadmat(file_name)
            if "X" in load_data:
                data_curr = input_to_batch_with_lnd(self.root_dir, load_data["X"][0], shape_name)
                # scipy.io.savemat(os.path.join(file_name), data_curr)
                lnd_curr = torch.Tensor(data_curr["lnd_idx"])
                
            verts_curr = torch.Tensor(data_curr["vert"]).float()
            faces_curr = torch.Tensor(data_curr["triv"]-1).long()
            shots_curr = torch.Tensor(data_curr["SHOT"])
            self.verts_list.append(verts_curr)
            self.faces_list.append(faces_curr)
            self.shot_list.append(shots_curr)
            self.lnd_list.append(lnd_curr)
            self.name_list.append(shape_name)
            print("Loaded file ", file_name, "")
            
        print("Compute all operators")
        self.frames_list, self.massvec_list, \
        self.L_list, self.evals_list, \
        self.evecs_list, self.gradX_list, self.gradY_list\
            = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
            
    def __getitem__(self, i):
        shape = dict()
        shape["verts"] = self.verts_list[i]
        shape["faces"] = self.faces_list[i]
        shape["frame"] = self.frames_list[i]
        shape["massv"] = self.massvec_list[i]
        shape["L"] = self.L_list[i]
        shape["evals"] = self.evals_list[i]
        shape["evecs"] = self.evecs_list[i]
        shape["gradX"] = self.gradX_list[i]
        shape["gradY"] = self.gradY_list[i]
        shape["SHOT"] = self.shot_list[i]
        shape["lnd"] = self.lnd_list[i]
        shape["tarlnd"] = self.lnd_idx
        shape["name"] = self.name_list[i]
        return shape

    def __len__(self):
            return len(self.verts_list)
            