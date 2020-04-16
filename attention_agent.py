import cv2
import numpy as np

from numpy.lib.stride_tricks import as_strided

import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector

torch.manual_seed(1)


def extract_patches(image, num_patches, patch_size=8, stride=1):
    h, w, d = image.shape
    patches = np.zeros((num_patches, patch_size, patch_size, d))
    locations = np.zeros((num_patches, 2))
    i = 0
    for x in range(0, w-patch_size, stride):
        for y in range(0, h-patch_size, stride):
            patch = image[x:x+patch_size, y:y+patch_size, :]
            patches[i] = patch
            locations[i] = [x, y]
            i += 1
    return patches, locations


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(0)


def f(k, locations):
    return locations[k]


class AttentionAgent():
    
    def __init__(self, L=96, M=7, S=4, K=10):
        self.L = L # image dimension after resizing is LxL
        self.M = M # patch size
        self.S = S # stride when extracting patches
        self.d = 4 # dimension of the transformed space
        self.K = K # number of best patches
        self.N = ((L-M)//S + 1)**2 # number of patches
        self.d_in = self.M * self.M * 3

        self.W_k = np.random.random((self.d_in, self.d))
        self.W_q = np.random.random((self.d_in, self.d))

        self.controller = nn.LSTM(20, 3, 16)


    def act(self, obs):
        obs = obs / 255. # normalize image on interval [0,1]
        obs = cv2.resize(obs, dsize=(self.L, self.L), interpolation=cv2.INTER_CUBIC)
        patches, locations = extract_patches(obs, self.N, self.M, self.S)
        patches = np.reshape(patches, (self.N, self.d_in))

        p1 = np.matmul(patches, self.W_k)
        p2 = np.matmul(patches, self.W_q)
        A = softmax(1/self.d_in * np.matmul(p1, p2.transpose()))

        scores = np.sum(A, axis=0)
        ind = np.argsort(scores)
        ind = ind[:self.K] # take K best

        positions = np.array([f(k, locations) for k in ind]) # index to position mapping
        positions = np.reshape(positions, (1,-1))

        #w = parameters_to_vector(self.controller.parameters()).detach().numpy()

        with torch.autograd.no_grad():
            positions = torch.from_numpy(positions.reshape(1, 1, -1))
            action, _ = self.controller(positions.float())
        
        action = np.reshape(action.numpy(), (-1,))
        return action
