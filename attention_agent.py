import cv2
import numpy as np
import cma

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
        # Attention Agent parameters
        self.L = L # image dimension after resizing is LxL
        self.M = M # patch size
        self.S = S # stride when extracting patches
        self.d = 4 # dimension of the transformed space
        self.K = K # number of best patches
        self.N = ((L-M)//S + 1)**2 # number of patches
        self.d_in = self.M * self.M * 3

        # Learnable parameters
        self.W_k = np.random.random((self.d_in, self.d))
        self.W_q = np.random.random((self.d_in, self.d))
        self.controller = nn.LSTM(20, 3, 16)

        # Evolution parameters
        self.pop_size = 5
        self.rewards = np.zeros(self.pop_size)
        self.current_ix = 0

        self.num_parameters = 2 * self.d_in * self.d + np.sum(p.numel() for p in self.controller.parameters())
        self.es = cma.CMAEvolutionStrategy(self.num_parameters * [0.], 1.)
        self.parameters = np.array(self.es.ask())
        self.best_param = None
        self.best_reward = -np.inf
        self.vec2param()


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

        with torch.autograd.no_grad():
            positions = torch.from_numpy(positions.reshape(1, 1, -1))
            action, _ = self.controller(positions)
        
        action = np.reshape(action.numpy(), (-1,))
        return action


    def episode_end(self, reward):
        if reward > self.best_reward:
            self.best_param = self.parameters[self.current_ix]
            self.best_reward = reward
            np.save('best_param.npy', self.best_param)
        self.rewards[self.current_ix] = reward
        self.current_ix += 1

        if self.current_ix >= self.pop_size:
            self.es.tell(self.parameters, self.rewards)
            self.parameters = self.es.ask()
            self.rewards = np.zeros(self.pop_size)
            self.current_ix = 0
        else:
            self.vec2param()


    def vec2param(self):
        """
        Convert current vector of parameters into matrices W_k, W_q and LSTM controller weights.
        Input: vector of parameters
        """
        pos_ix = 0
        vec = self.parameters[self.current_ix] # get current vector of params
        self.W_k = vec[:self.d_in * self.d].reshape((self.d_in,self.d))
        pos_ix += self.d_in * self.d
        self.W_q = vec[pos_ix : pos_ix + self.d_in * self.d].reshape((self.d_in,self.d))
        pos_ix += self.d_in * self.d

        for p in self.controller.parameters():
            t_shape = p.data.shape
            prod = np.prod(t_shape)
            if len(t_shape) == 1: p.data = torch.from_numpy(vec[pos_ix:pos_ix + prod].flatten())
            else: p.data = torch.from_numpy(vec[pos_ix:pos_ix + prod].reshape(t_shape[0],-1))
            pos_ix += prod


    def test(self):
        self.parameters = self.best_param
        self.vec2param()
