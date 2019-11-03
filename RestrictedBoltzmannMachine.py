#Restricted Boltzmann Machine
import torch


class RBM(nn.Module):
    '''
    A class containing the model of a single restricted Boltzmann machine.
    
    '''
    
    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3):
        super(RBM,self).__init__()
        self.desc = "RBM"
          
        self.num_visible = num_visible                                #number of visible nodes
        self.num_hidden = num_hidden                                  #number of hidden nodes
        self.k = k                                                    #number of Gibbs samplings
        self.learning_rate = learning_rate

        self.weights = torch.randn(num_visible, num_hidden) * 0.1     #initialize weight to random value
        self.visible_bias = torch.ones(num_visible) * 0.5             #initialize the visible bias to 0.5
        self.hidden_bias = torch.zeros(num_hidden)                    #initialize hidden bias to 0

    def sample_hidden(self, visible_probabilities):
        '''
        The function return an array of probabilites sampled at the hidden layer. 
        For more information refer to paper "Deep Belief Networks for phone recognition"
        page 2.
        '''
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        '''
        The function return an array of probabilites sampled at the visible layer. 
        For more information refer to paper "Deep Belief Networks for phone recognition"
        page 2.
        '''
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities
    
    
    def _sigmoid(self, x):
        '''
        Standard definition of a sigmoid function
        '''
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        '''
        Returns a torch array of random probabilities
        '''
        random_probabilities = torch.rand(num)


        return random_probabilities

    def contrastive_divergence(self, input_data):
        return error

