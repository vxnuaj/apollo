# testing flax patterns
from flax import linen as nn

class NN(nn.Module):
    hidden_dim: int = 32
    output_dim: int = 10
    
    def setup(self):
        self.dense1 = nn.Dense(features = self.hidden_dim) 
        self.dense2 = nn.Dense(features = self.output_dim)
    
    def __call__(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        
nn = NN(
    hidden_dim = 32,
    output_dim = 10
)        