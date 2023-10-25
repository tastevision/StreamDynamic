# 由chatgpt给出的batch内分支计算的方法
import torch
import torch.nn as nn

class BranchingModel(nn.Module):
    def __init__(self, num_branches, input_dim, hidden_dim, output_dim):
        super(BranchingModel, self).__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ) for _ in range(num_branches)])
        
    def forward(self, x):
        # x is a batch of input samples (batch_size, input_dim)
        
        # Use router to determine which branch to assign each sample to
        branch_indices = router(x)  # router(x) should return a tensor of shape (batch_size,)
        
        # Initialize an empty list to store branch outputs
        branch_outputs = []
        
        for i in range(self.num_branches):
            # Select samples assigned to the i-th branch
            branch_samples = x[branch_indices == i]
            if branch_samples.size(0) > 0:
                # If there are samples for this branch, compute the branch output
                branch_output = self.branches[i](branch_samples)
                branch_outputs.append(branch_output)
        
        if len(branch_outputs) > 0:
            # Concatenate the branch outputs and return
            return torch.cat(branch_outputs, dim=0)
        else:
            # If no samples were assigned to any branch, return None or a default value
            return None

class Router(nn.Module):
    def __init__(self, num_branches, input_dim):
        super(Router, self).__init__()
        self.num_branches = num_branches
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_branches)
        )
    
    def forward(self, x):
        # x is a batch of input samples (batch_size, input_dim)
        branch_indices = self.fc(x)  # Compute branch assignment logits
        return torch.argmax(branch_indices, dim=1)

# Example usage
batch_size = 16
input_dim = 100
hidden_dim = 50
output_dim = 10
num_branches = 3

# Create instances of the router and branching model
router = Router(num_branches, input_dim)
model = BranchingModel(num_branches, input_dim, hidden_dim, output_dim)

# Generate a random batch of input data
batch_input = torch.randn(batch_size, input_dim)

# Pass the input batch through the model with branch assignments
output = model(batch_input)

# You can now compute loss, perform backpropagation, and update the model's parameters
