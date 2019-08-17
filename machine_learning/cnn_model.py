import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#%% Model
class Net3d1(nn.Module):
    def __init__(self, args):
        super(Net3d1, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 2, 5)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(2, 4, 5)
        self.pool2 = nn.AdaptiveAvgPool3d((7, 7, 7))
        self.conv3 = nn.Conv3d(4, 8, 5)
        # self.pool3 = nn.AdaptiveAvgPool3d((3, 3, 3)) # SAME!?
        
        self.lin_dim = 8 * 3 * 3 * 3
        self.fc1 = nn.Linear(self.lin_dim, 2)  

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        return x
    
    def get_embedding(self, x):
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape) # b,2,20,25,20
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) # b,4,7,7,7
        x = F.relu(self.conv3(x)) #self.pool3(F.relu(self.conv3(x)))
        # print(x.shape) # b,8,3,3,3
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,216
        return x
    
class Net3d2(nn.Module):
    def __init__(self, args):
        super(Net3d2, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 6, 5)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16,5)
        self.pool2 = nn.MaxPool3d(2, 2)
        
        self.lin_dim = 16*8*10*8
        self.fc1 = nn.Linear(self.lin_dim, 2)  

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        return x
    
    def get_embedding(self, x):
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape) # b,6,20,25,20
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) # b,16,8,10,8
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,10240
        return x
    
class Net3d3(nn.Module):
    def __init__(self, args):
        super(Net3d3, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 6, 5)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16,5)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(16, 32, 5)
        self.pool3 = nn.AdaptiveAvgPool3d((5, 5, 5))
        
        self.lin_dim = 32*5*5*5
        self.fc1 = nn.Linear(self.lin_dim, 2)  

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        return x
    
    def get_embedding(self, x):   
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape) # b,6,20,25,20
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) # b,16,8,10,8
        x = self.pool3(F.relu(self.conv3(x)))
        # print(x.shape) # b,32,5,5,5
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,4000
        return x
    
class Net3d4(nn.Module):
    def __init__(self, args):
        super(Net3d4, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 6, 5)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16,5)
        self.pool2 = nn.AdaptiveAvgPool3d((5, 5, 5))
        
        self.lin_dim = 2000
        self.fc1 = nn.Linear(self.lin_dim, 2)  
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        x = self.drop(x)
        return x
    
    def get_embedding(self, x):
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape) # b,6,20,25,20
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) # b,16,5,5,5
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,2000
        return x
    
class Net3d5(nn.Module):
    def __init__(self, args):
        super(Net3d5, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 6, 5)
        self.pool1 = nn.MaxPool3d(4, 4)
        self.conv2 = nn.Conv3d(6, 16,5)
        self.pool2 = nn.MaxPool3d(2, 2)
        
        self.lin_dim = 576
        self.fc1 = nn.Linear(self.lin_dim, 2)  

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        return x
    
    def get_embedding(self, x):
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape) # b,6,10,12,10
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) # b,16,3,4,3
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,576
        return x 
    
class Net3d6(nn.Module):
    def __init__(self, args):
        super(Net3d6, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 6, 5)
        self.pool1 = nn.MaxPool3d(4, 4)
        self.conv2 = nn.Conv3d(6, 16,5)
        self.pool2 = nn.MaxPool3d(4, 4)
        
        self.lin_dim = 32
        self.fc1 = nn.Linear(self.lin_dim, 2)  

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        return x
    
    def get_embedding(self, x):
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape) # b,6,10,12,10
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) # b,16,1,2,1
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,32
        return x
    
    
class Net3d7(nn.Module):
    """
    Version 2 with dropout 0.5.
    """
    
    def __init__(self, args):
        super(Net3d7, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 6, 5)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16,5)
        self.pool2 = nn.MaxPool3d(2, 2)
        
        self.lin_dim = 16*8*10*8
        self.fc1 = nn.Linear(self.lin_dim, 2)  
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        x = self.drop(x)
        return x
    
    def get_embedding(self, x):
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape) # b,6,20,25,20
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) # b,16,8,10,8
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,10240
        return x

class Net3d8(nn.Module):
    """
    Version 2 with dropout 0.3.
    """
    
    def __init__(self, args):
        super(Net3d8, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 6, 5)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16,5)
        self.pool2 = nn.MaxPool3d(2, 2)
        
        self.lin_dim = 16*8*10*8
        self.fc1 = nn.Linear(self.lin_dim, 2)  
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        x = self.drop(x)
        return x
    
    def get_embedding(self, x):
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape) # b,6,20,25,20
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) # b,16,8,10,8
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,10240
        return x
    
class Net3d9(nn.Module):
    """
    Version 2 with dropout 0.7.
    """
    
    def __init__(self, args):
        super(Net3d9, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 6, 5)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16,5)
        self.pool2 = nn.MaxPool3d(2, 2)
        
        self.lin_dim = 16*8*10*8
        self.fc1 = nn.Linear(self.lin_dim, 2)  
        self.drop = nn.Dropout(0.7)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        x = self.drop(x)
        return x
    
    def get_embedding(self, x):
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(F.relu(self.conv1(x)))
        # print(x.shape) # b,6,20,25,20
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) # b,16,8,10,8
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,10240
        return x
    
class Net3d10(nn.Module):
    """
    Version 2, with batchnorm.
    """
    def __init__(self, args):
        super(Net3d10, self).__init__()
        self.args = args

        self.conv1 = nn.Conv3d(1, 6, 5)
        self.bnor1 = nn.BatchNorm3d(6)
        self.pool1 = nn.MaxPool3d(2, 2)
        
        self.conv2 = nn.Conv3d(6, 16,5)
        self.bnor2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2, 2)
        
        self.lin_dim = 16*8*10*8
        self.fc1 = nn.Linear(self.lin_dim, 2)  

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc1(x)
        return x
    
    def get_embedding(self, x):
        # print(x.shape) # b,1,45,54,45
        x = self.pool1(self.bnor1(F.relu(self.conv1(x))))
        # print(x.shape) # b,6,20,25,20
        x = self.pool2(self.bnor2(F.relu(self.conv2(x))))
        # print(x.shape) # b,16,8,10,8
        x = x.view(-1, self.lin_dim)
        # print(x.shape) # b,10240
        return x