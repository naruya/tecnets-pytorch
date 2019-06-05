import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ControlNet(nn.Module):
    def __init__(self, device=0):
        super(ControlNet, self).__init__()
        
        self.h = h = 16
        
        self.conv1 = nn.Conv2d(3+20, h, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(h, h, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(h, h, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(h, h, kernel_size=5, stride=2)
        
        self.ln1 = nn.LayerNorm([h, 61, 61])
        self.ln2 = nn.LayerNorm([h, 29, 29])
        self.ln3 = nn.LayerNorm([h, 13, 13])
        self.ln4 = nn.LayerNorm([h, 5, 5])
        
        self.fc1 = nn.Linear(h*5*5+20, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 7)
        
        self.ln5 = nn.LayerNorm([200])
        self.ln6 = nn.LayerNorm([200])
        
        self._init_weights()

    def forward(self, vision, sentence, state):

        sentence = sentence.view(-1, 20, 1, 1).expand(-1, 20, 125, 125)

        x = torch.cat((vision, sentence), 1) # 23x125x125
    
        x = F.elu(self.ln1(self.conv1(x))) # hx61x61
        x = F.elu(self.ln2(self.conv2(x))) # hx29x29
        x = F.elu(self.ln3(self.conv3(x))) # hx13x13
        x = F.elu(self.ln4(self.conv4(x))) # hx5x5
        
        x = x.view(x.shape[0], self.h*5*5) # h*5*5
        x = torch.cat((x, state), 1)  # h*5*5+20

        x = F.elu(self.ln5(self.fc1(x))) # 200
        x = F.elu(self.ln6(self.fc2(x))) # 200
        x = self.fc3(x)*2 # 7

        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # init.constant_(m.weight, 1)
                m.register_parameter('weight', None)
                init.constant_(m.bias, 0)
