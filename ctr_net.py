import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ControlNet(nn.Module):
    def __init__(self, device=0):
        super(ControlNet, self).__init__()
        
        self.h = h = 16
        
        self.conv1 = nn.Conv2d(3+20, h, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(h, h, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(h, h, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(h, h, kernel_size=3, stride=1)
        
        self.ln1 = nn.LayerNorm([h, 32, 32])
        self.ln2 = nn.LayerNorm([h, 16, 16])
        self.ln3 = nn.LayerNorm([h, 8, 8])
        self.ln4 = nn.LayerNorm([h, 4, 4])
        
        self.fc1 = nn.Linear(h*4*4+20, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7)
        
        self.ln5 = nn.LayerNorm([128])
        self.ln6 = nn.LayerNorm([128])
        
        self._init_weights()

    def forward(self, vision, sentence, state):

        sentence = sentence.view(-1, 20, 1, 1).expand(-1, 20, 64, 64)

        x = torch.cat((vision, sentence), 1) # 23x64x64
    
        x = F.elu(self.ln1(self.conv1(x))) # hx32x32
        x = F.elu(self.ln2(self.conv2(x))) # hx16x16
        x = F.elu(self.ln3(self.conv3(x))) # hx8x8
        x = F.elu(self.ln4(self.conv4(x))) # hx4x4
        
        x = x.view(x.shape[0], self.h*4*4) # h*4*4
        x = torch.cat((x, state), 1)  # h*4*4+20

        x = F.elu(self.ln5(self.fc1(x))) # 128
        x = F.elu(self.ln6(self.fc2(x))) # 128
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
                init.constant_(m.weight, 1)
                # m.register_parameter('weight', None)
                init.constant_(m.bias, 0)
