import torch
from torch import nn
class Conv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.GroupNorm(4,out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.conv(x)
        return x
        
class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self,encodeed_image_size=5):
        super(Encoder,self).__init__()
        self.encodeed_image_size = encodeed_image_size
        self.encoder = nn.Sequential(
            Conv(3,64),
            nn.MaxPool2d(3,2),
            Conv(64,128),
            Conv(128,256),
            nn.MaxPool2d(3,2),
            Conv(256,128),
            Conv(128,64),
            nn.MaxPool2d(3,2),
            Conv(64,32),
            Conv(32,32) 
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encodeed_image_size,encodeed_image_size))

    def forward(self,images):
        """
        Forward propagation

        :param images:images, a tensor of dimension (b,3,image_size,image_size)
        :return:encoded images
        """

        out = self.encoder(images) #(b,2018,imahe_size/8,imahe_size/8)
        out = self.adaptive_pool(out) ##(b,32,encoded_image_size,encoded_image_size)
        out = out.reshape((out.shape[0]),-1)# (b,5* 5* 32 )
        return out
                
class ActionNet(nn.Module):
    """
    Action Network
    """
    def __init__(self,in_dim=5*5*32,out_dim=3):
        super(ActionNet,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim,in_dim//4),
            nn.ReLU(),
            nn.Linear(in_dim//4,out_dim),
            # nn.Sigmoid()
        )
    def forward(self,images):
        """
        Forward propagation

        :param images:enoded vector, a tensor of dimension (b,encodeed_image_size*encodeed_image_size*32)
        :return:encoded images
        """
        out = self.fc(images)

        return out
    
class Encode_Action_Net(nn.Module):
    """
    Encode + Action Network
    """
    def __init__(self,in_dim=5*5*32,out_dim=3):
        super(Encode_Action_Net,self).__init__()
        #declare
        self.encoder = Encoder()
        self.Action_follow = ActionNet()
        self.Action_left = ActionNet()
        self.Action_right = ActionNet()
        self.Action_straight = ActionNet()

        #load weight
        self.encoder.load_state_dict(torch.load('weight/encoder1.pt'))
        self.Action_follow.load_state_dict(torch.load('weight/Action_follow1.pt'))
        self.Action_left.load_state_dict(torch.load('weight/Action_left1.pt'))
        self.Action_right.load_state_dict(torch.load('weight/Action_right1.pt'))
        self.Action_straight.load_state_dict(torch.load('weight/Action_straight1.pt'))
        
    def forward(self,images,cmd):
        """
        Forward propagation

        :param images:enoded vector, a tensor of dimension (b,encodeed_image_size*encodeed_image_size*32)
        :param cmd: 2 Follow lane, 3 Left, 4 Right, 5 Straight
        :return:action
        """
        out = self.encoder(images)
        if cmd == 2:
            action = self.Action_follow(out)
        elif cmd == 3:
            action = self.Action_left(out)
        elif cmd == 4:
            action = self.Action_right(out)
        elif cmd == 5:
            action = self.Action_straight(out)

        return action

