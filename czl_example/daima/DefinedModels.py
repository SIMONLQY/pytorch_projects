import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch
import numpy as np
from random import sample

class ConvBNRelu3D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), padding=0,stride=1):
        super(ConvBNRelu3D,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.conv=nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm3d(num_features=self.out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class ConvBNRelu2D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1,padding=0):
        super(ConvBNRelu2D,self).__init__()
        self.stride = stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.conv=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class Enc_VAE(nn.Module):
    def __init__(self,channel,output_dim,windowSize):
        # 调用Module的初始化
        super(Enc_VAE, self).__init__()
        self.channel=channel
        self.output_dim=output_dim
        self.windowSize=windowSize
        self.conv1 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
        self.conv2 = ConvBNRelu2D(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=0)
        self.conv3 = ConvBNRelu2D(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=0)
        self.conv4 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.pool=nn.AdaptiveAvgPool2d((4, 4))
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False))
        self.mu=nn.Linear(512,output_dim)
        self.log_sigma=nn.Linear(512,output_dim)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = x.reshape([x.shape[0],-1,x.shape[3],x.shape[4]])
        x = self.conv4(x)
        map = self.pool(x)
        h = map.reshape([map.shape[0], -1])
        x = self.projector(h)
        mu=self.mu(x)
        log_sigma=self.log_sigma(x)
        sigma=torch.exp(log_sigma)
        return h, mu, sigma
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Dec_VAE(nn.Module):#input (-1,128)
    def __init__(self, channel=30,windowSize=25,input_dim=64):
        super(Dec_VAE, self).__init__()
        self.channel = channel
        self.windowSize=windowSize
        self.fc1=nn.Linear(in_features=input_dim, out_features=256)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(in_features=256,out_features=64*(self.windowSize-8)*(self.windowSize-8))
        self.relu2 = nn.ReLU()
        #reshape to (-1,64,19,19) and then deconv.
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32 * (self.channel - 12), kernel_size=(3, 3), stride=(1, 1),
                                          padding=0)
        # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
        self.bn3 = nn.BatchNorm2d(num_features=32 * (self.channel - 12))
        self.relu3 = nn.ReLU()
        # (-1, 96, 21, 21)
        self.deconv4= nn.ConvTranspose2d(in_channels=32 * (self.channel - 12), out_channels=64, kernel_size=(3,3), stride=(1, 1),
                                          padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.relu4= nn.ReLU()
        #(-1 64, 23, 23)
        self.deconv5= nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                          padding=0)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.relu5= nn.ReLU()
        #[-1, 32, 25, 25]
        self.deconv6= nn.ConvTranspose2d(in_channels=32, out_channels=15, kernel_size=(3, 3), stride=(1, 1),
                                          padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=15)
        self.relu6= nn.ReLU()
        #[-1, 15, 27, 27]
        self._initialize_weights()
    def forward(self, x1, x2):
        x1=self.fc1(x1)
        x1=self.relu1(x1)
        x1=self.fc2(x1)
        x1=self.relu2(x1)
        x1=x1.view(-1,64,self.windowSize-8,self.windowSize-8)
        x = self.deconv3(x)
        x = self.bn3(x)
        x=self.relu3(x)
        x=x.view(-1,32,self.channel-12,self.windowSize-6,self.windowSize-6)
        x = self.deconv4(x)
        x = self.bn4(x)
        x=self.relu4(x)
        x = self.deconv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.deconv6(x)
        x = self.bn6(x)
        # x = self.relu6(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Ba0tchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VAE(nn.Module):
    def __init__(self, channel,input_dim,output_dim,windowSize):
        # 调用Module的初始化
        super(VAE, self).__init__()
        self.channel=channel
        self.output_dim=output_dim
        self.windowSize=windowSize
        self.conv11 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
        self.conv12 = ConvBNRelu2D(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=0)
        self.conv13 = ConvBNRelu2D(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=0)
        self.conv14 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)

        '''
        # (15, 27, 27)
        self.conv11 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(5,5),stride=1,padding=0)
        # (16, 23, 23)
        self.conv12 = ConvBNRelu2D(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=0)
        # (32, 21, 21)
        self.conv13 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
        # (64, 19, 19)
        
        '''


        self.pool1=nn.AdaptiveAvgPool2d((4, 4))
        self.projector1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False))
        self.mu1=nn.Linear(512,output_dim)
        self.log_sigma1=nn.Linear(512,output_dim)
        self.conv21 = ConvBNRelu2D(in_channels=self.channel, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
        self.conv22 = ConvBNRelu2D(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        self.conv23 = ConvBNRelu2D(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv24 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)

        '''
        # (15, 27, 27)
        self.conv21 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(5,5),stride=1,padding=0)
        # (16, 23, 23)
        self.conv22 = ConvBNRelu2D(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=0)
        # (32, 21, 21)
        self.conv23 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
        # (64, 19, 19)

        '''
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.projector2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False))
        self.mu2 = nn.Linear(512, output_dim)
        self.log_sigma2 = nn.Linear(512, output_dim)

        self.fc11 = nn.Linear(in_features=input_dim, out_features=256)
        self.relu1 = nn.ReLU()
        self.fc12 = nn.Linear(in_features=256, out_features=64 * (self.windowSize - 8) * (self.windowSize - 8))
        self.relu2 = nn.ReLU()

        self.fc21 = nn.Linear(in_features=input_dim, out_features=256)
        self.fc22 = nn.Linear(in_features=256, out_features=64 * (self.windowSize - 8) * (self.windowSize - 8))

        # reshape to (-1,64,19,19) and then deconv.
        self.deconv3 = nn.ConvTranspose2d(in_channels=64*4, out_channels=32 * (self.channel - 12), kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=0)
        # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
        self.bn3 = nn.BatchNorm2d(num_features=32 * (self.channel - 12))
        self.relu3 = nn.ReLU()
        # (-1, 96, 21, 21)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32 * (self.channel - 12) + 64 * 2, out_channels=64, kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.relu4 = nn.ReLU()
        # (-1 64, 23, 23)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64 + 32 * 2, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                          padding=0)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.relu5 = nn.ReLU()
        # [-1, 32, 25, 25]
        self.deconv6 = nn.ConvTranspose2d(in_channels=32 + 16 * 2, out_channels=30, kernel_size=(3, 3), stride=(1, 1),
                                          padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=30)
        self.relu6 = nn.ReLU()
        # [-1, 15, 27, 27]
        self._initialize_weights()

    def forward(self, x1, x2):
        x11 = self.conv11(x1) #16,25,25
        x12 = self.conv12(x11) #32,23,23
        x13 = self.conv13(x12) #64,21,21
        x14 = self.conv14(x13) #64,19,19
        map1 = self.pool1(x14)
        h1 = map1.reshape([map1.shape[0], -1])
        x1 = self.projector1(h1)
        mu1 = self.mu1(x1)
        log_sigma1 = self.log_sigma1(x1)
        sigma1 = torch.exp(log_sigma1)

        x21 = self.conv21(x2)
        x22 = self.conv22(x21)
        x23 = self.conv23(x22)
        x24 = self.conv24(x23)
        map2 = self.pool2(x24)
        h2 = map2.reshape([map2.shape[0], -1])
        x2 = self.projector2(h2)
        mu2 = self.mu1(x2)
        log_sigma2 = self.log_sigma2(x2)
        sigma2 = torch.exp(log_sigma2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        std_z1 = torch.from_numpy(np.random.normal(0, 1, size=sigma1.size())).float()
        std_z1.to(device)
        mu1.to(device)
        sigma1.to(device)
        code1 = mu1 + sigma1 * torch.tensor(std_z1, requires_grad=False).cuda()
        code1.to(device)

        std_z2 = torch.from_numpy(np.random.normal(0, 1, size=sigma2.size())).float()
        std_z2.to(device)
        mu2.to(device)
        sigma2.to(device)
        code2 = mu2 + sigma2 * torch.tensor(std_z2, requires_grad=False).cuda()
        code2.to(device)

        code1 = self.relu2(self.fc12(self.relu1(self.fc11(code1))))
        code2 = self.relu2(self.fc22(self.relu1(self.fc21(code2))))

        code1 = code1.view(-1, 64, 19, 19)
        code2 = code2.view(-1, 64, 19, 19)
        # print(code1.shape)
        # print(code2.shape)
        # print(x14.shape)
        # print(x24.shape)
        code = torch.cat((x14, code1, code2, x24),dim=1)

        code = self.relu3(self.bn3(self.deconv3(code)))
        code = torch.cat((x13, code, x23), dim=1)

        code = self.relu4(self.bn4(self.deconv4(code)))
        code = torch.cat((x12, code, x22), dim=1)

        code = self.relu5(self.bn5(self.deconv5(code)))
        code = torch.cat((x11, code, x21), dim=1)

        code = self.relu6(self.bn6(self.deconv6(code)))

        return code, h1, h2, mu1, mu2, sigma1, sigma2



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)        #############
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class VAE_new(nn.Module):
    def __init__(self, channel, input_dim, output_dim, windowSize):
        # 调用Module的初始化
        super(VAE_new, self).__init__()
        self.channel = channel
        self.output_dim = output_dim
        self.windowSize = windowSize
        self.conv11 = ConvBNRelu2D(in_channels=self.channel, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
        self.conv12 = ConvBNRelu2D(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        self.conv13 = ConvBNRelu2D(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv14 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)

        '''
        # (15, 27, 27)
        self.conv11 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(5,5),stride=1,padding=0)
        # (16, 23, 23)
        self.conv12 = ConvBNRelu2D(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=0)
        # (32, 21, 21)
        self.conv13 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
        # (64, 19, 19)

        '''

        self.pool1 = nn.AdaptiveAvgPool2d((4, 4))
        self.projector1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False))
        self.mu1 = nn.Linear(512, output_dim)
        self.log_sigma1 = nn.Linear(512, output_dim)
        self.conv21 = ConvBNRelu2D(in_channels=self.channel, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
        self.conv22 = ConvBNRelu2D(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        self.conv23 = ConvBNRelu2D(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv24 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)

        '''
        # (15, 27, 27)
        self.conv21 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(5,5),stride=1,padding=0)
        # (16, 23, 23)
        self.conv22 = ConvBNRelu2D(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=0)
        # (32, 21, 21)
        self.conv23 = ConvBNRelu2D(in_channels=self.channel,out_channels=16,kernel_size=(3,3),stride=1,padding=0)
        # (64, 19, 19)

        '''
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.projector2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False))
        self.mu2 = nn.Linear(512, output_dim)
        self.log_sigma2 = nn.Linear(512, output_dim)

        self.fc11 = nn.Linear(in_features=input_dim, out_features=256)
        self.relu1 = nn.ReLU()
        self.fc12 = nn.Linear(in_features=256, out_features=64 * (self.windowSize - 8) * (self.windowSize - 8))
        self.relu2 = nn.ReLU()

        self.fc21 = nn.Linear(in_features=input_dim, out_features=256)
        self.fc22 = nn.Linear(in_features=256, out_features=64 * (self.windowSize - 8) * (self.windowSize - 8))

        # reshape to (-1,64,19,19) and then deconv.
        self.deconv3 = nn.ConvTranspose2d(in_channels=64 * 4, out_channels=32 * (self.channel - 12), kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=0)
        # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
        self.bn3 = nn.BatchNorm2d(num_features=32 * (self.channel - 12))
        self.relu3 = nn.ReLU()
        # (-1, 96, 21, 21)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32 * (self.channel - 12) + 64 * 2, out_channels=64,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.relu4 = nn.ReLU()
        # (-1 64, 23, 23)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64 + 32 * 2, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                          padding=0)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.relu5 = nn.ReLU()
        # [-1, 32, 25, 25]
        self.deconv6 = nn.ConvTranspose2d(in_channels=32 + 16 * 2, out_channels=30, kernel_size=(3, 3), stride=(1, 1),
                                          padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=30)
        self.relu6 = nn.ReLU()
        # [-1, 15, 27, 27]
        self._initialize_weights()

    def forward(self, x1, x2):
        x11 = self.conv11(x1)  # 16,25,25
        x12 = self.conv12(x11)  # 32,23,23
        x13 = self.conv13(x12)  # 64,21,21
        x14 = self.conv14(x13)  # 64,19,19
        map1 = self.pool1(x14)
        h1 = map1.reshape([map1.shape[0], -1])
        x1 = self.projector1(h1)
        mu1 = self.mu1(x1)
        log_sigma1 = self.log_sigma1(x1)
        sigma1 = torch.exp(log_sigma1)

        x21 = self.conv21(x2)
        x22 = self.conv22(x21)
        x23 = self.conv23(x22)
        x24 = self.conv24(x23)
        map2 = self.pool2(x24)
        h2 = map2.reshape([map2.shape[0], -1])
        x2 = self.projector2(h2)
        mu2 = self.mu1(x2)
        log_sigma2 = self.log_sigma2(x2)
        sigma2 = torch.exp(log_sigma2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        std_z1 = torch.from_numpy(np.random.normal(0, 1, size=sigma1.size())).float()
        std_z1.to(device)
        mu1.to(device)
        sigma1.to(device)
        code1 = mu1 + sigma1 * torch.tensor(std_z1, requires_grad=False).cuda()
        code1.to(device)

        std_z2 = torch.from_numpy(np.random.normal(0, 1, size=sigma2.size())).float()
        std_z2.to(device)
        mu2.to(device)
        sigma2.to(device)
        code2 = mu2 + sigma2 * torch.tensor(std_z2, requires_grad=False).cuda()
        code2.to(device)

        code1 = self.relu2(self.fc12(self.relu1(self.fc11(code1))))
        code2 = self.relu2(self.fc22(self.relu1(self.fc21(code2))))

        code1 = code1.view(-1, 64, 19, 19)
        code2 = code2.view(-1, 64, 19, 19)
        # print(code1.shape)
        # print(code2.shape)
        # print(x14.shape)
        # print(x24.shape)
        code = torch.cat((x14, code1, code2, x24), dim=1)

        code = self.relu3(self.bn3(self.deconv3(code)))  # ()
        code = torch.cat((x13, code, x23), dim=1)

        code = self.relu4(self.bn4(self.deconv4(code)))
        code = torch.cat((x12, code, x22), dim=1)

        code = self.relu5(self.bn5(self.deconv5(code)))
        code = torch.cat((x11, code, x21), dim=1)

        code = self.relu6(self.bn6(self.deconv6(code)))

        return code, h1, h2, mu1, mu2, sigma1, sigma2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  #############
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Enc_AAE(nn.Module):
    def __init__(self,channel,output_dim,windowSize):
        # 调用Module的初始化
        super(Enc_AAE, self).__init__()        #############
        self.channel=channel
        self.output_dim=output_dim
        self.windowSize=windowSize
        self.conv11 = ConvBNRelu2D(in_channels=self.channel, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
        self.conv12 = ConvBNRelu2D(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        self.conv13 = ConvBNRelu2D(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv14 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.pool1=nn.AdaptiveAvgPool2d((4, 4))
        self.projector1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False))
        self.mu1=nn.Linear(512,output_dim)
        self.conv21 = ConvBNRelu2D(in_channels=self.channel, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
        self.conv22 = ConvBNRelu2D(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        self.conv23 = ConvBNRelu2D(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv24 = ConvBNRelu2D(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.pool2=nn.AdaptiveAvgPool2d((4, 4))
        self.projector2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False))
        self.mu2=nn.Linear(512,output_dim)
    def forward(self, x1, x2):
        x11 = self.conv11(x1)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        map1 = self.pool1(x14)
        h1 = map1.reshape([map1.shape[0], -1])
        x1 = self.projector1(h1)
        mu1=self.mu1(x1)






        x21 = self.conv21(x2)
        x22 = self.conv22(x21)
        x23 = self.conv23(x22)
        x24 = self.conv24(x23)
        map2 = self.pool2(x24)
        h2 = map2.reshape([map2.shape[0], -1])
        x2 = self.projector2(h2)
        mu2 = self.mu2(x2)

        # h = h2 - h1
        h = torch.cat((h1, h2), dim=1)

        return h, mu1, mu2, x11, x12, x13, x14, x21, x22, x23, x24
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Dec_AAE(nn.Module):#input (-1,128)
    def __init__(self, channel=30,windowSize=25,input_dim=64):
        super(Dec_AAE, self).__init__()
        self.channel = channel
        self.windowSize=windowSize
        self.fc11 = nn.Linear(in_features=input_dim, out_features=256)
        self.relu1 = nn.ReLU()
        self.fc12 = nn.Linear(in_features=256, out_features=64 * (self.windowSize - 8) * (self.windowSize - 8))
        self.relu2 = nn.ReLU()

        self.fc21 = nn.Linear(in_features=input_dim, out_features=256)
        self.fc22 = nn.Linear(in_features=256, out_features=64 * (self.windowSize - 8) * (self.windowSize - 8))

        # reshape to (-1,64,19,19) and then deconv.
        self.deconv3 = nn.ConvTranspose2d(in_channels=64 * 4, out_channels=32 * (self.channel - 12), kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=0)
        # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
        self.bn3 = nn.BatchNorm2d(num_features=32 * (self.channel - 12))
        self.relu3 = nn.ReLU()
        # (-1, 96, 21, 21)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32 * (self.channel - 12) + 64 * 2, out_channels=64,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.relu4 = nn.ReLU()
        # (-1 64, 23, 23)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64 + 32 * 2, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                          padding=0)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.relu5 = nn.ReLU()
        # [-1, 32, 25, 25]
        self.deconv6 = nn.ConvTranspose2d(in_channels=32 + 16 * 2, out_channels=30, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=30)
        self.relu6 = nn.ReLU()
        # [-1, 15, 27, 27]
        self._initialize_weights()
        #[-1,1,30,25,25]
        self._initialize_weights()
    def forward(self, x11, x12, x13, x14, code1, x21, x22, x23, x24, code2):

        code1 = self.relu2(self.fc12(self.relu1(self.fc11(code1))))
        code2 = self.relu2(self.fc22(self.relu1(self.fc21(code2))))

        #############################
        code1 = code1.view(-1, 64, 19, 19)
        code2 = code2.view(-1, 64, 19, 19)
        # print(code1.shape)
        # print(code2.shape)
        # print(x14.shape)
        # print(x24.shape)

        code = torch.cat((x14, code1, code2, x24), dim=1)

        code = self.relu3(self.bn3(self.deconv3(code)))
        code = torch.cat((x13, code, x23), dim=1)

        code = self.relu4(self.bn4(self.deconv4(code)))
        code = torch.cat((x12, code, x22), dim=1)

        code = self.relu5(self.bn5(self.deconv5(code)))
        code = torch.cat((x11, code, x21), dim=1)

        code = self.relu6(self.bn6(self.deconv6(code)))

        return code
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
#without class discriminant

class Contrast(nn.Module):
    def __init__(self,projection_dim=128,input_dim=128):
        # 调用Module的初始化
        super(Contrast, self).__init__()
        self.projection_dim=projection_dim
        self.input_dim=input_dim
        # 64*4*4
        self.module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # H_{out}=(H_{in}-1)stride-2padding+kernel_size+output_padding
            # 64*6*6
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                               stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # 64*9*9
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # 128*7*7
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # 64*5*5
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        )
        # 32*2*2
        self.projector=nn.Sequential(
            nn.Linear(in_features=128, out_features=self.projection_dim,bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.projection_dim, out_features=self.projection_dim,bias=True),
            # nn.ReLU(),
            # nn.Linear(in_features=self.projection_dim, out_features=self.projection_dim, bias=True)
        )
    def forward(self, x):
        # print(x.shape)

        x1 = x[:,0:1024]
        x2 = x[:,1024:2048]
        x1 = x1.reshape(-1, 64, 4, 4)
        x2 = x2.reshape(-1, 64, 4, 4)
        # # print(x1.shape)
        # # print(x2.shape)
        x = torch.cat((x1, x2), dim=1)
        # x=x.reshape(-1,64,4,4)
        x=self.module(x)
        h=x.reshape(-1,128)
        z=self.projector(h)
        return h, z

class Discriminant(nn.Module):
    def __init__(self,encoded_dim):
        super(Discriminant, self).__init__()
        self.lin1 = nn.Linear(encoded_dim, 256)
        self.relu=nn.ReLU(inplace=False)
        self.lin2 = nn.Linear(256, 64)
        self.relu2=nn.ReLU(inplace=False)
        self.lin3 = nn.Linear(64,1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = self.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        # x = self.relu2(self.lin3(x))
        return torch.tanh(x)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self,projection_dim=128,input_dim=128, r=16384, m=0.999, T=0.1):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = Contrast(projection_dim=projection_dim, input_dim=input_dim).cuda()
        self.encoder_k = Contrast(projection_dim=projection_dim, input_dim=input_dim).cuda()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(projection_dim, r))
        self.queue = nn.functional.normalize(self.queue.cuda(), dim=0).cuda()

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keh2ys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    @torch.no_grad()
    def get_shuffle_idx(self, bs, device):
        """shuffle index for ShuffleBN """
        shuffle_value = torch.randperm(bs).long().to(device)  # index 2 value
        reverse_idx = torch.zeros(bs).long().to(device)
        arange_index = torch.arange(bs).long().to(device)
        reverse_idx.index_copy_(0, shuffle_value, arange_index)  # value back to index
        return shuffle_value, reverse_idx
    def forward(self, im_q, im_k=None, is_eval=False, cluster_result=None, index=None, is_test=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        if is_test:
            h , k = self.encoder_q(im_q)
            k = nn.functional.normalize(k, dim=1)
            h = nn.functional.normalize(h, dim=1)
            return h, k
        if is_eval:
            h , k = self.encoder_k(im_q)
            k = nn.functional.normalize(k, dim=1)
            h = nn.functional.normalize(h, dim=1)
            return h, k

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            # shuffle for making use of BN
            shuffle_idx, reverse_idx = self.get_shuffle_idx(bs=im_q.shape[0],device= 'cuda:0')
            im_k = im_k[shuffle_idx]
            _, k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            # undo shuffle
            k = k[reverse_idx].detach()
            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute query features
        _, q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        # prototypical contrast
        if cluster_result is not None:
            proto_labels = []
            proto_logits = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())]
                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                neg_proto_id = sample(neg_proto_id, self.r)  # sample r negative prototypes
                neg_prototypes = prototypes[neg_proto_id]

                proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)

                # compute prototypical logits
                logits_proto = torch.mm(q, proto_selected.t())

                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0) - 1, steps=q.size(0)).long().cuda()

                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id, torch.LongTensor(neg_proto_id).cuda()], dim=0)]
                logits_proto /= temp_proto

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return logits, labels, proto_logits, proto_labels
        else:
            return logits, labels, None, None


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
class LogisticRegression(nn.Module):

    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.fc1 = nn.Linear(n_features,n_features)
        # self.relu=nn.ReLU()
        self.fc2 = nn.Linear(n_features, n_classes)
    def forward(self, x):
        x=self.fc1(x)
        # x=self.relu(x)
        x=self.fc2(x)
        return x

net=Contrast(projection_dim=128,input_dim=128)
net.cuda()
summary(net,tuple([2048]))
# summary(net,(1,15,27,27))
