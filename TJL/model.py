import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_BN_Relu(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride = 1, 
                 padding = 0, 
                 dilation = 1, 
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

class Conv_Block(nn.Module):
    def __init__(self, in_dim,out_dim,
                 use_bn=True
                 ) -> None:
        super().__init__()
        dim_list = [in_dim,in_dim,out_dim]
        conv_list = []
        for i in range(len(dim_list)-1):
            dim1,dim2 = dim_list[i:i+2]
            conv_list.append(nn.Conv2d(dim1,dim2,3,1,1))
            if use_bn:
                conv_list.append(nn.BatchNorm2d(dim2))
            conv_list.append(nn.ReLU())
        self.conv = nn.ModuleList(conv_list)
    def forward(self,x):
        for c in self.conv:
            x = c(x)
        return x

class Res_Conv_Block(nn.Module):
    def __init__(self, in_dim,out_dim,
                 use_bn=True
                 ) -> None:
        super().__init__()
        dim_list = [in_dim,in_dim,out_dim]
        conv_list = []
        for i in range(len(dim_list)-1):
            dim1,dim2 = dim_list[i:i+2]
            conv_list.append(nn.Conv2d(dim1,dim2,3,1,1))
            if use_bn:
                conv_list.append(nn.BatchNorm2d(dim2))
            conv_list.append(nn.ReLU())
        self.conv = nn.ModuleList(conv_list)
        self.res_conv = nn.Conv2d(in_dim,out_dim,1)
    def forward(self,x):
        x_res = self.res_conv(x)
        for c in self.conv:
            x = c(x)
        return x+x_res

# AG
class AttBlock(nn.Module):
    def __init__(self, x_dim,g_dim) -> None:
        super().__init__()
        self.wx = nn.Sequential(
            nn.Conv2d(x_dim,1,1),
            nn.BatchNorm2d(1)
            )
        self.wg = nn.Sequential(
            nn.Conv2d(g_dim,1,1),
            nn.BatchNorm2d(1)
            )
        self.phi = nn.Conv2d(1,1,1)
    def forward(self,x,g):
        att = F.relu(self.wx(x)+self.wg(g))
        att = F.sigmoid(self.phi(att))
        return x*att







class Encoder(nn.Module):
    def down(self, x):
        return F.max_pool2d(x, kernel_size=2)

    def __init__(self,
                 in_ch=3,
                 use_res_block=True # 是否使用残差卷积块
                 ) -> None:
        super().__init__()
        conv_block=Res_Conv_Block if use_res_block else Conv_Block
        self.en_1 = conv_block(in_ch,64)
        self.en_2 = conv_block(64,128)
        self.en_3 = conv_block(128,256)
        self.en_4 = conv_block(256,512)

    def forward(self,x):
        en1 = self.en_1(x)
        x = self.down(en1)
        en2 = self.en_2(x)
        x = self.down(en2)
        en3 = self.en_3(x)
        x = self.down(en3)
        en4 = self.en_4(x)
        x = self.down(en4)
        return [en1,en2,en3,en4],x


class Decoder(nn.Module):
    def up(self,x):
        _,_,h,w = x.shape
        return F.interpolate(x, size=(int(h*2),int(w*2)), mode="bilinear")
    def __init__(self,
                 in_ch = 1024,
                 use_res_block=True # 是否使用残差卷积块
                 ) -> None:
        super().__init__()
        conv_block=Res_Conv_Block if use_res_block else Conv_Block
        self.de_4_upconv = nn.ConvTranspose2d(in_ch,512,3,2,padding=1,output_padding=1)
        self.de_4 = conv_block(512*2,512)
        self.de_3_upconv = nn.ConvTranspose2d(512,256,3,2,padding=1,output_padding=1)
        self.de_3 = conv_block(256*2,256)
        self.de_2_upconv = nn.ConvTranspose2d(256,128,3,2,padding=1,output_padding=1)
        self.de_2 = conv_block(128*2,128)
        self.de_1_upconv = nn.ConvTranspose2d(128,64,3,2,padding=1,output_padding=1)
        self.de_1 = conv_block(64*2,64)

    def forward(self,en_list,x):
        en1,en2,en3,en4 = en_list
        x = self.de_4_upconv(x)
        x = torch.concat([en4,x],dim=1)
        x = self.de_4(x)
        x = self.de_3_upconv(x)
        x = torch.concat([en3,x],dim=1)
        x = self.de_3(x)
        x = self.de_2_upconv(x)
        x = torch.concat([en2,x],dim=1)
        x = self.de_2(x)
        x = self.de_1_upconv(x)
        x = torch.concat([en1,x],dim=1)
        x = self.de_1(x)
        return x



class UNet(nn.Module):
    def __init__(self,
                 in_ch=1,
                 center_ch=1024,
                 ) -> None:
        super().__init__()
        # 编码器
        self.encoder = Encoder(in_ch)
        # 瓶颈
        self.center = Res_Conv_Block(512,center_ch)
        # 前景解码器
        self.decoder = Decoder(center_ch)
        self.out = nn.Sequential(
            nn.Conv2d(64,1,1),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        # 编码
        en_list,x = self.encoder(x)
        # 瓶颈
        x = self.center(x)
        # 前景
        de = self.decoder(en_list,x)
        de = self.out(de)
        return de
    

class UNet_2DE(nn.Module):
    def __init__(self,
                 in_ch=1,
                 center_ch=1024,
                 ) -> None:
        super().__init__()
        # 编码器
        self.encoder = Encoder(in_ch)
        # 瓶颈
        self.center = Res_Conv_Block(512,center_ch)
        # 边缘解码器
        self.decoder_edge = Decoder(center_ch)
        self.out_edge = nn.Sequential(
            nn.Conv2d(64,1,1),
            nn.Sigmoid()
            )    
        # 前景解码器
        self.decoder = Decoder(center_ch)
        self.out = nn.Sequential(
            nn.Conv2d(64*2,1,1),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        # 编码
        en_list,x = self.encoder(x)
        # 瓶颈
        x = self.center(x)
        # 边缘
        de_edge = self.decoder_edge(en_list,x)
        edge = self.out_edge(de_edge)
        # 前景
        de = self.decoder(en_list,x)
        de = torch.concat([de,de_edge],1)
        de = self.out(de)
        return de,edge