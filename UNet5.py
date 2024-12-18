
import torch
import torch.nn as nn

class UNet5(nn.Module):
    def __init__(
            self,
            conv_kernel_size=3,
            conv_stride=1, ):
        super().__init__()

        #### encoding - the contractng path in the U-Net 
        # nb. of channels : 3 -> 64
        self.down_conv_0 = self.down_convolution_block(level=0, kernel_size=conv_kernel_size, stride=conv_stride)
        self.pool_0 = nn.MaxPool2d(2,2) # kernel=2, stride=2 => image size is divided by 2

        # nb. of channels : 64 -> 128
        self.down_conv_1 = self.down_convolution_block(level=1, kernel_size=conv_kernel_size, stride=conv_stride)
        self.pool_1 = nn.MaxPool2d(2,2)

        # nb. of channels : 128 -> 256
        self.down_conv_2 = self.down_convolution_block(level=2, kernel_size=conv_kernel_size, stride=conv_stride)
        self.pool_2 = nn.MaxPool2d(2,2)

        # nb. of channels : 256 -> 512
        self.down_conv_3 = self.down_convolution_block(level=3, kernel_size=conv_kernel_size, stride=conv_stride)
        self.pool_3 = nn.MaxPool2d(2,2)

        # nb. of channels : 512 -> 1024
        self.down_conv_4 = self.down_convolution_block(level=4, kernel_size=conv_kernel_size, stride=conv_stride)
        self.pool_4 = nn.MaxPool2d(2,2)

        #### Bottleneck - the bottom of the U 
        # nb. of channels : 1024 -> 2048
        self.bottleneck = self.down_convolution_block(level=5, kernel_size=conv_kernel_size, stride=conv_stride)
        self.up_sampling_5 = self.up_sampling(level=5) # scale=2 => image size is multiplied by 2

        #### Decoding - the expansive path in the U-Net
        # nb. of channels : 2048 -> 1024
        self.up_conv_4 = self.up_convolution_block(level=4, kernel_size=conv_kernel_size, stride=conv_stride)
        self.up_samlping_4 = self.up_sampling(level=4)
        
        # nb. of channels : 1024 -> 512
        self.up_conv_3 = self.up_convolution_block(level=3, kernel_size=conv_kernel_size, stride=conv_stride)
        self.up_samlping_3 = self.up_sampling(level=3)

        # nb. of channels : 512 -> 256
        self.up_conv_2 = self.up_convolution_block(level=2, kernel_size=conv_kernel_size, stride=conv_stride)
        self.up_samlping_2 = self.up_sampling(level=2)

        # nb. of channels : 256 -> 128
        self.up_conv_1 = self.up_convolution_block(level=1, kernel_size=conv_kernel_size, stride=conv_stride)
        self.up_samlping_1 = self.up_sampling(level=1)

        # nb. of channels : 128 -> 64
        self.up_conv_0 = self.up_convolution_block(level=0, kernel_size=conv_kernel_size, stride=conv_stride)

        # the last convolution 
        # nb. of channels : 64 -> 1
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)



    def convolution_block(self, number_of_in_channels, number_of_out_channels, kernel_size, stride, padding):
        """ 
        Performs two times the block :[2d_Convolution, normalization, activation function ReLU] 
        """
        return nn.Sequential(
            # first convolution
            nn.Conv2d(number_of_in_channels, number_of_out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(number_of_out_channels), 
            nn.ReLU(),
            # second convolution
            nn.Conv2d(number_of_out_channels, number_of_out_channels , kernel_size, stride, padding),
            nn.BatchNorm2d(number_of_out_channels),
            nn.ReLU(),
            )


    def down_convolution_block(self, level, kernel_size, stride):
        """
        level is (-minus) the 'height' in the U-Net. Ground level is 0, then each down sampling is adding +1 to level, each up sampling is adding -1 level. i.e. after three down sampling, the level is 3."""
        
        kernel_size = int(kernel_size - (kernel_size + 1 )%2) # to make sure kernel size is odd
        padding = int((kernel_size - 1)*0.5)
        
        if int(level) == 0: 
            number_of_in_channels = 3 # if first level, goes from 3 to 64 ( 64 = 2**6)
        else:
            number_of_in_channels = int(2**(5+level)) 
        
        number_of_out_channels = int(2**(6+level))
        
        return self.convolution_block(number_of_in_channels, number_of_out_channels, kernel_size, stride, padding)


    def up_convolution_block(self, level, kernel_size, stride):
        """ 
        level is (-minus) the 'height' in the U-Net. Ground level is 0, then each down sampling is adding +1 to level, each up sampling is adding -1 level. i.e. after three down sampling, the level is 3."""

        kernel_size = int(kernel_size - (kernel_size + 1 )%2) # to make sure kernel size is odd
        padding = int((kernel_size - 1)*0.5)

        number_of_in_channels  = int(2**(7+level)) # in channels > out channels, e.g. we go from 128 to 64 at level=0
        number_of_out_channels = int(0.5*number_of_in_channels)
        
        return self.convolution_block(number_of_in_channels, number_of_out_channels, kernel_size, stride, padding)
    

    def up_sampling(self, level):

        number_of_in_channels  = int(2**(6+level)) 
        number_of_out_channels = int(0.5*number_of_in_channels)

        return nn.Sequential(nn.ConvTranspose2d(in_channels=number_of_in_channels, out_channels=number_of_out_channels, kernel_size=2, stride=2))


    def forward(self, x):
        
        skip0 = self.down_conv_0(x)
        x = self.pool_0(skip0)

        skip1 = self.down_conv_1(x)
        x = self.pool_1(skip1)

        skip2 = self.down_conv_2(x)
        x = self.pool_2(skip2)

        skip3 = self.down_conv_3(x)
        x = self.pool_3(skip3)

        skip4 = self.down_conv_4(x)
        x = self.pool_4(skip4)

        x = self.bottleneck(x)
        x = self.up_sampling_5(x)

        x = torch.cat((x, skip4), dim=1)
        x = self.up_conv_4(x)
        x = self.up_samlping_4(x)

        x = torch.cat((x, skip3), dim=1)
        x = self.up_conv_3(x)
        x = self.up_samlping_3(x)

        x = torch.cat((x, skip2), dim=1)
        x = self.up_conv_2(x)
        x = self.up_samlping_2(x)

        x = torch.cat((x, skip1), dim=1)
        x = self.up_conv_1(x)
        x = self.up_samlping_1(x)

        x = torch.cat((x, skip0), dim=1)
        x = self.up_conv_0(x)

        x = self.final_conv(x)

        return x