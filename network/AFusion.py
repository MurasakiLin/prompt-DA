import torch
import torch.nn as nn


class AFusion(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AFusion, self).__init__()
        self.channel = channel
        self.fc_spatial = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_channel = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(channel//reduction),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_tf,x_cnn):
        B,C,W,H,D = x_tf.shape[0],x_tf.shape[1],x_tf.shape[2],x_tf.shape[3],x_tf.shape[4],
        #B L C
        x_tf = x_tf.reshape(B,C,W*H*D)      #B,512,8,8,8 ===> B,512,512
        x_cnn = x_cnn.reshape(B,C,W*H*D)

#        x_spatial_mask = self.fc_spatial(x_cnn)  # B L 1  /// B C 1
#        x_channel_mask = self.fc_channel(self.avg_pool(x_tf).permute(0,2,1))  # B 1 C
#        x_mask = self.sigmoid(x_spatial_mask.expand_as(x_cnn) + x_channel_mask.expand_as(x_tf))

        x_spatial_mask = self.fc_spatial(x_cnn.permute(0,2,1)) # B L 1
        x_channel_mask = self.fc_channel(self.avg_pool(x_tf).permute(0,2,1)) # B 1 C
        x_mask = self.sigmoid(x_spatial_mask.permute(0,2,1).expand_as(x_cnn) + x_channel_mask.permute(0,2,1).expand_as(x_tf))
        return (x_cnn * x_mask + x_tf * (1 - x_mask)).reshape(B,C,W,H,D)


'''
layer = DANE(512)

a = torch.randn(16,512,8,8,8)
b = torch.randn(16,512,8,8,8)

c = layer(a,b)
print(c.shape)
'''
