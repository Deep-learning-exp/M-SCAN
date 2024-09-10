import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.resnet import *


#------------------------------------------------
class MyDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        skip_channel,
        out_channel,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyUnetDecoder(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            # print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
            # print(block.conv1[0])
            # print('')
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor(0))
        self.register_buffer('std', torch.tensor(1))

        arch = 'resnet50d'

        encoder_dim ={
            'resnet18': [64, 64, 128, 256, 512,],
            'resnet50d': [64, 256, 512, 1024, 2048,],
        }[arch]
        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = timm.create_model(
            model_name=arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
        )

        self.decoder = MyUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] +[0],
            out_channel=decoder_dim,
        )
        self.logit  = nn.Conv2d(decoder_dim[-1], 6, kernel_size=1)  #num_class = num_level(keypoints) + 1 (none)


    def forward(self, batch):
        device= self.D.device
        image = batch['sagittal'].to(device)
        batch_size = len(image)
        B, _1_, H, W = image.shape

        x = image.float() / 255
        x = (x - self.mean) / self.std
        x = x.expand(-1, 3, -1, -1)

        #---------------------------------------
        encode=[]
        e = self.encoder
        x = e.act1(e.bn1(e.conv1(x))); encode.append(x)
        # x = e.maxpool(x)
        x = F.avg_pool2d(x,kernel_size=2,stride=2);
        x = e.layer1(x); encode.append(x)
        x = e.layer2(x); encode.append(x)
        x = e.layer3(x); encode.append(x)
        x = e.layer4(x); encode.append(x)
        ##[print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]

        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]+[None]
        )
        ##[print(f'decode_{i}', e.shape) for i,e in enumerate(decode)]
        ##print('last', last.shape)

        logit = self.logit(last)

        output = {}
        if 'loss' in self.output_type:
            truth = batch['mask'].long().to(device)
            output['mask_loss'] = F.cross_entropy(logit,truth)

        if 'infer' in self.output_type:
            p = torch.softmax(logit,1)
            output['probability'] = p

        return output


#------------------------------------------------------------------------
def run_check_net():
    image_size = 512
    batch_size = 4

    batch = {
        'sagittal': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 1, image_size, image_size))).byte(),
        'mask': torch.from_numpy(np.random.choice(3, (batch_size, image_size, image_size))).byte(),
    }

    net = Net(pretrained=True).cuda()
    #print(net)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)
    # ---
    print('batch')
    for k, v in batch.items():
        print(f'{k:>32} : {v.shape} ')

    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print(f'{k:>32} : {v.shape} ')
    print('loss')
    for k, v in output.items():
        if 'loss' in k:
            print(f'{k:>32} : {v.item()} ')


# main #################################################################
if __name__ == '__main__':
    run_check_net()


