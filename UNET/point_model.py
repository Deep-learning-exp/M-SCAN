import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from configure import *
num_condition=5
num_level=5


#post process
def heatmap_to_point(heatmap):
    B,_CL_plus_1_,D,H,W = heatmap.shape
    point = np.zeros((B,num_condition*num_level,3))
    return point


class PointNet(nn.Module):
    def __init__(self, pretrained=False):
        super(PointNet, self).__init__()
        self.output_type = ['infer', 'loss', 'more']

        self.model = None   #this is a 3d unet
        self.logit = None  # multi-class head

    def forward(self, batch):
        sagittal = batch['sagittal']
        B,C,D,H,W = sagittal.shape

        #dummy output
        logit = torch.zeros(B,num_condition*num_level+1,D,H,W)

        output = {}
        if 'loss' in self.output_type:
            #<todo>
            #output['ce_loss']
            pass

        if 'infer' in self.output_type:
            output['heatmap'] = torch.softmax(logit,1)

            if 'more' in self.output_type:
                output['point'] = heatmap_to_point(output['heatmap'] )

        return output


#------------------------------------------------------------------------
def run_check_net():
    image_size = 320
    num_slice  = 8
    batch_size = 2

    batch = {
        'sagittal': torch.from_numpy(np.random.choice(256, (batch_size, 1, num_slice,image_size, image_size))).byte(),
        'truth': torch.from_numpy(np.random.choice(num_condition*num_level+1, (batch_size, num_slice, image_size, image_size))).byte(),
    }

    net = PointNet(pretrained=False).cuda()
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
