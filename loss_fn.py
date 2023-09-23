import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange
import torch
from torch import tensor


class WeightedBCELoss(nn.Module):
    def __init__(self, temperature):
        super(WeightedBCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_scores, neg_scores):
        # (b, n, 1) -> (b, n)
        pos_scores = rearrange(pos_scores, 'b n l -> b (n l)')
        # log(sigmoid(x)) (b, n)
        pos_part = F.logsigmoid(pos_scores)
        # (b, n, num_negs)
        weight = F.softmax(neg_scores / self.temperature, dim=-1)
        # negative scores: (b, n, num_negs) -> (b, n)
        neg_part = reduce(F.softplus(neg_scores) * weight, 'b n num_negs -> b n', 'mean')
        loss = -pos_part + neg_part

        return loss
    

class distance_loss(nn.Module):
    def __init__(self,dis_weight):
        super(distance_loss, self).__init__()
        self.dis_weight=dis_weight
        
    def forward(self, predict_loc, trg_loc,id2gps,prob):
        [b,n,l]=predict_loc.shape
        predict_label_test = rearrange(predict_loc, 'b n l -> ( b n )l')
        predict_label =predict_label_test.expand(-1,2*l)
        trg_label = rearrange(trg_loc, 'b n l -> ( b n )l').expand(-1,2*l)
        prob = rearrange(prob, 'b n l -> ( b n )l')
        # trg_label =trg_label.expand(-1,2*l)
        value=tensor(list(id2gps.values())).to('cuda:0')
        # value=tensor(list(id2gps.values())).to('cpu')
        predict_gps_test = value.gather(0,predict_label)
        trg_gps_test = value.gather(0,trg_label)
        d_gps=trg_gps_test-predict_gps_test
        
        a = torch.sin(d_gps[:,0]/2).pow(2)
        +torch.cos(predict_gps_test[:,0])*torch.cos(trg_gps_test[:,0])*(torch.sin(d_gps[:,1]/2).pow(2))
        c=2*torch.asin(a.sqrt())
        r=6371
        dis=torch.floor(c*r)
        
        dis_loss= torch.mean(prob*dis)
        
        if(dis_loss < 1):
            return self.dis_weight*dis_loss
        
        
        return dis_loss