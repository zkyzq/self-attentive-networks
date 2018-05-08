import torch
import torch.nn as nn
import math


class ImageFeatures(nn.Module):

    def __init__(self, imgSize=28, n_out=64, drop_rate=0.1):
        super(ImageFeatures, self).__init__()
        self.imgSize = imgSize
        self.cnn = nn.Sequential(
            nn.Conv2d(1, n_out, 3, 1, 1), # 64@28*28
            nn.PReLU(n_out),
            nn.BatchNorm2d(n_out), # 64@14*14
            nn.MaxPool2d(2),

            nn.Conv2d(n_out, n_out, 3, 1, 1), #  64@14*14
            nn.PReLU(n_out),
            nn.BatchNorm2d(n_out), # 64@7
            nn.MaxPool2d(2),

            nn.Conv2d(n_out, n_out, 3, 1, 1), # 64@7
            nn.PReLU(n_out),
            nn.BatchNorm2d(n_out),
            nn.MaxPool2d(2),  # 64@3

            nn.Conv2d(n_out, n_out, 3, 1, 1), # 64@3
            nn.PReLU(n_out),
            nn.BatchNorm2d(n_out),
            nn.MaxPool2d(2),   # 64@1
        )
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.cnn(x)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        return x


class ScaledAttention(nn.Module):

    def __init__(self):
        super(ScaledAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V):
        # print(Q.size(), K.transpose(1, 2).size())
        mat = torch.bmm(Q, K.transpose(1, 2))
        scale = math.sqrt(Q.size()[-1])
        mat = mat / scale
        # print(mat.size(), self.softmax(mat).size(), V.size())
        return torch.bmm(self.softmax(mat), V)


class PositionLinear(nn.Module):

    def __init__(self, n_in, n_out, n_hidden = None):
        super(PositionLinear, self).__init__()
        if n_hidden == None:
            self.linear = nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.ReLU()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_out),
                nn.ReLU()
            )

    def forward(self, x):
        '''
        x is a 3-dimentions matrix: batch * position * embedding.
        return a position-wise linear product.
        return a 3-dimention matrix: batch * position * embedding'.
        '''
        output = []
        # each batch
        for i in range(x.size()[0]):
            tmp = []
            # each row, i.e, each feature embedding
            for j in range(x.size()[1]):
                tmp.append(self.linear(x[i, j, :]))
            tmp = torch.cat([torch.unsqueeze(item, 0) for item in tmp], 0)
            output.append(tmp)
        output = torch.cat([torch.unsqueeze(item, 0) for item in output], 0)
        return output


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, d_k, h):
        assert d_k * h == d_model
        super(MultiHeadSelfAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.linearQ = nn.ModuleList([PositionLinear(d_model, d_k) for _ in range(h)])
        self.linearK = nn.ModuleList([PositionLinear(d_model, d_k) for _ in range(h)])
        self.linearV = nn.ModuleList([PositionLinear(d_model, d_k) for _ in range(h)])
        self.scaled_attention = nn.ModuleList([ScaledAttention() for _ in range(h)])
        self.linear = PositionLinear(d_model, d_model)

    def forward(self, x):
        # rewrite it using PositionLinear class
        output = []
        for branch in range(self.h):
            b = branch
            Q = self.linearQ[b].forward(x)
            K = self.linearK[b].forward(x)
            V = self.linearV[b].forward(x)
            output.append(self.scaled_attention[b].forward(Q, K, V))
        out = torch.cat(output, 2)
        out = self.linear.forward(out)
        return out


class SubEncoder(nn.Module):

    def __init__(self, num_img, d_model, d_k, h, drop_rate=0.1):
        super(SubEncoder, self).__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention(d_model, d_k, h)
        self.normal1 = nn.BatchNorm1d(num_img)
        self.normal2 = nn.BatchNorm1d(num_img)
        self.linear = PositionLinear(d_model, d_model, d_model*4)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # print(x.size())
        x = x + self.multi_head_self_attention.forward(x)
        x = self.normal1(x)
        x = x + self.dropout(self.linear.forward(x))
        x = self.normal2(x)
        return x


class Encoder(nn.Module):

    def __init__(self, num_img, img_size, N, d_model, d_k, h, drop_rate=0.1):
        super(Encoder, self).__init__()
        self.N = N
        self.d_model = d_model
        self.d_k = d_k
        self.h = h
        self.num_img = num_img
        sub_encoders = [SubEncoder(num_img, d_model, d_k, h, drop_rate) for _ in range(self.N)]
        self.sub_encoders = nn.Sequential(*sub_encoders)
        self.image_features = ImageFeatures(img_size, 64, drop_rate=drop_rate)

    def forward(self, x):
        # x is a list of image
        assert self.num_img == len(x)
        x = [self.image_features.forward(img) for img in x]
        # print(x[0].size())
        x = torch.cat([torch.unsqueeze(feature, 1) for feature in x], 1)
        x = self.sub_encoders(x)
        return x


class DistanceNetwork(nn.Module):

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_feature, input_feature):
        eps = 1e-10
        input_feature = torch.unsqueeze(input_feature, 1).transpose(1, 2)
        dot_product = torch.squeeze(torch.bmm(support_feature, input_feature))
        sum_support = torch.sum(torch.pow(support_feature, 2), 2)
        support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
        cosine_similarity = dot_product * support_manitude
        return cosine_similarity


class transformer(nn.Module):

    def __init__(self, num_img, img_size, N, d_model, d_k, h, drop_rate=0.1):
        super(transformer, self).__init__()
        self.encoder = Encoder(num_img + 1, img_size, N, d_model, d_k, h, drop_rate)
        self.similarity = DistanceNetwork()

    def forward(self, support_set):
        '''
        support_set : a image list that represent support set.
        target: target image to classify.
        '''
        out = self.encoder.forward(support_set)
        support_feature = out[:, :-1, :]
        target_feature = out[:, -1, :]
        sim = self.similarity(support_feature, target_feature)
        return sim


# test
if __name__=='__main__':
     feature_net = ImageFeatures(28)
     print('feature net\n', feature_net)
#
    #  scaled_attention = ScaledAttention()
    #  print('scaled attention net\n', scaled_attention)
#
    #  multi_head_self_attention = MultiHeadSelfAttention(128, 32, 4)
    #  print('multi-head self-attention net\n', multi_head_self_attention)
#
    #  sub_encoder = SubEncoder(21, 128, 32, 4)
    #  print('sub-encoder net\n', sub_encoder)
#
    #  position_linear = PositionLinear(128, 32)
    #  print('position wise full connected net', position_linear)
#
    #  encoder = Encoder(20, 28, 4, 128, 32, 4)
    #  print('encoder net', encoder)
    # trans = transformer(20, 32, 4, 128, 32, 4)
    # print(trans, len(list(trans.parameters())))
