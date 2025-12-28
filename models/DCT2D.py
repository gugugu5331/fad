"""DNN architectures based on STRF kernels."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Multi-Layer Perceptron."""

    def __init__(self, indim, outdim, hiddims=[], bias=True,
                 activate_hid=nn.ReLU(), activate_out=nn.ReLU(),
                 batchnorm=[]):
        """Initialize a MLP.

        Parameters
        ----------
        indim: int
            Input dimension to the MLP.
        outdim: int
            Output dimension to the MLP.
        hiddims: list of int
            A list of hidden dimensions. Default ([]) means no hidden layers.
        bias: bool [True]
            Apply bias for this network?
        activate_hid: callable, optional
            Activation function for hidden layers. Default to ReLU.
        activate_out: callable, optional
            Activation function for output layer. Default to ReLU.

        """
        super(MLP, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hiddims = hiddims
        self.nhidden = len(hiddims)
        if self.nhidden == 0:
            print("No hidden layers.")
        # indims = [indim] + hiddims
        # outdims = hiddims + [outdim]
        self.layers = nn.ModuleList([])
        # for ii in range(self.nhidden):
        #     self.layers.append(nn.Linear(indims[ii], outdims[ii], bias=bias))
        #     if len(batchnorm) > 0 and batchnorm[ii]:
        #         self.layers.append(nn.BatchNorm1d(outdims[ii], momentum=0.05))
        #     self.layers.append(activate_hid)
        self.layers.append(nn.Linear(self.indim, self.outdim, bias=bias))
        if activate_out is not None:
            self.layers.append(activate_out)

    def forward(self, x):
        """One forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


class SelfAttention(nn.Module):
    """A self-attentive layer."""
    def __init__(self, indim, hiddim=256):
        super(SelfAttention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(indim, hiddim),
            nn.Tanh(),
            nn.Linear(hiddim, 1, bias=False)
        )

    def forward(self, x):
        """Transform a BxTxF input tensor."""
        y_attn = self.layers(x)
        attn = F.softmax(y_attn, dim=1)
        attn_applied = torch.matmul(x.transpose(2, 1), attn).squeeze(-1)
        return attn_applied, attn


class Model(nn.Module):
    """A generic STRFNet with generic or STRF kernels in the first layer.

       Processing workflow:
       Feat. -> STRF/conv2d ->  Residual CNN -> Attention -> MLP -> Class prob.
       BxTxF ----> BxTxF -------> BxTxF ---------> BxF ----> BxK -> BxC

    """
    def __init__(self):
        """See init_STRFNet for initializing each component."""
        super(Model, self).__init__()
        self.strf_layer = None

        num_kernels = 32
        d1, d2 = (3, 3)
        self.conv2d_layer = nn.Conv2d(
            1, num_kernels,  # Double to match the total number of STRFs
            (d1, d2), padding=(d1//2, d2//2)
        )

        residual_channels = [32, 32]
        # self.residual_layer = ModResnet(
        #     3 * num_kernels, residual_channels, False
        # )


        self.residual_layer = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ) # .view(x.size(0), -1)

        self.num_classes = 2
        embedding_dimension = 1024

        mlp_hiddims = []
        activate_out = nn.LogSoftmax(dim=1)

        flattened_dimension = 640
        num_rnn_layers = 2
        self.linear_layer = nn.Linear(flattened_dimension, embedding_dimension)
        self.rnn = nn.GRU(
            embedding_dimension, embedding_dimension, batch_first=True,
            num_layers=num_rnn_layers, bidirectional=True
        )

        self.attention_layer = SelfAttention(2*self.rnn.hidden_size)
        #self.att_layers1 = nn.Linear(63, 1, bias=True)
        #self.att_layers2 = nn.Linear(256,1, bias=True)
        self.mlp = nn.Linear(2*embedding_dimension, 256, bias=True)

        self.last_layer = nn.Linear(256, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
            # MLP(
            # 2*embedding_dimension, num_classes, hiddims=mlp_hiddims,
            # activate_hid=nn.LeakyReLU(),
            # activate_out=activate_out,
            # batchnorm=[True]*len(mlp_hiddims)

    def forward(self, x, return_embedding=False):
        """Forward pass a batch-by-time-by-frequency tensor."""

        def flatten(x):
            return x.transpose_(1, 2).reshape(x.size(0), x.size(1), -1)

        #print(x.size())
        x = self.conv2d_layer(x.unsqueeze(1))
        x = self.residual_layer(x)
        x = flatten(x)
        x = self.linear_layer(x)
        x, _ = self.rnn(x)
        x, attn = self.attention_layer(x)
        embedding = self.mlp(x)

        #print(x.size())
        #x = self.att_layers1(x.transpose(2,1))
        #print(x.size())

        out = self.last_layer(embedding)
        out = self.sigmoid(out)

        return embedding, out

# Model(
#   (conv2d_layer): Conv2d(1, 180, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (residual_layer): ModResnet(
#     (res_layers): Sequential(
#       (0): Sequential(
#         (0): ResidualBlock(
#           (convlayers): Sequential(
#             (0): Conv2d(180, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): LeakyReLU(negative_slope=0.01)
#             (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#           (downsample): Sequential(
#             (0): Conv2d(180, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#         (1): ResidualBlock(
#           (convlayers): Sequential(
#             (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): LeakyReLU(negative_slope=0.01)
#             (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#       )
#       (1): Sequential(
#         (0): ResidualBlock(
#           (convlayers): Sequential(
#             (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): LeakyReLU(negative_slope=0.01)
#             (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#           (downsample): Sequential(
#             (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#         (1): ResidualBlock(
#           (convlayers): Sequential(
#             (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): LeakyReLU(negative_slope=0.01)
#             (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#       )
#       (2): Sequential(
#         (0): ResidualBlock(
#           (convlayers): Sequential(
#             (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): LeakyReLU(negative_slope=0.01)
#             (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#           (downsample): Sequential(
#             (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#         (1): ResidualBlock(
#           (convlayers): Sequential(
#             (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): LeakyReLU(negative_slope=0.01)
#             (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )
#       )
#     )
#   )
#   (linear_layer): Linear(in_features=320, out_features=1024, bias=True)
#   (rnn): GRU(1024, 1024, num_layers=2, batch_first=True, bidirectional=True)
#   (attention_layer): SelfAttention(
#     (layers): Sequential(
#       (0): Linear(in_features=2048, out_features=256, bias=True)
#       (1): Tanh()
#       (2): Linear(in_features=256, out_features=1, bias=False)
#     )
#   )
#   (att_layers1): Linear(in_features=63, out_features=1, bias=True)
#   (mlp): Linear(in_features=2048, out_features=2, bias=True)
# )
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(ResidualBlock, self).__init__()
        self.convlayers = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x
        residual = residual + self.convlayers(x)
        return torch.relu(residual)


class ModResnet(nn.Module):
    """Modified ResNet from the PyTorch tutorial."""
    def __init__(self, in_chan, res_chans, pool=True):
        super(ModResnet, self).__init__()
        """Instantiate a series of residual blocks.

        Parameters
        ----------
        in_chan: int
            Input channel number
        res_chans: list(int)
            Channel number for each residual block.

        """
        self.in_channels = in_chan
        assert len(res_chans) > 0, "Requires at least one residual block!"
        res_layers = [self.make_layer(ResidualBlock, res_chans[0], 2)]
        for cc in res_chans:
            res_layers.append(self.make_layer(ResidualBlock, cc, 2, 2))

        self.res_layers = nn.Sequential(*res_layers)
        if pool:
            self.avg_pool = nn.AvgPool2d((8, 5))
        self.pool = pool

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.res_layers(x)
        if self.pool:  # average pool and then flatten out to single vector
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)

        return out

