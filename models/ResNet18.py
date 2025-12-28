import torch.nn as nn
import torch
import torchvision.models as vismodel
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.SR = 16000
        self.hidden_features = 1024
        self.type_num = 3
        #self.lfcc_feature = lfcc
        self.resnet18 = ResNet18Module(self.hidden_features)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 256),  # 512, 256
        #     nn.ReLU(True))
        #     #nn.Linear(256, 2))
        # self.last_linear = nn.Linear(256, 1)
        # self.sigmoid = nn.Sigmoid()

        self.sources_num = 5
        self.fake_task = nn.Sequential(
            nn.Linear(1024, 2),
            nn.Sigmoid()
        )
        self.source_task = nn.Sequential(
            nn.Linear(1024, self.sources_num),
            nn.Softmax(dim=1),
        )

    def forward(self, x):  # x [batchsize, len, dim]
        #spec = self.lfcc_feature(audio_input, fs=self.SR, nfilts=60, nfft=1024, win_len=0.030, win_hop=0.010)
        # torch.Size([batchsize, len, 60])
        #print(spec.size())
        spec = x.unsqueeze(1)  # Size([batchsize, len, 60])
        #print(spec.size())
        x = spec.repeat(1, 3, 1, 1)
        #print(spec.size())
        hidden = self.resnet18(x)
        #hidden = F.relu(self.fc1(x))
        task1_out = self.fake_task(hidden)
        task2_out = self.source_task(hidden)
        return hidden, task1_out, task2_out

class ThresholdModel(nn.Module):
    def __init__(self):
        super(ThresholdModel, self).__init__()
        self.base_model = Model()
        self.threshold = nn.Parameter(torch.tensor(0.0)) # 初始化阈值为0.5
        self.sig = nn.Sigmoid()

    def forward(self, x):
        scores, output2 = self.base_model(x)
        threshold_sig = self.sig(self.threshold)
        return scores, output2, threshold_sig

    def predict(self, x):
        scores, output2 = self.base_model(x)
        threshold_sig = self.sig(self.threshold)
        predictions = (scores > threshold_sig).float()
        return predictions, output2

class ResNet18Module(nn.Module):
    def __init__(self, out_features):
        super(ResNet18Module, self).__init__()
        self.resnet18_model = vismodel.resnet18(pretrained=True) # pre-trained true or false
        self.resnet18_model.fc = nn.Linear(self.resnet18_model.fc.in_features, out_features) # V1: 512, 512  # V2: 512, 256
        for param in self.resnet18_model.parameters():
            param.requires_grad = True  # v1 True

    def forward(self, spec):
        embeddings = self.resnet18_model(spec)
        return embeddings

