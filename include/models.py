import copy
import torch.nn as nn
from utility import *

'''
ResNet
'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 kernel convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  #indicating that the number of output channels will be 4 times the number of input channels

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  #residual connection
        out = self.relu(out)

        return out

#CNN to create the Feature Map
class ResNet(nn.Module):

    def __init__(self, block, layers, maps=32):
        self.inplanes = 64
        super(ResNet, self).__init__()
        #takes 3 chanels in input [RGB] and reutrn 64 chanels in output
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.conv = nn.Sequential(
            nn.Conv2d(512, maps, 1),
            nn.BatchNorm2d(maps),
            nn.ReLU(inplace=True)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv(x)
        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


'''
ReSViT
'''
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos


    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class ReSViT(nn.Module):
    def __init__(self):
        super(ReSViT, self).__init__()
        maps = 16 #32
        nhead = 4 #8
        dim_feature = 7*7
        dim_feedforward= 512 #512
        dropout = 0.1
        num_layers=2 #6

        self.base_model = resnet18(maps=maps)

        # d_model: dim of Q, K, V
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(
                  maps,
                  nhead,
                  dim_feedforward,
                  dropout)

        encoder_norm = nn.LayerNorm(maps)
        # num_encoder_layer: deeps of layers

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        self.feed = nn.Linear(maps, 2)

    def forward(self, x_in):
        feature = self.base_model(x_in)
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)

        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)

        #change the number 577 based on the size of the input tensor
        position = torch.from_numpy(np.arange(0, 577)).cuda()

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)

        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0]

        gaze = self.feed(feature)

        return gaze
    

class ConvolutionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Conv2d(in_channels=32, stride=1, out_channels=32, kernel_size=3, padding = 1)
        self.batch_norm_block = nn.BatchNorm2d(32)
        self.prelu_block = nn.PReLU()

    def forward(self, x):
        start = x
        x = self.conv_block(x)
        x = self.batch_norm_block(x)
        x = self.prelu_block(x)
        x = start + x
        return x

class EyeFeatureExtractor(nn.Module):
    def __init__(self):
        super(EyeFeatureExtractor, self).__init__()

        # Increase channels for skip connections
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.relu = nn.PReLU()
        self.block = ConvolutionBlock()
        self.pool = nn.MaxPool2d(kernel_size=8, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=9, stride=1, padding=1)
        # Merge Branches
        self.flatten = nn.Flatten()
    
    def forward(self, x1):
        # 'Upsampling'
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        # Branch 1
        x1 = self.block(x1)
        x1 = self.pool(x1)
        x1 = self.dropout(x1)
        # Downsampling
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        x1 = self.flatten(x1)

        return x1
    
class GazeCNN(nn.Module):
    def __init__(self, additional_features_size=7, hidden_size=256):
        super(GazeCNN, self).__init__()
        self.hidden_size = hidden_size

        # Eye feature extractor
        self.eye_feature_extractor = EyeFeatureExtractor()

        # Fully connected layers for additional features
        self.fc_additional = nn.Sequential(
            nn.Linear(additional_features_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 16),
            nn.PReLU()
        )

        # Merge both the eye features and additional features, initialized to None
        self.fc_merge = nn.Sequential(
                nn.Linear(1536 + 16, self.hidden_size),
                nn.PReLU()
            )

        # Output layer for x-coordinate
        self.fc_output = nn.Linear(hidden_size, 2)


    def forward(self, left_eye, x_additional):
        # Extract features from the eyes
        eye_features = self.eye_feature_extractor(left_eye)

        # Process additional features
        additional_features = self.fc_additional(x_additional)

        # Concatenate eye features with additional features
        merged_features = torch.cat([eye_features, additional_features], dim=1)

        # Merge both features
        merged_features = self.fc_merge(merged_features)

        # Output layers for x and y coordinates
        gaze = self.fc_output(merged_features)

        return gaze