from utility import *
from baseline import *
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_maps, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_maps, nhead, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(d_maps, dim_feedforward,dropout)
        self.norm1 = nn.LayerNorm(d_maps)
        self.norm2 = nn.LayerNorm(d_maps)
        self.dropout = nn.Dropout(dropout)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos

    def forward(self, src, pos):
        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
            output = self.norm(output)
        return output

class CNNTrans(nn.Module):
    '''
    params:
        - nhead: seq num
        - d_maps: dim of Q, K, V
        - dim_feedforward: dim of hidden linear layers
        - num__layer: deeps of layers
        - mlp_size: size of the mlp head
    '''
    def __init__(self, device, batch_size, d_maps=16, nhead=4, dim_feature=12*12, dim_feedforward=256, dropout=0.1, num_layers=4,additional_feature_size=7,mlp_size=256):
        super(CNNTrans, self).__init__()
        ### Params
        encoder_layer = TransformerEncoderLayer(
                  d_maps,
                  nhead,
                  dim_feedforward,
                  dropout)
        encoder_norm = nn.LayerNorm(d_maps)
        self.device = device
        self.BATCH_SIZE = batch_size
        ### Modules
        self.eye_feature_extractor = EyeFeatureExtractor()
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_maps))
        self.pos_embedding = nn.Embedding(dim_feature+1, d_maps)
        self.mpl_head = MLPHead(additional_features_size= additional_feature_size,hidden_size=mlp_size)

    def forward(self, x_in, additional_features):
        feature = self.eye_feature_extractor(x_in)
        feature = feature.flatten(2).permute(2, 0, 1)
        cls = self.cls_token.repeat( (1, self.BATCH_SIZE, 1))
        feature = torch.cat([cls, feature], 0)
        position = torch.from_numpy(np.arange(0, 145)).to(self.device)
        pos_feature = self.pos_embedding(position)
        # feature is [HxW, batch, channel]
        feature = self.encoder(feature, pos_feature)
        feature = feature.permute(1, 2, 0)
        feature = feature[:,:,0]
        gaze = self.mpl_head(feature, additional_features)
        return gaze