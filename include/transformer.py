from utility import *
from baseline import *   
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model , num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        # The transformer encoder layer
        self.layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layers, num_layers=num_layers)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        # No embedding layer needed for non-token input
        x = self.transformer_encoder(x)
        return x


class CNNTrans(nn.Module): 
    def __init__(self, d_model=8, nhead=8, ntoken = 4, num_layers=4):
        super(CNNTrans, self).__init__()
        self.eye_feature_extractor = EyeFeatureExtractor()
        self.encoder = TransformerEncoder(            
            d_model= d_model, 
            num_heads = nhead,
            ntoken = ntoken,
            num_layers = num_layers
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.mpl_head = MLPHead()
    def forward(self, x_in, additional_features):
        feature = self.eye_feature_extractor(x_in)
        # Feature extracted by EyeFeatureExtractor has dimensions [batch_size, channels, height, width]. 
        # However, the TransformerEncoder expects input with dimensions [sequence_length, batch_size, features].
        feature = feature.flatten(2).permute(2, 0, 1)
        feature = self.encoder(feature)
        feature = self.flatten(feature.permute(1,0,2))
        gaze = self.mpl_head(feature, additional_features)
        return gaze
    
