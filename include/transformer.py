from utility import *
from baseline import *   
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model , num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        # The transformer encoder layer
        self.layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layers, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # No embedding layer needed for non-token input
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        return x


class CNNTrans(nn.Module): 
    def __init__(self, d_model=24, nhead=2, num_layers=2):
        super(CNNTrans, self).__init__()
        self.eye_feature_extractor = EyeFeatureExtractor()
        self.eye_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(672, 17),
            nn.LeakyReLU()
        )
        self.encoder = TransformerEncoder(            
            d_model= d_model, 
            num_heads = nhead,
            num_layers = num_layers
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(24, 2),
            nn.LeakyReLU()
        )
        self.flatten= nn.Flatten()
    def forward(self, eye, additional_features):
        eye = self.eye_feature_extractor(eye)
        # Feature extracted by EyeFeatureExtractor has dimensions [batch_size, channels, height, width]. 
        # However, the TransformerEncoder expects input with dimensions [sequence_length, batch_size, features].
        eye = self.eye_head(eye)
        all_features = torch.cat((eye, additional_features), dim=1)
        encoded_features = self.encoder(all_features.unsqueeze(1))
        encoded_features = self.flatten(encoded_features)
        gaze = self.mlp_head(encoded_features)
        return gaze
    
