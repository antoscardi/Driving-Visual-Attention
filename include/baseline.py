from utility import *

class ConvolutionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Conv2d(in_channels=16, stride=1, out_channels=16, kernel_size=3, padding = 1)
        self.batch_norm_block = nn.BatchNorm2d(16)
        self.relu_block = nn.LeakyReLU()

    def forward(self, x):
        start = x
        x = self.conv_block(x)
        x = self.batch_norm_block(x)
        x = self.relu_block(x)
        x = start + x
        return x

class EyeFeatureExtractor(nn.Module):
    def __init__(self):
        super(EyeFeatureExtractor, self).__init__()
        # Increase channels for skip connections
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.block = ConvolutionBlock()
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x1):
        # 'Upsampling'
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        # Convolution block 
        x1 = self.block(x1)
        x1 = self.pool(x1)
        x1 = self.dropout(x1)
        # Downsampling
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        return x1

class MLPHead(nn.Module):
    def __init__(self, input_size = 672 + 16, additional_features_size=7, hidden_size=64):
        super(MLPHead, self).__init__()
        # Process additional features
        self.fc_additional = nn.Sequential(
            nn.Linear(additional_features_size, 16),
            nn.LeakyReLU()
        )
        # Process merged features
        self.fc_merged = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 2),
            nn.LeakyReLU()
        )

    def forward(self, eye_features, additional_features):
        # Process additional features
        additional_features = self.fc_additional(additional_features)
        # Concatenate eye features with additional features
        merged_features = torch.cat([eye_features, additional_features], dim=1)
        # Merge both features
        gaze = self.fc_merged(merged_features)
        return gaze
    
class GazeCNN(nn.Module):
    def __init__(self, additional_features_size=7, hidden_size=64):
        super(GazeCNN, self).__init__()
        self.eye_feature_extractor = EyeFeatureExtractor()
        self.flatten = nn.Flatten()
        self. mlp_head = MLPHead(input_size=672 + 16,additional_features_size=additional_features_size,hidden_size=hidden_size)
    
    def forward(self, left_eye, x_additional):
        eye_features = self.eye_feature_extractor(left_eye)
        eye_features = self.flatten(eye_features)
        gaze = self.mlp_head(eye_features,x_additional)
        return gaze
