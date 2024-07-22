import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.identity_conv = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.identity_conv:
            identity = self.identity_conv(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, out_channels, input_channels=3):
        super(ResNet, self).__init__()
        self.block1 = ResNetBlock(input_channels, 32)
        self.block2 = ResNetBlock(32, 64)
        self.block3 = ResNetBlock(64, out_channels)
        self.region_pool = nn.MaxPool2d(kernel_size=16, stride=16)

    def forward(self, x, coords=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        pooled_x = self.region_pool(x)

        if coords is not None:
            batch_size, num_points, _ = coords.size()
            region_coords = (coords / 16).long()
            i = region_coords[..., 0]  # i coordinates for all points
            j = region_coords[..., 1]  # j coordinates for all points

            embeddings = pooled_x[torch.arange(batch_size)[:, None], :, i, j]

            return embeddings

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(query, key, value, attn_mask=mask)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, dropout, forward_expansion):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.dropout(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class xMINT(nn.Module):
    def __init__(self,
                 num_known_gene,
                 num_gene_total, 
                 image_channels=3, 
                 embed_size=256, 
                 num_transformer_layers=4, 
                 transformer_heads=8, 
                 transformer_dropout=0.1, 
                 transformer_forward_expansion=8):
        
        super().__init__()
        self.global_transformer = Transformer(
            embed_size=2 * embed_size,
            num_layers=num_transformer_layers,
            heads=transformer_heads,
            dropout=transformer_dropout,
            forward_expansion=transformer_forward_expansion
        )

        self.gene_transformer = Transformer(
            embed_size=embed_size, 
            num_layers=num_transformer_layers,
            heads=transformer_heads,
            dropout=transformer_dropout,
            forward_expansion=transformer_forward_expansion
        )

        # Convolutional layers for extracting image features
        self.conv_layers = ResNet(out_channels=embed_size, input_channels=image_channels)
        
        # Linear projection layer for mapping known_gene to an embed_size vector
        self.gene_projection = nn.Sequential(
            nn.Linear(num_known_gene, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size))

        # Output layer with adjusted dimensions of imputation genes.
        self.output_proj_layer = nn.Linear(2 * embed_size, num_gene_total-num_known_gene)

    def forward(self, image, coordinates, known_gene):

        # Extract and reshape convolutional features
        conv_features = self.conv_layers(image, coordinates)

        # Project masked_gene to the embedding space
        projected_gene = self.gene_projection(known_gene)
        projected_gene = self.gene_transformer(projected_gene, None)

        # Normalize both conv_features and projected_gene
        conv_features_norm = torch.norm(conv_features, p=2, dim=2, keepdim=True)
        projected_gene_norm = torch.norm(projected_gene, p=2, dim=2, keepdim=True)
        conv_features_normalized = conv_features / conv_features_norm
        projected_gene_normalized = projected_gene / projected_gene_norm

        # Combine adjusted projected_gene with conv_features
        combined_features = torch.cat((conv_features_normalized, projected_gene_normalized), dim=-1)

        transformer_out = self.global_transformer(combined_features, None)

        output_embeddings = self.output_proj_layer(transformer_out)

        return output_embeddings
