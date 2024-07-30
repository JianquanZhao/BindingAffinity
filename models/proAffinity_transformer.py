import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        # self.positional_encoding = PositionalEncoding(hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dropout=0.1),
            num_layers
        )
        self.lowerd = nn.Linear(hidden_size,16)
        self.fc = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input, mask):
        embedded = self.embedding(input)
        embedded = embedded.transpose(0,1)
        encoder_output = self.encoder(embedded,src_key_padding_mask=~mask)
        encoder_output = self.lowerd(encoder_output)
        encoder_output = encoder_output.permute(1, 2, 0)
        seq_embed = F.adaptive_avg_pool1d(encoder_output,32)
        encoder_output = encoder_output.transpose(2, 1)
        seq_embed = seq_embed.flatten(start_dim=1)
        logits = self.fc(seq_embed)
        return logits