import torch
import torch.nn as nn
import random

from .i3d import Inception3D


class Encoder(nn.Module):
    """
    Encoder module using a CNN-LSTM architecture for processing video sequences.

    Args:
        in_channels (int): Number of input channels for the video frames.
        hidden_dim (int): Dimension of the hidden state in the LSTM.
        n_layers (int): Number of layers in the LSTM.
        dropout (float): Dropout rate applied to the LSTM and CNN outputs.
    """
    def __init__(self, in_channels, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cnn = Inception3D(in_channels)  # Extract spatial features
        self.rnn = nn.LSTM(1024, hidden_dim, n_layers, dropout=dropout)  # Temporal processing
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Forward pass for the encoder.

        Args:
            src (Tensor): Input tensor of shape [src_length, batch_size, in_channels, C, H, W].

        Returns:
            tuple: Hidden and cell states from the LSTM.
        """
        src_length = src.size(0)
        cnn_outputs = []

        for i in range(src_length):
            frame = src[i] 
            out = self.dropout(self.cnn(frame))
            cnn_outputs.append(out)

        cnn_outputs = torch.stack(cnn_outputs)  # [src_length, batch_size, embedding_dim]
        outputs, (hidden, cell) = self.rnn(cnn_outputs)  # LSTM processing

        return hidden, cell



class Decoder(nn.Module):
    """
    Decoder module using an LSTM and a linear layer for generating sequences.

    Args:
        output_dim (int): Size of the output vocabulary.
        embedding_dim (int): Dimension of the embedding layer.
        hidden_dim (int): Dimension of the hidden state in the LSTM.
        n_layers (int): Number of layers in the LSTM.
        dropout (float): Dropout rate applied to embeddings and LSTM outputs.
    """

    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        """
        Forward pass for the decoder.

        Args:
            input (Tensor): Input token indices of shape [batch_size].
            hidden (Tensor): Hidden state from the previous time step.
            cell (Tensor): Cell state from the previous time step.

        Returns:
            tuple: Prediction, hidden state, and cell state.
        """
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, embedding_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))  # [batch_size, output_dim]

        return prediction, hidden, cell



class Vid2Seq(nn.Module):
    """
    Sequence-to-sequence model for video-to-text generation.

    Combines an Encoder and a Decoder to process video input and generate text output.

    Args:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Ensure compatibility between encoder and decoder
        assert encoder.hidden_dim == decoder.hidden_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Number of layers in encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio):
        """
        Forward pass for the Vid2Seq model.

        Args:
            src (Tensor): Input video tensor of shape [src_length, batch_size, frames, C, H, W].
            trg (Tensor): Target sequence tensor of shape [trg_length, batch_size].
            teacher_forcing_ratio (float): Probability of using teacher forcing.

        Returns:
            Tensor: Output predictions of shape [trg_length, batch_size, output_dim].
        """
        trg_length, batch_size = trg.size(0), trg.size(1)
        outputs = []

        hidden, cell = self.encoder(src)  # Encode the video input
        input = trg[0, :]  # First input is the <sos> token
        
        for t in range(trg_length):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs.append(output)

            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else output.argmax(1)

        outputs = torch.stack(outputs)  # [trg_length, batch_size, output_dim]
        return outputs
