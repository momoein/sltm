import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

from .seq2vid import Vid2Seq

from src.constants import VIDEO_IDS, SENTENCE_IDS


class LightVid2Seq(L.LightningModule):
    """
    A PyTorch Lightning module for training and evaluating the Vid2Seq model.

    Attributes:
        model (Vid2Seq): The Vid2Seq model combining the encoder and decoder.
        criterion (nn.CrossEntropyLoss): Loss function ignoring the pad index.
        teacher_forcing_ratio (float): Probability of using teacher forcing during training.
        clip (float): Maximum gradient norm for gradient clipping.
    """

    def __init__(self, encoder, decoder, pad_index, teacher_forcing_ratio, clip):
        super(LightVid2Seq, self).__init__()
        self.model = Vid2Seq(encoder, decoder)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.clip = clip

    def forward(self, src, trg, teacher_forcing_ratio):
        """
        Forward pass through the Vid2Seq model.

        Args:
            src (Tensor): Input video tensor of shape [src_length, batch_size, frames, C, H, W].
            trg (Tensor): Target sequence tensor of shape [trg_length, batch_size].
            teacher_forcing_ratio (float): Probability of using teacher forcing during training.

        Returns:
            Tensor: Output predictions of shape [trg_length, batch_size, output_dim].
        """
        return self.model(src, trg, teacher_forcing_ratio)

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        return optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Args:
            batch (dict): A batch of data containing "vi_ids" and "en_ids".
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        src = batch[VIDEO_IDS]
        trg = batch[SENTENCE_IDS]

        # Forward pass with teacher forcing
        output = self.model(src, trg, self.teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # Compute loss
        loss = self.criterion(output, trg)
        self.log("train_loss", loss)
        return loss

    def backward(self, loss):
        """
        Perform backpropagation and gradient clipping.

        Args:
            loss (torch.Tensor): Computed loss for the batch.
        """
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

    def valid_test_step(self, batch, batch_idx):
        """
        Shared logic for validation and testing steps.

        Args:
            batch (dict): A batch of data containing "vi_ids" and "en_ids".
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        src = batch[VIDEO_IDS]
        trg = batch[SENTENCE_IDS]

        output = self.model(src, trg, 0.)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # Compute loss
        return self.criterion(output, trg)

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Args:
            batch (dict): A batch of data containing "vi_ids" and "en_ids".
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        loss = self.valid_test_step(batch, batch_idx)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.

        Args:
            batch (dict): A batch of data containing "vi_ids" and "en_ids".
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        loss = self.valid_test_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss
