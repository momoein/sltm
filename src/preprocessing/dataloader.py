import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.io import read_video

from src.constants import (
    FRAME_SIZE,
    VIDEO,
    VIDEO_IDS,
    SENTENCE_IDS
)


# Utility function for padding videos to a fixed length
def pad_video_sequence(video, max_frames):
    pad_size = max_frames - video.size(0)
    return F.pad(video, pad=(0, 0, 0, 0, 0, 0, 0, pad_size), mode='constant', value=0)


# Frame padding and sequence adjustment function
def frame_pad_sequence_videos(batch_video):
    max_frames = max(video.size(0) for video in batch_video)
    if max_frames % 10 != 0:
        max_frames += 10 - max_frames % 10  # Adjust to nearest multiple of 10

    # Pad videos to max_frames
    padded_videos = [
        pad_video_sequence(video, max_frames).view(max_frames // 10, 10, *FRAME_SIZE)
        for video in batch_video
    ]
    return padded_videos


# Function to sample frames from the video at a specified rate
def frames_sampling(batch_video, rate):
    new_batch_video = []

    for i, video in enumerate(batch_video):
        if rate > video.size(0):
            video = pad_video_sequence(video, rate)  # Pad video if frames < rate

        sampling_idxs = list(range(0, video.size(0), video.size(0) // rate))[:rate]
        new_video = torch.stack([video[idx] for idx in sampling_idxs])
        new_batch_video.append(new_video.view(1, rate, *FRAME_SIZE))

    return new_batch_video


# Collate function to prepare batches
def get_collate_fn(pad_index: float, video_dir: str, sampling_rate: int = 10):
    def collate_fn(batch):
        # Extract English and Vietnamese ids
        batch_sentence_ids = [example[SENTENCE_IDS] for example in batch]
        batch_video_paths = [f"{video_dir}/{example[VIDEO]}" for example in batch]

        # Read videos
        batch_video_ids = [
            read_video(vi_path, pts_unit="sec", output_format="TCHW")[0] for vi_path in batch_video_paths
        ]

        # Pad English sequences
        batch_sentence_ids = nn.utils.rnn.pad_sequence(batch_sentence_ids, padding_value=pad_index)

        # Pad and sample video frames
        batch_video_ids = nn.utils.rnn.pad_sequence(
            frames_sampling(batch_video_ids, sampling_rate),
            batch_first=False
        ).float() / 255  # Normalize video frames

        # Return the batch
        return {
            SENTENCE_IDS: batch_sentence_ids,
            VIDEO_IDS: batch_video_ids,
        }

    return collate_fn


# DataLoader setup function
def get_data_loader(dataset, batch_size, pad_index, video_dir, sampling_rate=10, shuffle=False, num_worker=1):
    collate_fn = get_collate_fn(pad_index, video_dir, sampling_rate)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_worker,
    )
