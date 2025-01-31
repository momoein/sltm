VIDEO = "VIDEO"
SENTENCE = "SENTENCE"

# Constants
MAX_LENGTH = 1_000
LOWER = True
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
MIN_FREQ = 1

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]

SENTENCE_TOKENS = "sentence_tokens"
SENTENCE_IDS = "sentence_ids"

# Constants
FRAME_SIZE = (3, 224, 224)  # Channels, Height, Width
BATCH_DIM = 4  # For 4D Tensor padding (sequence, frames, channels, height, width)

VIDEO_IDS = "video_ids"