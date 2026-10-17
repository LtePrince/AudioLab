"""Shared audio evidence, independent of difficulty, lanes and note types."""

SCHEMA = 'audiolab-audio-understanding-v1'
ANNOTATION_SCHEMA = 'audiolab-audio-events-v1'
FEATURE_SCHEMA = 'torchaudio-normalized-logmel-22050-2048-512-128-v1'
SAMPLE_RATE = 22050
HOP_LENGTH = 512
FRAME_SECONDS = HOP_LENGTH / SAMPLE_RATE
