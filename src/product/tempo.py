"""Explicit, uncalibrated tempo baseline; never quantize events onto beats."""
from __future__ import annotations

import math

import librosa
import numpy as np
from scipy.signal import find_peaks

from src.audio_understanding import FRAME_SECONDS, SAMPLE_RATE


class TempoRequired(ValueError):
    pass


def estimate_tempo(mel, duration):
    """Reuse whole-song log-mel, retaining priors and octave ambiguity.

    Periodicity is a heuristic correlation score, never a calibrated chance
    of a correct beat. A single export timebase is not a tempo map.
    """
    config = dict(sr=SAMPLE_RATE, hop_length=512, stft_n_fft=2048, stft_center=True,
                  feature_frame_seconds=FRAME_SECONDS, feature_origin_seconds=0.,
                  flux_time_usage='periodicity_only_not_event_timestamps',
                  onset_lag=1, onset_max_size=1,
                  onset_center=False, start_bpm_prior=120., std_bpm_prior=1.,
                  ac_size_seconds=8., max_tempo=320., aggregate='mean')
    result = dict(schema='audiolab-tempo-estimate-v1', method='librosa-flux-autocorrelation-v1',
                  library_version=librosa.__version__, config=config,
                  analysis_range_seconds=[0,duration], source='estimated',
                  status='needs_tempo', timebase_bpm=None, musical_tempo_estimate=None,
                  candidates=[], confidence_calibration='unvalidated', tempo_map=None,
                  selection_policy='global-librosa-estimate-directly-as-timebase-v1')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('tempo analysis requires positive finite duration')
    array = mel.detach().cpu().numpy() if hasattr(mel,'detach') else np.asarray(mel)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('tempo analysis requires finite log-mel')
    if duration < 4:
        return dict(result, reason='audio_too_short_for_tempo')
    envelope = librosa.onset.onset_strength(S=array, sr=SAMPLE_RATE, hop_length=512,
                                           lag=1, max_size=1, center=False)
    peak = float(np.max(envelope))
    if peak <= 1e-6:
        return dict(result, reason='no_onset_strength')
    peaks, _ = find_peaks(envelope, distance=max(1,round(.08/FRAME_SECONDS)), prominence=peak*.1)
    if len(peaks) < 4:
        return dict(result, reason='too_few_onset_peaks', onset_peak_count=len(peaks))
    window = round(8/FRAME_SECONDS)
    tempogram = librosa.feature.tempogram(onset_envelope=envelope, sr=SAMPLE_RATE,
                                         hop_length=512, win_length=window)
    kwargs = dict(tg=tempogram, sr=SAMPLE_RATE, hop_length=512, start_bpm=120.,
                  std_bpm=1., max_tempo=320.)
    bpm = float(librosa.feature.tempo(**kwargs).item())
    if not math.isfinite(bpm) or not 1 <= bpm <= 1000:
        return dict(result, reason='no_valid_tempo_estimate')
    dynamic = np.asarray(librosa.feature.tempo(**kwargs, aggregate=None))
    # Ignore local estimates in silent windows, where a prior can dominate.
    energy = np.convolve(envelope, np.ones(min(window,len(envelope)))/min(window,len(envelope)), mode='same')
    active = energy > float(energy.max())*.05
    local = dynamic[active]
    low, high = [float(x) for x in np.percentile(local,[10,90])]
    spread = (high-low)/max(float(np.median(local)),1e-8)
    correlation = tempogram.mean(axis=-1)
    def strength(candidate):
        lag = round(60/(candidate*FRAME_SECONDS))
        return float(correlation[lag]) if 0 < lag < len(correlation) else None
    candidates = [dict(bpm=bpm*factor, relative=factor, periodicity_score=strength(bpm*factor),
                       selected=factor==1) for factor in [.5,1.,2.] if 1 <= bpm*factor <= 1000]
    selected_strength = strength(bpm)
    octave_strengths = [c['periodicity_score'] for c in candidates if not c['selected'] and c['periodicity_score'] is not None]
    ambiguous = bool(octave_strengths and max(octave_strengths) >= max(selected_strength or 0,.001)*.8)
    # Correlation strength cannot resolve metrical interpretation. Even a clean
    # click track may be returned at half speed under the declared prior.
    diagnostics = ['half_double_interpretation_unresolved']
    if spread > .12:
        diagnostics.append('local_tempo_variation_or_octave_switching')
    if ambiguous:
        diagnostics.append('half_double_tempo_ambiguity')
    if selected_strength is None or selected_strength < .15:
        diagnostics.append('weak_periodicity')
    return dict(result, status='estimated_unconfirmed', timebase_bpm=bpm,
                musical_tempo_estimate=dict(bpm=bpm, verified=False), candidates=candidates,
                onset_peak_count=len(peaks), periodicity_score=selected_strength,
                uncertainty=dict(diagnostics=diagnostics, local_p10_bpm=low,local_p90_bpm=high,
                                 local_relative_spread=spread,
                                 note='variation may be octave switching, not a verified tempo change'),
                reason='uncalibrated_algorithmic_estimate')


def provided_tempo(bpm, duration):
    if not math.isfinite(bpm) or not 1 <= bpm <= 1000:
        raise ValueError('BPM must be finite in [1, 1000]')
    return dict(schema='audiolab-tempo-estimate-v1',source='provided',status='provided',
                timebase_bpm=bpm,musical_tempo_estimate=None,analysis_range_seconds=None,
                duration_seconds=duration,method='manual_override',candidates=[],tempo_map=None,
                selection_policy='explicit-user-timebase-v1',confidence_calibration='not_applicable')
