"""
================================================================================
SONIC SIGNATURES: FROM AUDIO SIGNALS TO SONG RECOGNITION
GROUP 2 - TASK 6: PEAK-PAIR MATCHING (SHAZAM-STYLE RECOGNISER)
================================================================================
Implements:
1. Audio loading and spectrogram computation (consistent with Task 1 & 2).
2. 2D Spectral peak detection (local maxima filtering with thresholding).
3. Combinatorial peak-pairing and hash generation (Wang 2003 Shazam algorithm).
4. Inverted index database creation for O(1) hash lookup.
5. Time-offset voting histogram matching to identify song and start offset.
6. Visualization tool for offset-voting histograms.
================================================================================
"""

import numpy as np
import scipy.ndimage as ndimage
import librosa
import matplotlib.pyplot as plt
from collections import defaultdict

# ------------------------------------------------------------------------------
# 1. DSP & AUDIO CONFIGURATION PARAMETERS (Consistent with Tasks 1, 2 & 5)
# ------------------------------------------------------------------------------
TARGET_SRATE = 22050        # Sampling rate in Hz
SAMPLE_COUNT = 2048        # FFT window size (N_FFT)
HOP_LENGTH = 512           # Hop length in samples (75% overlap)
MAX_FREQ = 5000            # Upper frequency bound in Hz
MIN_FREQ = 80              # Lower frequency bound to ignore DC / hum

# Peak detection parameters
PEAK_NEIGHBORHOOD_FREQ = 10  # Minimum frequency separation in bins (~108 Hz)
PEAK_NEIGHBORHOOD_TIME = 10  # Minimum time separation in frames (~232 ms)
AMP_THRESHOLD_DB = -40.0     # Minimum decibel magnitude for peak candidate

# Combinatorial peak-pairing parameters (Shazam Target Zone)
TARGET_ZONE_T_MIN = 2       # Minimum forward time offset (~46 ms)
TARGET_ZONE_T_MAX = 50      # Maximum forward time offset (~1.16 s)
TARGET_ZONE_F_MAX = 80      # Maximum frequency difference in bins (~860 Hz)
FAN_OUT = 7                 # Max target peaks paired with each anchor peak

# ------------------------------------------------------------------------------
# 2. AUDIO LOADING & SPECTROGRAM GENERATION
# ------------------------------------------------------------------------------
def load_audio(path, target_srate=TARGET_SRATE, normalize=True):
    """
    Loads an audio file, converts it to mono, resamples to target_srate,
    and normalizes peak amplitude to 1.0.
    """
    y, srate = librosa.load(path, sr=None, mono=False)
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    if srate != target_srate:
        y_resampled = librosa.resample(y_mono, orig_sr=srate, target_sr=target_srate)
    else:
        y_resampled = y_mono
    if normalize:
        peak = np.max(np.abs(y_resampled))
        if peak > 0:
            y_resampled = y_resampled / peak
    return y_resampled, target_srate

def comp_spectrogram(y, sample_count=SAMPLE_COUNT, hop_length=HOP_LENGTH):
    """
    Computes a decibel magnitude spectrogram using Short-Time Fourier Transform.
    """
    y_stft = librosa.stft(y, n_fft=sample_count, hop_length=hop_length)
    y_mag = np.abs(y_stft)
    return librosa.amplitude_to_db(y_mag, ref=np.max)

# ------------------------------------------------------------------------------
# 3. SPECTRAL PEAK DETECTION (CONSTELLATION MAP)
# ------------------------------------------------------------------------------
def find_spectral_peaks(spec_db,
                        neighborhood_freq=PEAK_NEIGHBORHOOD_FREQ,
                        neighborhood_time=PEAK_NEIGHBORHOOD_TIME,
                        amp_threshold_db=AMP_THRESHOLD_DB,
                        min_freq=MIN_FREQ,
                        max_freq=MAX_FREQ,
                        sr=TARGET_SRATE,
                        sample_count=SAMPLE_COUNT):
    """
    Extracts prominent local maxima from a decibel spectrogram.
    Uses a 2D maximum filter to suppress non-maximal points within
    (2*freq + 1, 2*time + 1) bins.
    
    Returns:
        peaks: List of tuples (freq_bin, time_frame, amp_db) sorted by time_frame.
    """
    # Restrict to meaningful frequency bounds
    freq_resolution = sr / sample_count
    min_bin = max(0, int(min_freq / freq_resolution))
    max_bin = min(int(max_freq / freq_resolution), spec_db.shape[0] - 1)

    bounded_spec = spec_db[min_bin:max_bin + 1, :]
    filter_size = (2 * neighborhood_freq + 1, 2 * neighborhood_time + 1)
    local_max = ndimage.maximum_filter(bounded_spec, size=filter_size)

    # A cell is a peak if it equals the neighborhood maximum and exceeds the threshold
    is_peak = (bounded_spec == local_max) & (bounded_spec >= amp_threshold_db)

    peak_freq_indices, peak_time_indices = np.where(is_peak)
    
    # Map frequency indices back to original spectrogram coordinates
    peaks = []
    for f_idx, t_idx in zip(peak_freq_indices, peak_time_indices):
        orig_f_bin = f_idx + min_bin
        amp = spec_db[orig_f_bin, t_idx]
        peaks.append((orig_f_bin, int(t_idx), float(amp)))

    # Sort peaks by time frame, then frequency bin
    peaks.sort(key=lambda p: (p[1], p[0]))
    return peaks

# ------------------------------------------------------------------------------
# 4. COMBINATORIAL PEAK-PAIRING & HASH GENERATION (WANG 2003)
# ------------------------------------------------------------------------------
def generate_peak_pair_hashes(peaks,
                              target_zone_t_min=TARGET_ZONE_T_MIN,
                              target_zone_t_max=TARGET_ZONE_T_MAX,
                              target_zone_f_max=TARGET_ZONE_F_MAX,
                              fan_out=FAN_OUT):
    """
    Forms pairs of spectral peaks following Avery Wang's Shazam algorithm.
    For each anchor peak (f1, t1), searches a forward target zone for up to
    fan_out target peaks (f2, t2) where:
        t1 + target_zone_t_min <= t2 <= t1 + target_zone_t_max
        |f2 - f1| <= target_zone_f_max
        
    Hash key format:
        (f1, f2, delta_t) represented as an integer: (f1 << 20) | (f2 << 8) | delta_t
        or as a tuple (f1, f2, delta_t).
        
    Returns:
        fingerprints: List of (hash_key, t1_anchor_frame)
    """
    fingerprints = []
    n_peaks = len(peaks)

    for i in range(n_peaks):
        f1, t1, _ = peaks[i]
        targets_paired = 0

        for j in range(i + 1, n_peaks):
            f2, t2, _ = peaks[j]
            delta_t = t2 - t1

            if delta_t < target_zone_t_min:
                continue
            if delta_t > target_zone_t_max:
                break  # Since peaks are sorted chronologically, no future peak can satisfy this

            if abs(f2 - f1) <= target_zone_f_max:
                # Pack (f1, f2, delta_t) into a hash key
                # f1 (0-1024, ~10 bits), f2 (0-1024, ~10 bits), delta_t (0-255, ~8 bits)
                hash_key = (int(f1) << 20) | (int(f2) << 8) | int(delta_t)
                fingerprints.append((hash_key, t1))

                targets_paired += 1
                if targets_paired >= fan_out:
                    break

    return fingerprints

# ------------------------------------------------------------------------------
# 5. SHAZAM PEAK-PAIR RECOGNISER CLASS
# ------------------------------------------------------------------------------
class ShazamPeakPairRecogniser:
    """
    Complete Shazam-style audio recogniser utilizing combinatorial peak-pair
    hashing and time-offset voting histograms.
    """
    def __init__(self,
                 sr=TARGET_SRATE,
                 sample_count=SAMPLE_COUNT,
                 hop_length=HOP_LENGTH,
                 neighborhood_freq=PEAK_NEIGHBORHOOD_FREQ,
                 neighborhood_time=PEAK_NEIGHBORHOOD_TIME,
                 amp_threshold_db=AMP_THRESHOLD_DB,
                 target_zone_t_min=TARGET_ZONE_T_MIN,
                 target_zone_t_max=TARGET_ZONE_T_MAX,
                 target_zone_f_max=TARGET_ZONE_F_MAX,
                 fan_out=FAN_OUT):
        self.sr = sr
        self.sample_count = sample_count
        self.hop_length = hop_length
        self.neighborhood_freq = neighborhood_freq
        self.neighborhood_time = neighborhood_time
        self.amp_threshold_db = amp_threshold_db
        self.target_zone_t_min = target_zone_t_min
        self.target_zone_t_max = target_zone_t_max
        self.target_zone_f_max = target_zone_f_max
        self.fan_out = fan_out

        # Inverted index: hash_key -> list of (song_name, anchor_frame)
        self.database = defaultdict(list)
        self.songs_info = {}

    def extract_fingerprints(self, y):
        """
        Processes audio to extract STFT peaks and generates peak-pair hashes.
        Returns:
            fingerprints: List of (hash_key, anchor_frame)
            peaks: List of (f, t, amp)
            spec_db: Decibel spectrogram
        """
        spec_db = comp_spectrogram(y, sample_count=self.sample_count, hop_length=self.hop_length)
        peaks = find_spectral_peaks(
            spec_db,
            neighborhood_freq=self.neighborhood_freq,
            neighborhood_time=self.neighborhood_time,
            amp_threshold_db=self.amp_threshold_db,
            sr=self.sr,
            sample_count=self.sample_count
        )
        fingerprints = generate_peak_pair_hashes(
            peaks,
            target_zone_t_min=self.target_zone_t_min,
            target_zone_t_max=self.target_zone_t_max,
            target_zone_f_max=self.target_zone_f_max,
            fan_out=self.fan_out
        )
        return fingerprints, peaks, spec_db

    def index_song(self, song_name, y):
        """
        Extracts fingerprints from audio and populates the inverted index.
        """
        fingerprints, peaks, _ = self.extract_fingerprints(y)
        duration_sec = len(y) / self.sr

        for hash_key, anchor_t in fingerprints:
            self.database[hash_key].append((song_name, anchor_t))

        self.songs_info[song_name] = {
            "duration_sec": duration_sec,
            "num_peaks": len(peaks),
            "num_fingerprints": len(fingerprints)
        }

    def recognize(self, query_y):
        """
        Identifies the source song of an unknown query clip using offset voting.
        
        Returns:
            predicted_song: Name of best matching song
            best_offset_sec: Estimated time in seconds where query starts in reference track
            best_score: Peak vote count at best offset
            offset_histograms: Dict mapping song_name -> (offsets_sec_array, vote_counts_array)
            leaderboard: Ranked list of (song_name, peak_votes, best_offset_sec, confidence_ratio)
        """
        if not self.database:
            raise ValueError("Database is empty! Call index_song() before recognition.")

        query_fps, _, _ = self.extract_fingerprints(query_y)
        if not query_fps:
            raise ValueError("No spectral peaks detected in query clip; audio may be silent.")

        # Track offset votes: song_name -> {offset_frame: count}
        # offset_frame = song_anchor_t - query_anchor_t
        song_offset_votes = defaultdict(lambda: defaultdict(int))

        for q_hash, q_t in query_fps:
            matches = self.database.get(q_hash, None)
            if matches:
                for song_name, song_t in matches:
                    offset = song_t - q_t
                    song_offset_votes[song_name][offset] += 1

        leaderboard = []
        offset_histograms = {}

        # Evaluate each song in the database
        for song_name in self.songs_info.keys():
            votes_dict = song_offset_votes.get(song_name, {})
            if votes_dict:
                # Find maximum votes in any offset bin
                best_offset_frame, peak_votes = max(votes_dict.items(), key=lambda x: x[1])
                best_offset_sec = best_offset_frame * (self.hop_length / self.sr)

                # Convert votes_dict to sorted numpy arrays for histogram plotting
                sorted_items = sorted(votes_dict.items(), key=lambda x: x[0])
                offsets_sec = np.array([item[0] * (self.hop_length / self.sr) for item in sorted_items])
                vote_counts = np.array([item[1] for item in sorted_items])

                # Calculate confidence: peak votes vs median background vote count
                median_noise = float(np.median(vote_counts)) if len(vote_counts) > 1 else 1.0
                confidence = float(peak_votes / max(median_noise, 1.0))
            else:
                best_offset_sec = 0.0
                peak_votes = 0
                offsets_sec = np.array([])
                vote_counts = np.array([])
                confidence = 0.0

            offset_histograms[song_name] = (offsets_sec, vote_counts)
            leaderboard.append((song_name, peak_votes, best_offset_sec, confidence))

        # Rank candidates by peak vote count
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        predicted_song, best_score, best_offset_sec, _ = leaderboard[0]

        return predicted_song, best_offset_sec, best_score, offset_histograms, leaderboard

# ------------------------------------------------------------------------------
# 6. DELIVERABLE HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def identify_query_clip_shazam(query_audio, songs_dict):
    """
    Standard high-level deliverable function for Task 6.
    Indexes a dictionary of {song_name: (audio, sr)} and identifies query_audio.
    
    Returns:
        predicted_song, best_offset_sec, best_score, offset_histograms
    """
    recogniser = ShazamPeakPairRecogniser()
    for name, (y, sr) in songs_dict.items():
        recogniser.index_song(name, y)
    predicted_song, best_offset_sec, score, offset_histograms, _ = recogniser.recognize(query_audio)
    return predicted_song, best_offset_sec, score, offset_histograms

def extract_query_clip(y, sr, start_sec, duration_sec, noise_snr_db=None, gain=1.0):
    """
    Utility function to slice a segment from an audio track and optionally add noise.
    """
    start_sample = int(start_sec * sr)
    end_sample = start_sample + int(duration_sec * sr)
    clip = y[start_sample:min(end_sample, len(y))].copy() * gain
    if noise_snr_db is not None:
        signal_power = np.mean(clip ** 2)
        noise_power = signal_power / (10 ** (noise_snr_db / 10))
        noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-12)), len(clip))
        clip = clip + noise
    return clip

def plot_offset_voting_histogram(offset_histograms, predicted_song,
                                 best_offset_sec=None, query_label="Query Clip",
                                 save_path=None):
    """
    Generates and optionally saves the required offset-voting histogram plot.
    Shows the sharp alignment spike for the matching song versus flat noise
    for non-matching songs.
    """
    n_songs = len(offset_histograms)
    fig, axes = plt.subplots(n_songs, 1, figsize=(11, 3.2 * n_songs), sharex=False)
    if n_songs == 1:
        axes = [axes]

    for ax, (song_name, (offsets, counts)) in zip(axes, offset_histograms.items()):
        is_winner = (song_name == predicted_song)
        bar_color = "#1f77b4" if is_winner else "#7f7f7f"
        alpha_val = 0.85 if is_winner else 0.55

        if len(offsets) > 0:
            ax.bar(offsets, counts, width=0.1, color=bar_color, alpha=alpha_val,
                   label=f"{song_name} votes")
            max_c = np.max(counts)
            best_off = offsets[np.argmax(counts)]

            if is_winner:
                ax.axvline(best_off, color="red", linestyle="--", linewidth=1.8,
                           label=f"Peak alignment @ {best_off:.2f}s ({max_c} votes)")
                ax.annotate(f"Identified Match!\nOffset = {best_off:.2f}s\nVotes = {max_c}",
                            xy=(best_off, max_c),
                            xytext=(best_off + 2.0, max(max_c * 0.7, 1)),
                            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
                            bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="red", lw=1.2),
                            fontsize=9, fontweight="bold")
            ax.set_ylim(0, max_c * 1.25 if max_c > 0 else 5)
        else:
            ax.text(0.5, 0.5, "No Matching Hashes", horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes, color="gray")

        ax.set_title(f"Offset Voting Histogram: {song_name} {'[WINNING MATCH]' if is_winner else '[Candidate]'}",
                     fontsize=11, fontweight="bold" if is_winner else "normal")
        ax.set_xlabel("Time Offset (Song Anchor Time - Query Anchor Time in seconds)", fontsize=9)
        ax.set_ylabel("Matching Hash Votes", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle(f"Shazam Peak-Pair Offset Alignment | Query: '{query_label}'",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f" -> Offset-voting histogram saved successfully to: {save_path}")
    return fig

