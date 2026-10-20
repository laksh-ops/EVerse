"""
================================================================================
SONIC SIGNATURES: FROM AUDIO SIGNALS TO SONG RECOGNITION
GROUP 2 - TASK 6: COMPREHENSIVE TEST SUITE & VERIFICATION SCRIPT
================================================================================
This script executes a full suite of automated tests for Task 6:
1. Auto-detects Python environment (.venv) if run from unconfigured shell.
2. Generates 3 distinct reference tracks with unique non-repeating note progressions.
3. Unit tests spectral peak detection and constellation map extraction.
4. Unit tests combinatorial peak-pairing and hash generation.
5. Tests full database inverted-index population.
6. Tests recognition of 10-second unknown query clips sliced at arbitrary offsets.
7. Tests noise robustness under challenging SNR conditions (15 dB, 10 dB, 5 dB).
8. Verifies the time-offset voting histogram deliverable and saves plot to disk.
================================================================================
"""

import os
import sys

# Auto-detect and re-execute with .venv python if numpy/librosa are missing in current environment
try:
    import numpy as np
    import librosa
except ModuleNotFoundError:
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
        import subprocess
        result = subprocess.run([venv_python, __file__] + sys.argv[1:])
        sys.exit(result.returncode)
    else:
        raise

import matplotlib
matplotlib.use("Agg")  # Headless backend for automated test runs
import matplotlib.pyplot as plt

from task6_peak_pair_matching import (
    TARGET_SRATE,
    SAMPLE_COUNT,
    HOP_LENGTH,
    comp_spectrogram,
    find_spectral_peaks,
    generate_peak_pair_hashes,
    ShazamPeakPairRecogniser,
    identify_query_clip_shazam,
    extract_query_clip,
    plot_offset_voting_histogram
)

def create_synthetic_song(base_freq, duration_sec=40.0, sr=TARGET_SRATE, seed=42):
    """
    Synthesizes a realistic audio track composed of an evolving, non-repeating
    musical progression across time to ensure unique constellation maps.
    """
    rng = np.random.RandomState(seed)
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    audio = np.zeros(total_samples, dtype=float)

    note_duration = 0.5  # half second per note
    samples_per_note = int(note_duration * sr)
    num_notes = int(duration_sec / note_duration)

    intervals = [1.0, 1.2, 1.25, 1.333, 1.5, 1.667, 1.75, 2.0]

    for i in range(num_notes):
        start_sample = i * samples_per_note
        end_sample = min(start_sample + samples_per_note, total_samples)
        t_note = t[start_sample:end_sample]

        int1 = float(rng.choice(intervals))
        int2 = float(rng.choice(intervals))
        while int2 == int1:
            int2 = float(rng.choice(intervals))

        f1 = base_freq * int1
        f2 = base_freq * int2

        env = np.sin(np.pi * np.linspace(0, 1, len(t_note))) ** 0.5
        note_sig = (0.6 * np.sin(2 * np.pi * f1 * t_note) + 0.4 * np.sin(2 * np.pi * f2 * t_note)) * env
        audio[start_sample:end_sample] += note_sig

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def build_test_database():
    """
    Builds 3 distinct reference tracks (40 seconds each).
    Can also load real MP3 files if placed in workspace or Colab.
    """
    sr = TARGET_SRATE
    candidate_mp3s = {
        "Song1": "Achachak - First Movement.mp3",
        "Song2": "Joint C Beat Laboratory - Margo.mp3",
        "Song3": "Joint C Beat Laboratory - Nerves.mp3"
    }

    songs_db = {}
    found_real = True
    for name, filename in candidate_mp3s.items():
        search_dirs = [".", "..", "/content", "audio", "songs"]
        found_path = None
        for d in search_dirs:
            p = os.path.join(d, filename)
            if os.path.exists(p):
                found_path = p
                break
        if found_path is not None:
            y, s = librosa.load(found_path, sr=sr, mono=True)
            y = y / np.max(np.abs(y))
            songs_db[name] = (y, sr)
        else:
            found_real = False
            break

    if not found_real or len(songs_db) < 3:
        print(" -> Using synthesized multi-tone tracks with distinct non-repeating progressions...")
        y1 = create_synthetic_song(base_freq=220.0, duration_sec=40.0, sr=sr, seed=101)
        y2 = create_synthetic_song(base_freq=550.0, duration_sec=40.0, sr=sr, seed=202)
        y3 = create_synthetic_song(base_freq=1100.0, duration_sec=40.0, sr=sr, seed=303)

        songs_db["Song1"] = (y1, sr)
        songs_db["Song2"] = (y2, sr)
        songs_db["Song3"] = (y3, sr)
    else:
        print(" -> Found real MP3 reference files; loaded successfully.")

    return songs_db

def run_tests():
    print("=" * 75)
    print("        TASK 6: SHAZAM-STYLE PEAK-PAIR RECOGNISER TEST SUITE        ")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # STEP 1: CREATE OR LOAD DATABASE
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing reference songs database...")
    songs_db = build_test_database()
    sr = TARGET_SRATE
    for name, (audio, _) in songs_db.items():
        dur = len(audio) / sr
        print(f"   * {name}: Duration = {dur:.1f}s | Sample Rate = {sr} Hz")
    assert len(songs_db) == 3, "Database must contain 3 songs!"
    print("   -> Reference tracks initialized successfully.")

    # -------------------------------------------------------------------------
    # STEP 2: UNIT TEST - SPECTRAL PEAK DETECTION
    # -------------------------------------------------------------------------
    print("\n[Step 2] Unit testing 2D spectral peak detection...")
    y_test, _ = songs_db["Song1"]
    spec_db = comp_spectrogram(y_test, sample_count=SAMPLE_COUNT, hop_length=HOP_LENGTH)
    peaks = find_spectral_peaks(spec_db, neighborhood_freq=10, neighborhood_time=10, amp_threshold_db=-40.0)
    print(f"   -> Spectrogram shape: {spec_db.shape} (Freq Bins x Time Frames)")
    print(f"   -> Detected {len(peaks)} local maxima peaks.")
    assert len(peaks) > 20, "Peak detector should find distinct spectral peaks in musical audio!"
    
    # Check peak ordering
    is_ordered = all(peaks[i][1] <= peaks[i+1][1] for i in range(len(peaks)-1))
    assert is_ordered, "Peaks must be sorted chronologically by time frame!"
    print("   -> Peak detection test: PASSED (Sorted and non-empty).")

    # -------------------------------------------------------------------------
    # STEP 3: UNIT TEST - COMBINATORIAL PEAK-PAIR HASH GENERATION
    # -------------------------------------------------------------------------
    print("\n[Step 3] Unit testing combinatorial peak-pair hashing...")
    fingerprints = generate_peak_pair_hashes(peaks, target_zone_t_min=2, target_zone_t_max=50, fan_out=7)
    print(f"   -> Generated {len(fingerprints)} peak-pair fingerprint hashes.")
    assert len(fingerprints) > 0, "Failed to form peak pairs from detected peaks!"
    sample_hash, sample_t = fingerprints[0]
    print(f"   -> Sample fingerprint: Hash = {sample_hash} (Hex: {hex(sample_hash)}), Anchor Frame = {sample_t}")
    print("   -> Peak-pair hashing test: PASSED.")

    # -------------------------------------------------------------------------
    # STEP 4: INDEXING SONGS INTO SHAZAM RECOGNISER
    # -------------------------------------------------------------------------
    print("\n[Step 4] Indexing songs into ShazamPeakPairRecogniser inverted index...")
    recogniser = ShazamPeakPairRecogniser()
    for name, (audio, _) in songs_db.items():
        recogniser.index_song(name, audio)
        info = recogniser.songs_info[name]
        print(f"   * Indexed '{name}': {info['num_peaks']} peaks, {info['num_fingerprints']} hashes.")
    
    unique_hashes = len(recogniser.database)
    total_entries = sum(len(lst) for lst in recogniser.database.values())
    print(f"   -> Total unique hashes in inverted index: {unique_hashes}")
    print(f"   -> Total postings entries: {total_entries}")
    assert unique_hashes > 50, "Inverted index must contain rich unique hashes!"

    # -------------------------------------------------------------------------
    # STEP 5: EXACT TRACK SANITY CHECK (Offset should be 0.0s)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Sanity check: Querying full Song1 against database...")
    pred_s1, off_s1, votes_s1, _, _ = recogniser.recognize(songs_db["Song1"][0])
    print(f"   -> Prediction: '{pred_s1}' | Offset: {off_s1:.2f}s | Matching Votes: {votes_s1}")
    assert pred_s1 == "Song1", "Full song sanity check failed to predict Song1!"
    assert abs(off_s1) < 0.1, f"Expected offset ~0.0s, got {off_s1}s"
    print("   -> Identity sanity check: PASSED.")

    # -------------------------------------------------------------------------
    # STEP 6: ARBITRARY QUERY CLIPS AT VARIOUS OFFSETS
    # -------------------------------------------------------------------------
    print("\n[Step 6] Testing 10-second unknown query clips sliced at arbitrary offsets...")
    test_queries = [
        ("Song1", 6.5, 10.0),
        ("Song2", 14.0, 10.0),
        ("Song3", 22.5, 10.0)
    ]

    all_queries_passed = True
    last_histograms = None
    last_winner = None
    last_best_off = None

    for true_song, start_sec, dur_sec in test_queries:
        y_ref = songs_db[true_song][0]
        query = extract_query_clip(y_ref, sr, start_sec=start_sec, duration_sec=dur_sec)

        pred_song, est_offset, peak_score, offset_hists, leaderboard = recogniser.recognize(query)

        # Standalone deliverable function check
        deliv_pred, deliv_offset, deliv_score, deliv_hists = identify_query_clip_shazam(query, songs_db)
        assert deliv_pred == pred_song, "Standalone helper function mismatch!"

        offset_err = abs(est_offset - start_sec)
        passed = (pred_song == true_song) and (offset_err <= 0.25)
        if not passed:
            all_queries_passed = False

        status = "PASSED [OK]" if passed else "FAILED"
        print(f"\n   Query: Sliced from [{true_song}] at t = {start_sec:.2f}s (duration = {dur_sec:.1f}s)")
        print(f"   -> Predicted Song: '{pred_song}' (Truth: '{true_song}')")
        print(f"   -> Estimated Offset: {est_offset:.2f}s (Error: {offset_err*1000:.1f} ms)")
        print(f"   -> Peak Votes: {peak_score} | Status: {status}")
        print("      Candidate Leaderboard:")
        for rank, (cand, votes, off, conf) in enumerate(leaderboard, 1):
            print(f"        {rank}. {cand:6s} | Peak Votes: {votes:4d} | Est. Offset: {off:5.2f}s | Confidence: {conf:.1f}x")

        last_histograms = offset_hists
        last_winner = pred_song
        last_best_off = est_offset

    assert all_queries_passed, "One or more clean query clips failed recognition or offset alignment!"

    # -------------------------------------------------------------------------
    # STEP 7: NOISE ROBUSTNESS TEST (Additive Gaussian White Noise)
    # -------------------------------------------------------------------------
    print("\n[Step 7] Testing noise robustness under additive noise...")
    noise_snrs = [15.0, 10.0, 5.0]  # dB SNR
    target_song = "Song2"
    target_start = 12.0
    target_dur = 10.0
    ref_audio = songs_db[target_song][0]

    for snr in noise_snrs:
        noisy_query = extract_query_clip(ref_audio, sr, start_sec=target_start, duration_sec=target_dur, noise_snr_db=snr)
        pred_noisy, off_noisy, score_noisy, _, _ = recogniser.recognize(noisy_query)
        err = abs(off_noisy - target_start)
        noisy_passed = (pred_noisy == target_song) and (err <= 0.3)
        print(f"   * Noise SNR = {snr:4.1f} dB -> Predicted: '{pred_noisy}' | Offset: {off_noisy:.2f}s | Votes: {score_noisy:3d} | {'PASSED' if noisy_passed else 'DEGRADED'}")
        assert noisy_passed, f"Shazam recognition should succeed at {snr} dB SNR!"
    print("   -> Noise robustness test: PASSED.")

    # -------------------------------------------------------------------------
    # STEP 8: DELIVERABLE HISTOGRAM PLOT GENERATION & VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[Step 8] Generating and saving offset-voting histogram deliverable...")
    output_plot_path = "task6_offset_histogram.png"
    fig = plot_offset_voting_histogram(
        last_histograms,
        predicted_song=last_winner,
        best_offset_sec=last_best_off,
        query_label=f"10s Clip from {last_winner} at {test_queries[-1][1]}s",
        save_path=output_plot_path
    )
    plt.close(fig)
    assert os.path.exists(output_plot_path), "Failed to save offset-voting histogram image!"
    file_size_kb = os.path.getsize(output_plot_path) / 1024
    print(f"   -> Successfully verified {output_plot_path} ({file_size_kb:.1f} KB).")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("   >>> ALL TASK 6 VERIFICATION & UNIT TESTS PASSED SUCCESSFULLY! <<<   ")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
