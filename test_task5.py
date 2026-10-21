import numpy as np
from task5_baseline_recogniser import (
    TARGET_SRATE,
    extract_average_spectral_features,
    compute_similarity,
    BaselineSpectralRecogniser,
    extract_query_clip,
    identify_query_clip,
    sliding_window_recogniser
)

def run_tests():
    print("=" * 70)
    print("             TASK 5 VERIFICATION & TEST SUITE             ")
    print("=" * 70)

    # 1. GENERATE SYNTHETIC TEST AUDIO (3 DISTINCT TRACKS)
    print("\n[Step 1] Creating 3 distinct reference tracks for testing...")
    duration = 30.0  # 30 seconds each
    sr = TARGET_SRATE
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Song 1: Low-frequency acoustic/bass (220 Hz + 440 Hz)
    y1 = 0.7 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    y1 = y1 / np.max(np.abs(y1))

    # Song 2: Mid-frequency melody/brass (880 Hz + 1320 Hz)
    y2 = 0.5 * np.sin(2 * np.pi * 880 * t) + 0.5 * np.sin(2 * np.pi * 1320 * t)
    y2 = y2 / np.max(np.abs(y2))

    # Song 3: High-frequency synth/chimes (1760 Hz + 2640 Hz)
    y3 = 0.6 * np.sin(2 * np.pi * 1760 * t) + 0.4 * np.sin(2 * np.pi * 2640 * t)
    y3 = y3 / np.max(np.abs(y3))

    songs_db = {
        "Song1": (y1, sr),
        "Song2": (y2, sr),
        "Song3": (y3, sr)
    }
    print(" -> Successfully generated 3 reference songs (30s each, 22050 Hz).")

    # 2. TEST RECOGNISER INDEXING
    print("\n[Step 2] Testing database indexing...")
    recogniser = BaselineSpectralRecogniser(metric="cosine")
    for name, (audio, sample_rate) in songs_db.items():
        recogniser.index_song(name, audio, sample_rate)
    print(f" -> Indexed {len(recogniser.database)} songs successfully.")
    assert len(recogniser.database) == 3, "Database indexing failed!"

    # 3. TEST FEATURE EXTRACTION & SHAPE
    feat1, freq_axis = extract_average_spectral_features(y1, sr=sr)
    print(f" -> Feature vector size: {feat1.shape[0]} frequency bins (up to 5000 Hz).")
    assert feat1.ndim == 1, "Feature vector must be 1-dimensional!"

    # 4. TEST IDENTICAL QUERY SANITY CHECK (Cosine similarity with self should be ~1.0)
    print("\n[Step 3] Running identity sanity check (Self-similarity)...")
    self_sim = compute_similarity(feat1, feat1, metric="cosine")
    print(f" -> Song1 self-similarity: {self_sim:.6f}")
    assert abs(self_sim - 1.0) < 1e-4, "Self-similarity must be 1.0!"

    # 5. TEST QUERY CLIPS AT RANDOM OFFSETS
    print("\n[Step 4] Testing recognition of 10-second unknown query clips...")
    test_cases = [
        ("Song1", 5.0, 10.0),
        ("Song2", 12.0, 10.0),
        ("Song3", 18.0, 10.0)
    ]

    all_passed = True
    for true_song, start_sec, dur_sec in test_cases:
        y_ref, s_rate = songs_db[true_song]
        query = extract_query_clip(y_ref, s_rate, start_sec=start_sec, duration_sec=dur_sec)
        
        # Test standalone function deliverable
        pred_song, score = identify_query_clip(query, songs_db, metric="cosine")
        _, _, leaderboard = recogniser.recognize(query, sr=s_rate)

        passed = (pred_song == true_song)
        if not passed:
            all_passed = False

        status = "PASSED (OK)" if passed else "FAILED"
        print(f"\n   Query: Sliced from [{true_song}] from {start_sec}s to {start_sec + dur_sec}s")
        print(f"   -> Result: Predicted '{pred_song}' with score {score:.4f} [{status}]")
        print("      Scores:")
        for rank, (cand_name, cand_score) in enumerate(leaderboard, 1):
            print(f"        {rank}. {cand_name}: {cand_score:.4f}")

    # 6. TEST SLIDING-WINDOW ESTIMATION
    print("\n[Step 5] Testing sliding-window time-offset estimation...")
    target_offset = 14.0
    query_sw = extract_query_clip(songs_db["Song2"][0], sr, start_sec=target_offset, duration_sec=10.0)
    sw_song, sw_score, est_offset, _ = sliding_window_recogniser(
        query_sw, songs_db, chunk_duration_sec=10.0, step_sec=2.0, sr=sr
    )
    print(f"   Ground Truth: Song2 at {target_offset}s")
    print(f"   -> Sliding Window Prediction: {sw_song} (Score: {sw_score:.4f}, Estimated Offset: {est_offset:.1f}s)")
    assert sw_song == "Song2", "Sliding window prediction failed!"

    print("\n" + "=" * 70)
    if all_passed:
        print("   >>> ALL UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY! <<<")
    else:
        print("   >>> SOME TESTS FAILED! <<<")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

