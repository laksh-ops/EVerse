"""
================================================================================
SONIC SIGNATURES: FROM AUDIO SIGNALS TO SONG RECOGNITION
GROUP 2 - TASK 6: PEAK-PAIR MATCHING (SHAZAM-STYLE RECOGNISER)
================================================================================
"""

from task6_peak_pair_matching import (
    TARGET_SRATE,
    SAMPLE_COUNT,
    HOP_LENGTH,
    MAX_FREQ,
    MIN_FREQ,
    PEAK_NEIGHBORHOOD_FREQ,
    PEAK_NEIGHBORHOOD_TIME,
    AMP_THRESHOLD_DB,
    TARGET_ZONE_T_MIN,
    TARGET_ZONE_T_MAX,
    TARGET_ZONE_F_MAX,
    FAN_OUT,
    load_audio,
    comp_spectrogram,
    find_spectral_peaks,
    generate_peak_pair_hashes,
    ShazamPeakPairRecogniser,
    identify_query_clip_shazam,
    extract_query_clip,
    plot_offset_voting_histogram
)

if __name__ == "__main__":
    from test_task6 import run_tests
    run_tests()

