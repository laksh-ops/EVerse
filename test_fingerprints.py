import unittest
import numpy as np

# Replace 'fingerprint_generator' with the actual name of your Python file
from fingerprint_generator import generate_fingerprints

class TestGenerateFingerprints(unittest.TestCase):

    def setUp(self):
        # Basic sorted mock data
        self.peak_times = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        self.peak_freqs = np.array([100, 200, 150, 300, 250])

    def test_basic_fingerprint_generation(self):
        """Test that fingerprints are generated correctly with default generous bounds."""
        # fan_out=15 easily covers all 5 peaks
        fps = generate_fingerprints(self.peak_times, self.peak_freqs, fan_out=15, min_time_delta=0.0, max_time_delta=5.0)
        
        # Total combinations for 5 items = 4 + 3 + 2 + 1 = 10 pairs
        self.assertEqual(len(fps), 10)
        
        # Verify the first pair (Anchor: 1.0s, Target: 1.5s)
        self.assertEqual(fps[0]['t1'], 1.0)
        self.assertEqual(fps[0]['t2'], 1.5)
        self.assertEqual(fps[0]['f1'], 100)
        self.assertEqual(fps[0]['f2'], 200)
        self.assertAlmostEqual(fps[0]['delta_t'], 0.5)

    def test_fan_out_limit(self):
        """Test that the fan_out parameter restricts the number of forward pairs."""
        # Limit fan_out to 2. 
        # Peak 1 pairs with 2, 3. Peak 2 pairs with 3, 4. Peak 3 pairs with 4, 5.
        # Peak 4 pairs with 5. Peak 5 pairs with nothing.
        # Total pairs = 2 + 2 + 2 + 1 = 7 pairs
        fps = generate_fingerprints(self.peak_times, self.peak_freqs, fan_out=2, min_time_delta=0.0, max_time_delta=5.0)
        
        self.assertEqual(len(fps), 7)
        
        # Check that anchor at 1.0s only pairs with 1.5s and 2.0s
        anchor_1_pairs = [fp for fp in fps if fp['t1'] == 1.0]
        self.assertEqual(len(anchor_1_pairs), 2)
        self.assertEqual(anchor_1_pairs[-1]['t2'], 2.0) # Farthest target is 2.0s

    def test_time_delta_filtering(self):
        """Test that min_time_delta and max_time_delta correctly filter out pairs."""
        # Only allow pairs that are between 0.8s and 1.2s apart
        # For anchor 1.0: target 2.0 (delta 1.0) is valid.
        # For anchor 1.5: target 2.5 (delta 1.0) is valid.
        # For anchor 2.0: target 3.0 (delta 1.0) is valid.
        fps = generate_fingerprints(self.peak_times, self.peak_freqs, fan_out=15, min_time_delta=0.8, max_time_delta=1.2)
        
        self.assertEqual(len(fps), 3)
        for fp in fps:
            self.assertTrue(0.8 <= fp['delta_t'] <= 1.2)

    def test_unsorted_input_handling(self):
        """Test that the function correctly sorts chronological inputs before processing."""
        unsorted_times = np.array([2.0, 1.0, 3.0])
        unsorted_freqs = np.array([200, 100, 300])
        
        fps = generate_fingerprints(unsorted_times, unsorted_freqs, fan_out=15)
        
        # The first pair should be the chronologically first anchor (1.0) to the next (2.0)
        self.assertEqual(fps[0]['t1'], 1.0)
        self.assertEqual(fps[0]['t2'], 2.0)

    def test_empty_input(self):
        """Test that the function handles empty arrays gracefully."""
        empty_times = np.array([])
        empty_freqs = np.array([])
        
        fps = generate_fingerprints(empty_times, empty_freqs)
        self.assertEqual(len(fps), 0)
        self.assertIsInstance(fps, list)

if __name__ == '__main__':
    unittest.main()
