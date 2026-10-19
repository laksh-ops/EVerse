# ============================================================
# GROUP 2 - TASK 4: FINGERPRINT REPRESENTATION
# imported libraries include numpy and matplotlib
# This code assumes that Task 3 has already run
# This task is dedicated towards the fingerprint representation for each song. This is done so by taking extracted peaks from task 3 of GROUP1, 
# recording their frequencies,their time separation.
# This approach is further used by task 5 of GROUP2 to compare a query clip with the three songs provided as input in task 1 of GROUP1
# However, NO song matching is done in this task yet.Song matching is done later in the task 6 of GROUP2 using the "generate_fingerprints" function 
# defined below.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

def generate_fingerprints(peak_times, peak_freqs, fan_out=15, min_time_delta=0.0, max_time_delta=2.0):
    """
    Generate peak-pair fingerprints from a list of peak times and frequencies.
    
    Parameters
    ----------
    peak_times : NumPy array of peak times in seconds.
    peak_freqs : NumPy array of peak frequencies in Hz.
    fan_out : int, number of adjacent peaks to pair with each anchor peak.
    min_time_delta : float, minimum time separation for a valid pair.
    max_time_delta : float, maximum time separation for a valid pair.
    
    Returns
    -------
    fingerprints : list of dictionaries containing peak pair data.
    """
    # Sort peaks chronologically to easily define the forward "target zone"
    sort_idx = np.argsort(peak_times)
    t_sorted = peak_times[sort_idx]
    f_sorted = peak_freqs[sort_idx]

    fingerprints = []
    
    # Iterate over each peak to act as an "anchor" point
    for i in range(len(t_sorted)):
        anchor_t = t_sorted[i]
        anchor_f = f_sorted[i]
        
        # Look ahead at the next few peaks bounded by the fan_out limit
        for j in range(i + 1, min(i + 1 + fan_out, len(t_sorted))):
            target_t = t_sorted[j]
            target_f = f_sorted[j]
            
            delta_t = target_t - anchor_t
            
            # Record the pair if the time separation is within our desired window
            if min_time_delta <= delta_t <= max_time_delta:
                fingerprints.append({
                    'f1': anchor_f,
                    'f2': target_f,
                    'delta_t': delta_t,
                    't1': anchor_t, # Saved for visualization purposes
                    't2': target_t  # Saved for visualization purposes
                })
                
    return fingerprints

# ------------------------------------------------------------
# PART A: GENERATE FINGERPRINTS FOR ALL SONGS
# ------------------------------------------------------------
fingerprint_results = {}

for name in songs.keys():
    # Retrieve the peak coordinates detected in Task 3
    ptimes, pfreqs = peak_results[name]
    
    # Generate the peak pairs
    fps = generate_fingerprints(ptimes, pfreqs, fan_out=15)
    fingerprint_results[name] = fps
    
    print(f"{name}: {len(fps)} peak-pair fingerprints generated")

# ------------------------------------------------------------
# PART B: VISUALISE THE FINGERPRINTS
# ------------------------------------------------------------
# We'll plot the fingerprints for the first song. 
# We restrict the plot to a subset of pairs so the graph isn't too cluttered.
sample_song = list(songs.keys())[0]
sample_fps = fingerprint_results[sample_song]

fig, ax = plt.subplots(figsize=(14, 6))

# Plot the original peaks
ptimes, pfreqs = peak_results[sample_song]
ax.scatter(ptimes, pfreqs, color='blue', marker='o', s=15, label="Spectral Peaks")

# Draw lines connecting the anchor and target peaks (our fingerprints)
# Plotting just a 300-pair subset to keep the visualization clear
subset_fps = sample_fps[:300] 

for idx, fp in enumerate(subset_fps):
    # Only add the label to the first line to avoid legend duplication
    label = "Peak Pair (Fingerprint)" if idx == 0 else None
    ax.plot([fp['t1'], fp['t2']], [fp['f1'], fp['f2']], 
            color='red', alpha=0.4, linewidth=0.8, label=label)

ax.set_title(f"{sample_song}: Fingerprint Representation (Subset)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_ylim(0, 5000)
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
