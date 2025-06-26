# Yale/UNC-CH - Geophysical Waveform Inversion
### Overview
In this competition, you’ll estimate subsurface properties—like velocity maps—from on seismic waveform data. Known as Full Waveform Inversion (FWI), this process could lead to more accurate and efficient seismic analysis, making it more useful in a variety of fields.

### Description
Imagine a doctor analyzing an ultrasound scan—not just to see a blurry outline but to achieve a clearer, more detailed image crucial for an accurate diagnosis. That's the challenge geophysicists face when imaging the Earth's hidden structures. Beneath the surface lie vital resources, potential hazards, and clues to our planet's history—all requiring sharper, more precise subsurface imaging to be fully understood and effectively utilized.

Full Waveform Inversion (FWI) is the key to unlocking these secrets. This powerful technique, crucial for energy exploration, carbon storage, medical ultrasound, and advanced material testing, aims to build a detailed picture of the subsurface by analyzing the entire shape of seismic waves. But current methods are hindered by a noisy reality.

Traditional physics-based approaches are accurate, but incredibly slow and prone to errors when the signal is weak from noisy data. Pure machine learning solutions are faster, but require vast amounts of labeled data and often fail to generalize to new, unfamiliar "signal."

This competition challenges you to bridge the gap by combining physics and machine learning to advance FWI. Success here could transform not only subsurface energy exploration but also a wide range of applications, from medical diagnostics to non-destructive material testing—anywhere precise imaging matters.

### Evaluation
Submissions are evaluated on the mean absolute error (MAE) across all columns and rows. Only the odd position values of x_ (columns) should be submitted, as shown below and in the sample_submission.csv file. All y_ positions (rows) must be predicted, with the predictions of each oid stacked onto each other.
