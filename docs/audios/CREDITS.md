# Third-party audio credits

## water_flow.wav

- **Source**: [WATRFlow_FlowingaroundRocks_dhi67540_ZH1e](https://freesound.org/people/dhi67540/sounds/826658/) by [dhi67540](https://freesound.org/people/dhi67540/) on Freesound
- **License**: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0) - attribution required, modification and commercial use permitted
- **Original description**: "Water flowing around rocks in a fast flowing creek. Recorded inches away from the water's surface." Recorded on a Zoom H1n at 96kHz/32-bit float, normalized in Reaper then rendered to 48kHz/24-bit.
- **Modification made here**: trimmed to the 5s-45s range (40s) of the original ~50.6s recording, downmixed to mono, re-encoded to 16-bit PCM @ 48kHz. No other processing applied.
- **Used by**: `docs/js/water-noise-generator.js` (the "Water flow" noise color), looped with a 15s equal-power crossfade - see that file's own header comment.
- **Spectral centroid**: 3597.9678237838875 Hz (magnitude-weighted, one FFT over the whole 40s clip - see `docs/js/spectral-centroid.js`'s own header for the exact definition), hardcoded as `WATER_NATIVE_CENTROID_HZ` in `water-noise-generator.js` and used as the reference the "Match source" feature (`stream.js`'s `processChunk`) pitch-shifts away from to track the live source's own centroid. Amplitude calibration doesn't change this value (a pure gain scale preserves every frequency's relative weight), so it was computed on the raw trimmed samples before calibration.

Reproduction: `ffmpeg -i <original file> -ss 5 -to 45 -ac 1 -ar 48000 -acodec pcm_s16le -map_metadata -1 water_flow.wav`, then the centroid itself (Python/numpy, `x`/`sr` from the resulting file):

```python
n2 = 1 << (len(x) - 1).bit_length()
mag = np.abs(np.fft.rfft(x, n=n2))
freqs = np.fft.rfftfreq(n2, d=1 / sr)
centroid = np.sum(freqs * mag) / np.sum(mag)
```
