# Third-party audio credits

## water_flow.wav

- **Source**: [WATRFlow_FlowingaroundRocks_dhi67540_ZH1e](https://freesound.org/people/dhi67540/sounds/826658/) by [dhi67540](https://freesound.org/people/dhi67540/) on Freesound
- **License**: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0) - attribution required, modification and commercial use permitted
- **Original description**: "Water flowing around rocks in a fast flowing creek. Recorded inches away from the water's surface." Recorded on a Zoom H1n at 96kHz/32-bit float, normalized in Reaper then rendered to 48kHz/24-bit.
- **Modification made here**: trimmed to the 5s-45s range (40s) of the original ~50.6s recording, downmixed to mono, re-encoded to 16-bit PCM @ 48kHz. No other processing applied.
- **Used by**: the "Water flow" noise color, registered in `docs/js/recorded-noise-colors.js` and played back by the shared `docs/js/recorded-noise-generator.js` (15s equal-power crossfade loop - see that file's own header comment).
- **Spectral centroid**: 3597.9678237838875 Hz (magnitude-weighted, one FFT over the whole 40s clip - see `docs/js/spectral-centroid.js`'s own header for the exact definition), stored as this color's `nativeCentroidHz` in `recorded-noise-colors.js` and used as the reference the "Match source" feature (`stream.js`'s `processChunk`) pitch-shifts away from to track the live source's own centroid. Amplitude calibration doesn't change this value (a pure gain scale preserves every frequency's relative weight), so it was computed on the raw trimmed samples before calibration.

Reproduction: `ffmpeg -i <original file> -ss 5 -to 45 -ac 1 -ar 48000 -acodec pcm_s16le -map_metadata -1 water_flow.wav`, then the centroid itself (Python/numpy, `x`/`sr` from the resulting file):

```python
n2 = 1 << (len(x) - 1).bit_length()
mag = np.abs(np.fft.rfft(x, n=n2))
freqs = np.fft.rfftfreq(n2, d=1 / sr)
centroid = np.sum(freqs * mag) / np.sum(mag)
```

## fx_water.wav

- **Source**: [WATRUndwtr_Underwater Bubble Flow Loop_KOLBYR_FREE SOUNDS](https://freesound.org/people/kolbyrfx/sounds/852478/) by [KolbyRFX](https://freesound.org/people/kolbyrfx/) on Freesound
- **License**: [Creative Commons 0](https://creativecommons.org/publicdomain/zero/1.0/) (CC0) - public domain dedication, no attribution legally required (credited here anyway)
- **Original description**: "UNDERWATER BUBBLE FLOW - Loop - Water Surface, Continuous Bubble Stream, Light Rising Bubbles, Soft Surface Interaction, Loopable." Originally 96kHz/24-bit stereo with an embedded cover-art video stream.
- **Modification made here**: video/art stream and metadata dropped, downmixed to mono, re-encoded to 16-bit PCM @ 48kHz. No trim (full ~98.66s original length kept) and no other processing applied.
- **Used by**: the "FX water" noise color, registered in `docs/js/recorded-noise-colors.js` and played back by the shared `docs/js/recorded-noise-generator.js` (15s equal-power crossfade loop).
- **Spectral centroid**: 857.8702035911797 Hz (same magnitude-weighted whole-clip FFT definition as water_flow.wav above), stored as this color's `nativeCentroidHz` in `recorded-noise-colors.js`, used by "Match source" the same way as water_flow.wav's own centroid.

Reproduction: `ffmpeg -i <original file> -vn -ac 1 -ar 48000 -acodec pcm_s16le -map_metadata -1 fx_water.wav`, then the same centroid snippet as above on the resulting file's samples.

## bubbles_cathedral.wav

- **Source**: synthesized in this project, not a recording - a physical model of bubble acoustics (Minnaert resonance: `f0 = 6520 / diameter_mm`, each bubble a damped sinusoid `A0*sin(2*pi*f(t)*t)*exp(-beta*t)`, per the Sound Design Toolkit's and Cornell's published bubble-synthesis work), scattered in time as clustered "glug" bursts. Generator: `tools/bubble-synth/synth_bubbles.mjs`; reverb convolution: `tools/bubble-synth/add_ir_reverb.mjs` - see that directory's own README.md for how to reproduce or tweak this file.
- **Embedded third-party component**: the reverb baked into this file is a real convolution against **"St Nicolaes Church.wav"**, an impulse response from Aleksey Vaneev's (Voxengo) free impulse-response pack, found locally in `/home/user/Documents/post-doc/Code/lofigen/samples/impulses/` (see that folder's own `license.txt`: "Permission is granted to anyone to use these impulse files royalty-free for any purpose, including commercial usage" - redistribution of the IR file itself is restricted, but using it to process other audio, as done here, is explicitly permitted).
- **Modification made here**: 50s rendered (dry bubbles convolved with the church IR, dry/wet mixed 0.55/0.45), trimmed to the 5s-45s range (40s), mono 16-bit PCM @ 48kHz - same convention as the other two files here.
- **Used by**: the "Bubbles" noise color, registered in `docs/js/recorded-noise-colors.js`.
- **Spectral centroid**: 1150.185107041403 Hz (same magnitude-weighted whole-clip FFT definition as above), stored as this color's `nativeCentroidHz` in `recorded-noise-colors.js`.
