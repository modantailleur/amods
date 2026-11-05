// Central registry of every "recorded" Noise color - a real audio clip
// looped via RecordedNoiseGenerator (see that file's own header), as
// opposed to white/pink/speechShaped's synthesized generators. Adding a
// new one of these is now just adding an entry here (plus the audio file
// itself in docs/audios/ and a docs/audios/CREDITS.md entry) - nothing in
// stream.js, worker-engine.js, main.js, or the HTML needs to change:
// stream.js's _createNoiseGen and "Match source" centroid lookup both key
// off this registry directly, and main.js injects one <option> per entry
// into the "Type" dropdown on init rather than requiring a hardcoded one
// in index.html/debug.html.
//
// nativeCentroidHz - see water-noise-generator's old own comment (now
// here): the clip's own magnitude-weighted spectral centroid (see
// spectral-centroid.js), computed ONCE offline (amplitude calibration
// doesn't change it - a pure gain scale preserves every frequency's
// relative weight) and hardcoded - see docs/audios/CREDITS.md for each
// clip's exact reproduction steps. Used by "Match source" (stream.js's
// processChunk) as the reference a color's playback rate is pitch-shifted
// away from to track the live source's own tracked centroid.
export const RECORDED_NOISE_COLORS = {
  water: {
    label: 'Water flow',
    audioPath: '../audios/water_flow.wav',
    loopCrossfadeSeconds: 15,
    nativeCentroidHz: 3597.9678237838875,
  },
  fxWater: {
    label: 'FX water',
    audioPath: '../audios/fx_water.wav',
    loopCrossfadeSeconds: 15,
    nativeCentroidHz: 857.8702035911797,
  },
  bubbles: {
    label: 'Bubbles',
    audioPath: '../audios/bubbles_cathedral.wav',
    loopCrossfadeSeconds: 15,
    nativeCentroidHz: 1150.185107041403,
  },
};
