# bubble-synth

Offline Node tooling that generated `docs/audios/bubbles_cathedral.wav` (the
"Bubbles" noise color). Not part of the browser app - run manually when
tweaking or regenerating that file.

## What it does

1. **`synth_bubbles.mjs`** - physical-model bubble synthesis. Each "bubble"
   is a damped sinusoid at its own Minnaert resonance frequency
   (`f0 = 6520 / diameter_mm`), scattered in time as clustered "glug"
   bursts (see the file's own header comment for the full design
   rationale and the parameters that were tuned by ear along the way).
   Writes `out/bubbles_dry.wav`.
2. **`add_ir_reverb.mjs`** - convolves that dry signal with a real
   impulse response ("St Nicolaes Church.wav", from Aleksey Vaneev's/
   Voxengo's free impulse-response pack) for a cathedral-scale reverb,
   dry/wet mixed. Writes `out/bubbles_cathedral_full.wav`.
3. Trim the 5s-45s window into `docs/audios/`, matching this project's
   existing convention (see `docs/audios/CREDITS.md`'s other entries):
   ```
   ffmpeg -i out/bubbles_cathedral_full.wav -ss 5 -to 45 -ac 1 -ar 48000 \
     -acodec pcm_s16le -map_metadata -1 ../../docs/audios/bubbles_cathedral.wav
   ```
4. Compute its spectral centroid (see `docs/js/spectral-centroid.js`'s
   own header for the exact definition) and update that value in both
   `docs/js/recorded-noise-colors.js` and `docs/audios/CREDITS.md` - the
   centroid shifts slightly any time the dry synthesis or reverb mix
   changes, so it must be recomputed, not assumed.

## Running it

```
cd tools/bubble-synth
node synth_bubbles.mjs
node add_ir_reverb.mjs
```

`add_ir_reverb.mjs` expects the IR file at
`/home/user/Documents/post-doc/Code/lofigen/samples/impulses/St Nicolaes Church.wav`
by default (override with `IR_PATH=...`). **That IR file is intentionally
NOT committed to this repo** - its license (see that folder's own
`license.txt`) permits using it to process other audio "for any purpose,
including commercial usage," which is exactly what happened here (the
*output* of that convolution, `bubbles_cathedral.wav`, is fully fine to
commit), but redistributing the raw impulse file itself has separate
conditions (unaltered copies only, license notice retained, no charge)
that aren't worth taking on for this.

## Adding another color in this style (synthesized + baked-in effects)

1. Write/adapt a generator script here (or a new `tools/<name>/`
   directory if the synthesis approach is unrelated to bubbles).
2. Render, process (reverb/filtering/whatever), trim to a clean loop
   window, and save into `docs/audios/<name>.wav` - mono, 16-bit PCM,
   48kHz, matching every other file there.
3. Compute its spectral centroid the same way as above.
4. Add one entry to `docs/js/recorded-noise-colors.js`:
   ```js
   myColor: {
     label: 'My Color',
     audioPath: '../audios/<name>.wav',
     loopCrossfadeSeconds: 15,
     nativeCentroidHz: <computed value>,
   },
   ```
5. Add a section to `docs/audios/CREDITS.md` documenting the source
   (synthesized, or a real recording with its own license) and any
   embedded third-party components.

That's it - **no edits to `stream.js`, `worker-engine.js`, `main.js`, or
either HTML file are needed.** The dropdown option, the noise generator
instantiation, and the "Match source" centroid lookup are all driven by
that one registry file (see its own header comment for exactly how).
