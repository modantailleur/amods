"""
Standalone tool to:
  1. Verify which mic/speaker device actually works on this computer (play a
     test tone through the chosen output, record+play back through the
     chosen input), since device indices are unstable across reconnects/docks
     and generic names like "pulse"/"default" don't tell you what's really
     behind them.
  2. Report sounddevice/PortAudio's self-reported I/O latency for that pair
     (the same value/method the GUI displays) - this is the driver's own
     estimate, not an independently measured round-trip time.

Usage:
    amods-latency                                # lists devices
    amods-latency --device-in 15 --device-out 11
"""
import argparse

import numpy as np


def list_devices():
    """Print every audio device PortAudio knows about, with its index."""
    import sounddevice as sd
    print(sd.query_devices())


def test_output(device_out, sr=48000, duration=1.5, freq=440.0):
    """Play a short fading test tone through ``device_out`` and ask the user to confirm they heard it."""
    import sounddevice as sd

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade_n = int(sr * 0.05)
    fade = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    tone[:fade_n] *= fade
    tone[-fade_n:] *= fade[::-1]

    print(f"\nPlaying a {freq:.0f} Hz test tone through device {device_out}...")
    sd.play(tone, samplerate=sr, device=device_out)
    sd.wait()
    input("Did you hear it? Press Enter to continue (Ctrl+C to abort otherwise)...")


def test_input(device_in, device_out_for_playback, sr=48000, duration=3.0):
    """Record a few seconds from ``device_in``, then play it back so the user can confirm it captured their voice."""
    import sounddevice as sd

    print(f"\nRecording {duration:.0f}s from device {device_in} - speak now!")
    recording = sd.rec(int(sr * duration), samplerate=sr, channels=1, device=device_in, dtype="float32")
    sd.wait()

    rms = float(np.sqrt(np.mean(recording ** 2)))
    peak = float(np.max(np.abs(recording)))
    print(f"Recorded. rms={rms:.4f} peak={peak:.4f}", end="")
    if peak < 0.01:
        print("  <- looks silent/near-silent, this mic may not be capturing anything")
    else:
        print()

    print(f"Playing it back through device {device_out_for_playback}...")
    sd.play(recording, samplerate=sr, device=device_out_for_playback)
    sd.wait()
    input("Did you hear your own voice played back? Press Enter to continue...")


def report_self_reported_latency(device_in, device_out, sr=48000):
    """Open a duplex stream on ``device_in``/``device_out`` and print PortAudio's own reported I/O latency for it."""
    import sounddevice as sd

    stream = sd.Stream(
        samplerate=sr, channels=1, device=(device_in, device_out),
        dtype="float32", latency="low",
    )
    stream.start()
    input_latency, output_latency = stream.latency
    stream.stop()
    stream.close()

    print(
        f"\nSelf-reported I/O latency (PortAudio's own estimate, not independently "
        f"measured):\n"
        f"  input:  {input_latency * 1000:.1f} ms\n"
        f"  output: {output_latency * 1000:.1f} ms\n"
        f"  total:  {(input_latency + output_latency) * 1000:.1f} ms"
    )


def build_parser():
    """Build the argparse parser for the ``amods-latency`` command."""
    parser = argparse.ArgumentParser(
        description="Verify mic/speaker devices actually work, then report PortAudio's self-reported I/O latency for them."
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--device-in", type=int, default=None, help="Input (mic) device index")
    parser.add_argument("--device-out", type=int, default=None, help="Output (speaker) device index")
    parser.add_argument("--sr", type=int, default=48000, help="Sample rate (default: 48000)")
    return parser


def main(argv=None):
    """Entry point for the ``amods-latency`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        list_devices()
        return

    if args.device_in is None or args.device_out is None:
        print(
            "Pass --device-in and --device-out to test a mic/speaker pair "
            "(run --list-devices to see available indices), e.g.:\n"
            "  amods-latency --list-devices\n"
            "  amods-latency --device-in 15 --device-out 11\n"
        )
        return

    test_output(args.device_out, sr=args.sr)
    test_input(args.device_in, args.device_out, sr=args.sr)

    if args.device_in == args.device_out:
        print(
            "\ndevice-in and device-out are the same device: skipping the combined "
            "latency check. Opening one device for simultaneous input+output has "
            "been observed to crash on some systems."
        )
        return

    report_self_reported_latency(args.device_in, args.device_out, sr=args.sr)


if __name__ == "__main__":
    main()
