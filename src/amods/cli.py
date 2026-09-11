"""Command-line entry point: run the concealer on a mic stream or a wav file."""
import argparse

import numpy as np

from .stream import run_file_mode, run_stream_mode

np.random.seed(0)


def build_parser():
    """Build the argparse parser for the ``amods`` command."""
    parser = argparse.ArgumentParser(description="Run concealer on mic stream or on a wav file, using the SAME callback.")
    parser.add_argument("--infile", type=str, default=None, help="Input wav for file mode (e.g., LJ044-0130.wav). If set to None, stream mode is used directly from microphone.")
    parser.add_argument("--indir", type=str, default="./audios/", help="Input directory for file mode (e.g., ./audios/)")
    parser.add_argument("--outprefix", type=str, default="", help="Output prefix for file mode")
    parser.add_argument("--outdir", type=str, default="./output/", help="Output directory for file mode")
    parser.add_argument("--concealerconfig", type=str, default="default", help="Concealer config file name. Check folder configs/concealer/ to see available configs.")
    parser.add_argument("--streamconfig", type=str, default="default", help="Stream config file name. Check folder configs/stream/ to see available configs.")
    parser.add_argument("--sourcevadconfig", type=str, default="default_source", help="Source VAD config file name. Check folder configs/vad/ to see available configs.")
    parser.add_argument("--forecasterconfig", type=str, default="default", help="Forecaster config file name. Check folder configs/forecaster/ to see available configs.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
    return parser


def main(argv=None):
    """Entry point for the ``amods`` console script: file mode if ``--infile`` is set, else live stream mode."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.infile is not None:
        run_file_mode(
            indir=args.indir,
            infile=args.infile,
            outdir=args.outdir,
            outprefix=args.outprefix,
            concealerconfig=args.concealerconfig,
            streamconfig=args.streamconfig,
            sourcevadconfig=args.sourcevadconfig,
            forecasterconfig=args.forecasterconfig,
            debug=args.debug,
        )
    else:
        run_stream_mode(
            outdir=args.outdir,
            outprefix=args.outprefix,
            concealerconfig=args.concealerconfig,
            streamconfig=args.streamconfig,
            sourcevadconfig=args.sourcevadconfig,
            forecasterconfig=args.forecasterconfig,
        )


if __name__ == "__main__":
    main()
