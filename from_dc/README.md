# Optional local mirror

The directory `gaussian-splatting-dc/` here is **not** part of the public `pinhole_calib` repository. Keep a private clone or symlink at:

`from_dc/gaussian-splatting-dc`

so that it contains a top-level `dc_reality/` package, **or** install that repo elsewhere and either:

- `pip install -e /path/to/gaussian-splatting-dc`, or
- `export PINHOLE_CALIB_GAUSSIAN_SPLATTING_DC=/path/to/gaussian-splatting-dc`

See the root [README.md](../README.md), [docs/from_dc_integration.md](../docs/from_dc_integration.md), and [guide.md](guide.md) (absolute-path notes for local mirrors).
