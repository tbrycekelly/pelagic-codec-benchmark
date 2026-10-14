# Pelagic Codec Benchmark

This is a small, reproducible benchmark foundation for measuring image-codec effects on pelagic-imaging pixels and threshold segmentations. It supports canonical **2D grayscale `uint8`** images and unambiguous grayscale video frames, PNG/DEFLATE; TIFF uncompressed, Deflate, and Zstd; libjpeg-turbo JPEG; OpenJPEG JPEG 2000; libjxl JPEG XL; WebP; AVIF; and CharLS JPEG-LS. Each treatment always encodes the same canonical source array; it never chains codecs.

## Install and run

This project uses Python 3.12 and `uv`. Pelagia is a required dependency and currently expects the sibling Pelagia and Oracle Builder checkouts used by this workspace.

```bash
uv sync
uv run pelagic-bench validate configs/quick-test.toml
uv run pelagic-bench inspect
uv run pelagic-bench run configs/quick-test.toml
uv run pelagic-bench verify runs/<run-id>
uv run pelagic-bench summarize runs/<run-id>
```

`uv sync` must be run in the paired local Pelagia/Oracle Builder workspace. Pelagia is not yet independently installable because its current package metadata references a sibling Oracle Builder source package. The OpenCV fallback remains only as a clear runtime failure-mode guard, not as a supported standard installation.

`configs/quick-test.toml` uses two committed PGM synthetic images and runs PNG, JPEG q95, and JPEG q75. `configs/uaf-test1.toml` is a full native-codec matrix: PNG/DEFLATE, TIFF (uncompressed/Deflate/Zstd), JPEG qualities, JPEG 2000 lossless/lossy layers, JPEG XL lossless/distances, WebP lossless/lossy qualities, AVIF levels, and JPEG-LS lossless/near-lossless.

`configs/uaf-test1.toml` is the full UAF codec configuration. It is currently set to a deterministic zero-based frame-index subset; set `frame_selection = "all"` to process every sequentially decodable frame of every AVI in `data/uaf-test1`, or retain/provide `frame_indices` for a selected subset.

Set `dataset.invert_grayscale = true` to make `255 - pixel` the reference frame after each preprocessing condition. It is therefore applied after the raw identity step or after Pelagia flatfielding, and before codecs, pixel/segmentation metrics, and ROI comparisons. Source-frame hashes remain hashes of decoded source pixels; each inverted preprocessing reference receives its own hash. The order is recorded in the immutable configuration and manifest; set it to `false` (the default) to retain source polarity.

`[benchmark]` can enable both independent measurements. `in_memory_enabled` measures codec encode/decode on arrays and bytes only. `roundtrip_enabled` writes each encoded treatment to the configured `roundtrip_directory`, reopens it, and decodes it; it records separate `write`, `read`, and `decode` observations. The configured directory identifies the filesystem under test. Encoded scratch files are deleted immediately after each observation, and filesystem-cache state is therefore naturally warm-after-write rather than a controlled cold-cache result.

## Scientific model

An input file is decoded once to canonical `uint8` pixels and gets both a source SHA-256 and a domain-separated pixel SHA-256 (dtype and shape included). Every codec treatment encodes those in-memory canonical pixels. Their decoded pixels are compared to the reference, then reference and treatment images are independently segmented. The reference is represented explicitly as `codec=reference, setting_id=reference` in segmentation results.

Each `[[preprocessing]]` entry defines an independent pipeline condition. A condition can leave canonical source pixels unchanged or use Pelagia's stored-profile flatfield correction with an explicit row/column-mean profile, axis, window stride, window width, and field bounds. It creates its own processed-reference hash, and its ID keys every treatment and analysis row. This permits a storage/performance-only original path and a flatfield-plus-segmentation path in the same immutable run without conflating their references.

Fixed segmentation uses Pelagia-compatible bright-foreground semantics, `pixel > threshold`. Otsu computes an independent threshold for each image and records the reference threshold, treatment threshold, and delta. The requested threshold-crossing metric is stored using `(reference < T) != (decoded < T)` and the distinction is declared in `manifest.json`.

Pelagia’s pure threshold primitives are used by the standard installation. Component areas are always calculated from the benchmark masks—not Pelagia's current ROI `area` field, which is bounding-box area. The benchmark uses its own full-frame overlap matcher because Pelagia's available matchers target cross-frame processing rather than reference-vs-codec comparison.

## Output

Each execution creates a new immutable directory such as:

```text
runs/2026-09-28T220417Z_a81f37c/
  config.toml  manifest.json
  images.parquet  codec_results.parquet  benchmark_results.parquet
  pixel_results.parquet  segmentation_results.parquet
  roi_matches.parquet  errors.parquet
```

`images` identifies source/canonical pixels. `codec_results` holds compressed size and treatment settings. `benchmark_results` contains individual in-memory encode/decode timings (warmups excluded). `pixel_results` contains MAE, RMSE, maximum error, PSNR, and exact equality. `segmentation_results` contains threshold and mask statistics; `roi_matches` records overlap edges and 1:1/split/merge/lost/new classifications with IoU and basic geometry. `errors` records nonfatal image/treatment failures.

`manifest.json` records run/config hashes, source table hashes and counts, package versions, Git state where available, machine details, codec implementation details, and the storage/segmentation policies. `verify` checks its table checksums, schemas, and counts.

When enabled, `system_benchmark_results.parquet` records context rather than codec results: configurable 1/16/64 MiB disk writes, durability requests (`fsync`), and reads, plus NumPy `float32` matrix-multiply and 2D real-FFT timings. Disk reads are declared warm-after-write; the temporary files are removed immediately. These measurements help interpret a run but do not replace a controlled cold-cache storage study.

## Storage boundary

Codec timing is strictly array/bytes in memory, so disk or Pelagia storage latency cannot contaminate measurements. `retain_artifacts = true` stores debug copies below `artifacts/`; it is off by default. Pelagia storage is not used in this slice. A future storage integration must name and record a fixed storage codec/profile separately from benchmark treatments, including its parameters and verification hashes.

## Current limits and next steps

No resume, color/high-bit-depth images, SSIM, boundary metrics, disk benchmarks, or richer morphometrics are included. The benchmark itself is grayscale-only. WebP requires an explicit RGB repetition adapter; AVIF uses the same documented adapter so its quality-level treatments follow one consistent path. The first decoded RGB channel is converted back to gray for all metrics, and that adapter is recorded in each treatment's settings. Natural next additions are color/high-bit-depth support, a versioned Pelagia processing-only package, fixed-reference Otsu mode, richer component measurements, and downstream classifier stability.
