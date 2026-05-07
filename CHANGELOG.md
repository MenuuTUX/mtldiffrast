# Changelog

All notable changes to MTLDiffRast will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Per-pixel barycentric coordinates exposed on `RasterOutput.barycentrics`.
- Real depth-output backward pass: gradients flow into clip-space `z` and `w`
  via an atomic compare-and-swap float accumulator (compatible with macOS 12).
- GPU implementations of `interpolate` and `antialias` (previously CPU stubs).
- Edge-aware antialiasing that preserves silhouettes instead of averaging
  triangle ids.

### Changed
- Metal kernels split into one file per kernel under `Sources/MTLDiffRast/Resources`:
  `ForwardRasterize.metal`, `BackwardRasterize.metal`, `Interpolate.metal`,
  `Antialias.metal`. Each file is self-contained (no shared header).
- `Package.swift` now uses `.copy("Resources")`. Kernels are loaded at runtime
  via `device.makeLibrary(source:)`, working around SPM's lack of a `.metal`
  build rule.
- All scalar kernel inputs are passed via packed `Int32` parameter structs and
  `encoder.setBytes`, fixing a 64/32-bit size mismatch with Swift's `Int`.

### Fixed
- `MTLDevice.isArgumentBufferSupported` reference replaced with current API.
- Demo `Shader.metal` now uses the per-vertex `depth` for the z component
  instead of hard-coding 0.
- Broken `.gitignore` (was wrapped in markdown code fences).

## [0.1.0] - Initial public release

- Forward rasterization
- Stub backward / interpolation / antialias paths
