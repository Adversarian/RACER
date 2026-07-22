# Changelog

## 1.3.0 - 2026-07-23

- Made preprocessing safe to fit per training fold, including unknown-category handling and boundary clipping for out-of-range numeric values.
- Added the optional paper-faithful `ig-paper` discretizer and `partial-match` prediction fallback while preserving existing defaults.
- Accelerated rule coverage with a semantics-preserving uint64 word-packed evaluator.
- Fixed zero-rule prediction and corrected the `_confusion` return-order documentation.
