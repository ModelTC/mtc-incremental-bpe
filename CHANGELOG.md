# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0](https://github.com/ModelTC/mtc-incremental-bpe/compare/v0.6.0...v0.7.0) - 2025-12-09

### Added

- [**breaking**] add eager tokenization

### Fixed

- remove `EagerBpeToken`, `feed_len` is useless for external users

## [0.6.0](https://github.com/ModelTC/mtc-incremental-bpe/compare/v0.5.0...v0.6.0) - 2025-12-08

### Added

- expose byte and char lookup operations
- support empty tokens
- add built-in byte and char lookup tables to `Vocab`
- heavy light decomposition

### Fixed

- use rapidhash instead of default hash
- move byte to token id table to heap

### Other

- make use of type inference
- remove `two_diff_mut` from `TypedVec`
- remove authors field in Cargo.toml
- *(tests)* move heap bpe into test utils
- *(deps)* update `derive_more`
- keep transition table in the order of heavy chains
- *(aho_corasik)* use sqrt decomposition to reduce memory footprint

## [0.5.0](https://github.com/ModelTC/mtc-incremental-bpe/compare/v0.4.1...v0.5.0) - 2025-11-28

### Added

- [**breaking**] rename functions and expose more for `IncBpeTokenization`

### Fixed

- use the roots of the subtrees as indicators of parents, fix #19
- expose priority of a token in normalized dict, fix panic when token id exceeded vocab size
- use `LinkedList` for suffix chain
- check token length explicitly
- add `non_exhaustive` to errors
- use u16 as `skip_len`

### Other

- use `tinyvec` replacing `smallvec` to reduce memory footprint
- *(tests)* add tests on repeated characters
- unify integer literals

## [0.4.1](https://github.com/ModelTC/mtc-incremental-bpe/compare/v0.4.0...v0.4.1) - 2025-11-27

### Other

- optimize constructors in debug mode ([#17](https://github.com/ModelTC/mtc-incremental-bpe/pull/17))

## [0.4.0](https://github.com/ModelTC/mtc-incremental-bpe/compare/v0.3.1...v0.4.0) - 2025-11-20

### Added

- [**breaking**] expose position in `IncBpeTokenChainIter`

## [0.3.1](https://github.com/ModelTC/mtc-incremental-bpe/compare/v0.3.0...v0.3.1) - 2025-11-19

### Fixed

- expose `NormalizedDictBuildError`

## [0.3.0](https://github.com/ModelTC/mtc-incremental-bpe/compare/v0.2.1...v0.3.0) - 2025-11-19

### Added

- [**breaking**] make `NormalizedDict::new` return `Result`, adjust several interfaces

### Other

- rename parameters for clarity

## [0.2.1](https://github.com/ModelTC/mtc-incremental-bpe/compare/v0.2.0...v0.2.1) - 2025-11-18

### Added

- expose IncBpeTokenChainIter

## [0.2.0](https://github.com/ModelTC/mtc-incremental-bpe/compare/v0.1.0...v0.2.0) - 2025-11-18

### Added

- [**breaking**] fetch token chain using iterator instead of vec for performance
- [**breaking**] expose more context when checking if a token is single

### Other

- clean up code
- optimize validation to reduce execution time
- pre-allocate vector whenever possible
- reorder functions

## [0.1.0](https://github.com/ModelTC/mtc-incremental-bpe/releases/tag/v0.1.0) - 2025-11-17

### Added

- init

### Other

- add more events to trigger build and test ([#2](https://github.com/ModelTC/mtc-incremental-bpe/pull/2))
