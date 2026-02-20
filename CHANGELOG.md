# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-20
### Added
- README now renders on PyPI
- GitHub repository URL in package metadata
- License field in pyproject.toml

## [0.1.0] - 2026-02-20
### Added
- Initial release
- `fit_aic.scipy.curve_fit` wrapper with AIC and AICc added to infodict when `full_output=True`
- `fit_aic.lmfit.Model` subclass with `result.aicc` attribute
- Full type hints with `@overload` for correct return type inference
- Input validation for empty and mismatched arrays
- Test suite with pytest covering scipy and lmfit wrappers
