<p align="center">
  <img src="https://raw.githubusercontent.com/Kr4t0n/vectorlab/main/static/logo.png"/>
</p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/vectorlab)](https://pypi.org/project/vectorlab/)
[![Package Status](https://img.shields.io/pypi/status/vectorlab)](https://pypi.org/project/vectorlab/)
[![Testing Status](https://img.shields.io/github/actions/workflow/status/Kr4t0n/vectorlab/testing.yml?label=Testing&logo=github)](https://github.com/Kr4t0n/vectorlab/actions/workflows/testing.yml)
[![Coverage](https://codecov.io/github/Kr4t0n/vectorlab/coverage.svg?branch=dev-feat)](https://codecov.io/gh/Kr4t0n/vectorlab)
[![License](https://img.shields.io/pypi/l/vectorlab)](https://github.com/Kr4t0n/vectorlab/blob/main/LICENSE.txt)
[![Python Version](https://img.shields.io/pypi/pyversions/vectorlab)](https://pypi.org/project/vectorlab/)

VectorLab is a library to help solve scientific and engineering problems of mathematics and machine learning. It is built upon various fabulous softwares and tries to fill the gap when playing around with those softwares.

VectorLab consists of many modules including data generation, graph processing, series processing, statistics, optimization, ensemble mechanism and more.

VectorLab is distributed under the 3-Clause BSD license.

## Installation

### Dependencies

VectorLab supports **Python 3.8+** only.

### PyPI Installation

The lastest version of VectorLab could be installed from PyPI.

```
pip install -U vectorlab
```

VectorLab also needs PyTorch and PyG support. It is highly
**recommended** to handle these two package installations **manually**. 
If you only care for the CPU version of PyTorch and PyG, you can
also install vectorlab from PyPI with extensions.

```
pip install -U -e vectorlab[.full]
```

## Development

We welcome contributions from all sources.

### Source code

You can check the latest commit with *git*:

```
git clone https://github.com/Kr4t0n/vectorlab.git
```

### Testing

After the regular installation, you will need *pytest* to test VectorLab. You can install corresponding dependencies with:

```
pip install -U -e vectorlab[.test]
```

You can launch the test suite with:

```
pytest vectorlab
```