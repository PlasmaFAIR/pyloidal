# pyloidal

Python utilities for tokamak science.

## Install

Install from PyPI:

```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install pyloidal
```

Install from GitHub repo:

```bash
$ git clone https://github.com/PlasmaFAIR/pyloidal
$ cd pyloidal
$ python3 -m pip install --upgrade pip
$ python3 -m pip install .
```

## Tests

First clone the repo, then:

```bash
$ python3 -m pip install .[test]
$ pytest --cov=pyloidal ./tests
```

## License

Copyrighted under the MIT License. Uses elements from [OMAS][OMAS], which is also
copyrighted under MIT, 2017 Orso Meneghini.

[OMAS]: https://github.com/gafusion/omas
