A framework for human-informed reinforcement learning by subjective logic

[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Unit tests](https://github.com/dagenaik/Uncertainty-in-Reinforcement-Learning/actions/workflows/ci.yaml/badge.svg)](https://github.com/dagenaik/Uncertainty-in-Reinforcement-Learning/actions/workflows/ci.yaml)


# Repository structure

- [/input](https://github.com/dagenaik/Uncertainty-in-Reinforcement-Learning/tree/main/input) - Input files: maps and human opinions
- [/src](https://github.com/dagenaik/Uncertainty-in-Reinforcement-Learning/tree/main/src) - Source code
  - Main
    - `runner.py` - Main module
    - `model.py` - Model classes
  - Opinion/SL modules
    - `opinion_parser.py` - Parses human input from `/input`. Input file naming convention: `opinions-[SIZE]x[SIZE]-seed[SEED].txt` Format:
      ```
      grid size [1]
      uncertainty [1]
      opinions [*]
      ```
    - `sl.py` - Subjective logic utilities
  - Map module
    - `map_tools.py` - Generator, renderer, and parser for maps. Saves maps under `/files` as `.xslx` files.
- [/tests](https://github.com/dagenaik/Uncertainty-in-Reinforcement-Learning/tree/main/tests) - Unit tests

# Setup guide
- Clone this repository.
- Install requirements via ```pip install -r requirements.txt```.

# How to use
- Generate a map by running `python .\src\map_tools.py -generate -render -size [SIZE] -seed [SEED]` -- Replace `[SIZE]` and `[SEED]` with the values (int) you need. The `-render` flag is optional.
- Create an opinion file with the following name: `opinions-[SIZE]x[SIZE]-seed[SEED].txt` (e.g., `opinions-6x6-seed10.txt`)
- Run the experiment using `python .\src\runner.py -log [DEBUG_LEVEL]` -- The `[DEBUG_LEVEL]` value is one of the following: `critical`, `error`, `warn`, `warning`, `info`, `debug`.
- Settings (size, seed, numexperiments, maxepisodes) can be set in `runner.__name__`.
