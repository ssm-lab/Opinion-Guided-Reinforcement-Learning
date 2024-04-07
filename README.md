A framework for human-informed reinforcement learning by subjective logic

[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Unit tests](https://github.com/dagenaik/Uncertainty-in-Reinforcement-Learning/actions/workflows/ci.yaml/badge.svg)](https://github.com/dagenaik/Uncertainty-in-Reinforcement-Learning/actions/workflows/ci.yaml)


# Repository structure

- [/src](https://github.com/dagenaik/Uncertainty-in-Reinforcement-Learning/tree/main/src) - Source code
  - Main
    - `runner.py` - Main module
    - `model.py` - Model classes
  - Hint/SL modules
    - `parser.py` - Human hint parser. Parses hints from `files/opinions.txt`
    - `files/opinions.txt` - Human input. Format:
      ```
      grid size[1]
      uncertainty[1]
      hints[*]
      ```
    - `sl.py` - Subjective logic utilities
  - Map modules
    - `map_generator.py` - Generates map for human inspection. Saves maps under `/files` as `.xslx` files.
    - `map_parser.py` - Parses map for experiments
- [/tests](https://github.com/dagenaik/Uncertainty-in-Reinforcement-Learning/tree/main/tests) - Unit tests

# Setup guide
- Clone this repository.
- Install requirements via ```pip install -r requirements.txt```.

# How to use
- Set `SEED` and `SIZE` in `src/map_parser.py` and run `python .\src\map_parser.py`
- Create an opinion file with the following name: `opinions-[SIZE]x[SIZE]-seed[SEED].txt` (e.g., `opinions-6x6-seed10.txt`)
- Set the same `SEED` and `SIZE` in `src/runner.py` and run `python .\src\runner.py`
