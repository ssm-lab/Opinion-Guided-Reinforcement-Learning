import pytest
import os
from src.advice_parser import AdviceParser


@pytest.fixture()
def advice_parser():
    yield AdviceParser()


def test_valid_parser_input(advice_parser):
    file = os.path.abspath("tests/valid_parser_input.txt")
    with open(file, 'r') as f:
        lines = len(f.readlines())
        expected_num_advice = lines - 1

        advisor_input = advice_parser.parse(file)

        assert advisor_input.map_size >= 0
        assert expected_num_advice == len(advisor_input.advice_list)


if __name__ == "main":
    pytest.main()
