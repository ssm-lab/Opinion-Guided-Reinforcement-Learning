import pytest
from src.binomial_opinion import BinomialOpinion


@pytest.fixture
def opinion1():
    return BinomialOpinion(0.5, 0.0, 0.5, 0.25)


@pytest.fixture
def opinion2():
    return BinomialOpinion(0.6, 0.2, 0.2, 0.25)


def test_valid_binomial_opinion(opinion1):
    assert opinion1


@pytest.mark.parametrize('b, d, u, a', [(1.0, 0.1, 0.1, 0.25), (0.1, 1.0, 0.1, 0.25), (0.1, 0.1, 1.0, 0.25)])
def test_additivity_requirement_satisfied(b, d, u, a):
    with pytest.raises(AssertionError):
        BinomialOpinion(b, d, u, a)


@pytest.mark.parametrize('b, d, u, a', [(-0.25, 0.70, 0.5, 0.25), (0.25, -0.70, 0.5, 0.25), (0.25, 0.70, -0.5, 0.25),
                                        (-0.25, 0.70, 0.5, -0.25)])
def test_param_domain_requirement_satisfied(b, d, u, a):
    with pytest.raises(AssertionError):
        BinomialOpinion(b, d, u, a)


def test_equality(opinion1):
    assert opinion1 == opinion1


def test_inequality(opinion1, opinion2):
    assert opinion1 != opinion2


def test_belief_constraint_fusion(opinion1, opinion2):
    fused_opinion = opinion1.belief_constraint_fusion(opinion2)

    assert fused_opinion.b == pytest.approx(0.7777, rel=0.001)
    assert fused_opinion.d == pytest.approx(0.1111, rel=0.001)
    assert fused_opinion.u == pytest.approx(0.1111, rel=0.001)
    assert fused_opinion.a == pytest.approx(0.25, rel=0.001)


def test_belief_constraint_fusion_vacuous():
    opinion1 = BinomialOpinion(0.4, 0.6, 0, 0.25)
    opinion2 = BinomialOpinion(0.8, 0.2, 0, 0.25)
    fused_opinion = opinion1.belief_constraint_fusion(opinion2)

    assert fused_opinion.b == pytest.approx(0.727, rel=0.001)
    assert fused_opinion.d == pytest.approx(0.273, rel=0.001)
    assert fused_opinion.u == pytest.approx(0.0, rel=0.001)
    assert fused_opinion.a == pytest.approx(0.25, rel=0.001)


def test_probability_to_opinion():
    probability = 1/4
    opinion = BinomialOpinion.probability_to_opinion(probability)
    assert opinion.tuple == (0.25, 0.75, 0.0, 0.25)


def test_opinion_to_probability(opinion1):
    assert opinion1.opinion_to_probability() == pytest.approx(0.625, rel=0.001)


if __name__ == "main":
    pytest.main()
