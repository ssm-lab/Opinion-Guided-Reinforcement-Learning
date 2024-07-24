# Binomial opinion module
import math


class BinomialOpinion:
    """A binomial opinion about an object, with belief, disbelief, uncertainty, and base rate"""

    def __init__(self, b: float, d: float, u: float, a: float):
        self.b, self.d, self.u, self.a = b, d, u, a
        self.tuple = (self.b, self.d, self.u, self.a)
        assert math.isclose((self.b + self.d + self.u), 1.0, rel_tol=0.001)
        assert (0.0 <= (self.b and self.d and self.u and self.a) <= 1.0)

    def __eq__(self, other):
        return self.b == other.b and self.d == other.d and self.u == other.u and self.a == other.a

    def __str__(self):
        return f'Opinion (b = {self.b}, d = {self.d}, u = {self.u}, a = {self.a}).'

    def belief_constraint_fusion(self, other):
        [b1, d1, u1, a1] = self.tuple
        [b2, d2, u2, a2] = other.tuple

        harmony = b1 * b2 + b1 * u2 + b2 * u1
        conflict = b1 * d2 + b2 * d1
        b = harmony / (1 - conflict)
        u = u1 * u2 / (1 - conflict)
        d = 1 - (b + u)
        a = (a1 * (1 - u1) + a2 * (1 - u2)) / (2 - u1 - u2)

        fused_opinion = BinomialOpinion(b=b, d=d, u=u, a=a)

        return fused_opinion

    @staticmethod
    def probability_to_opinion(probability: float):
        return BinomialOpinion(probability, 1 - probability, 0.0, probability)

    def opinion_to_probability(self):
        return self.b + self.u * self.a
