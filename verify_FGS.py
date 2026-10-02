"""
Exact verifier for the finite part of
"The Furedi-Gyarfas-Simonyi Conjecture for t <= 99".

The proof itself is in FGS_t99_simple.tex.
All assertions below use integers or fractions.Fraction.
No floating-point number is used to decide an inequality.

By default the script performs the complete verification, including the
pairing lemma for every 23 <= t <= 99.  That is the slow part.
Use --fast to skip that finite lemma check and verify only the final certificate.
"""

from fractions import Fraction
from math import factorial, comb
from functools import lru_cache
import argparse


# ---------------------------------------------------------------------------
# Basic exact matching counts
# ---------------------------------------------------------------------------

@lru_cache(None)
def fact(n):
    return factorial(n)


@lru_cache(None)
def odd_df(q):
    if q <= 0:
        return 1
    z = 1
    for x in range(q, 0, -2):
        z *= x
    return z


@lru_cache(None)
def perfect_count(a, b, j):
    """Perfect matchings with exactly j marked-marked pairs."""
    x = a - 2*j
    if x < 0 or x > b:
        return 0
    r = b - x
    if r < 0 or r % 2:
        return 0
    return (
        fact(a) // (fact(x) * 2**j * fact(j))
        * fact(b) // fact(r)
        * odd_df(r - 1)
    )


@lru_cache(None)
def z_distribution(N, c):
    """Distribution of Z_{N,c} for odd N."""
    den = odd_df(N)
    out = []
    for j in range(c//2 + 1):
        cnt = (
            c * perfect_count(c-1, N-c, j)
            + (N-c) * perfect_count(c, N-c-1, j)
        )
        out.append(Fraction(cnt, den) if cnt else Fraction(0))
    assert sum(out, Fraction(0)) == 1
    return tuple(out)


def shearer_values(D):
    s = [Fraction(1)]
    for d in range(1, D+1):
        s.append(
            (Fraction(1) + d*(d-1)*s[-1]) / (d*d + 1)
        )
    return s


S = shearer_values(100)


def g_value(t, c):
    N = 4*t - 3
    return sum(p*S[j] for j, p in enumerate(z_distribution(N, c)))


@lru_cache(None)
def matching_count(a, i):
    if 2*i > a:
        return 0
    return fact(a) // (fact(a-2*i) * 2**i * fact(i))


@lru_cache(None)
def H_value(N, b):
    return sum(
        p * (Fraction(1, 2) - S[j+1])
        for j, p in enumerate(z_distribution(N, b))
    )


def Phi(t, c, d):
    """The oriented full-leaf kernel Phi_t(c,d)."""
    R = 4*t - 5
    a = c - 2
    b = d - 2
    den = odd_df(R)
    total = Fraction(0)
    for i in range(a//2 + 1):
        coeff = Fraction(matching_count(a, i) * odd_df(R-2*i), den)
        term = coeff * H_value(R-2*i, b)
        total += -term if i % 2 else term
    return total


# ---------------------------------------------------------------------------
# Pairing lemma: exact coefficient count
# ---------------------------------------------------------------------------

@lru_cache(None)
def paired_coeff(m, i, x, y):
    """Coefficient of u^i x^x y^y in (1+2u+2x+2y+2xy)^m."""
    if min(i, x, y) < 0:
        return 0
    lo = max(0, i+x+y-m)
    hi = min(x, y)
    if lo > hi:
        return 0
    fm = fact(m)
    ans = 0
    for k in range(lo, hi+1):
        rest = m-i-x-y+k
        ans += fm * (1 << (i+x+y-k)) // (
            fact(i) * fact(x-k) * fact(y-k) * fact(k) * fact(rest)
        )
    return ans


@lru_cache(None)
def good_count(R, a, b, z):
    """Allowed category placements for two sets of sizes a,b and overlap z."""
    m = (R-1)//2
    i = z
    x = a-z
    y = b-z
    if min(i, x, y) < 0 or a+b-z > R:
        return 0
    # Singleton is O, I, X, or Y.
    return (
        paired_coeff(m, i,   x,   y)
        + paired_coeff(m, i-1, x,   y)
        + paired_coeff(m, i,   x-1, y)
        + paired_coeff(m, i,   x,   y-1)
    )


@lru_cache(None)
def total_categories(R, a, b, z):
    i = z
    x = a-z
    y = b-z
    o = R-(a+b-z)
    if min(i, x, y, o) < 0:
        return 0
    return fact(R) // (fact(i)*fact(x)*fact(y)*fact(o))


@lru_cache(None)
def single_good(R, a):
    """Allowed category placements for one a-set."""
    m = (R-1)//2
    ans = 0
    # Singleton outside the set.
    if a <= m:
        ans += comb(m, a) * (1 << a)
    # Singleton inside the set.
    if a >= 1 and a-1 <= m:
        ans += comb(m, a-1) * (1 << (a-1))
    return ans


def check_pairing_t(t):
    """Check both inequalities in the pairing lemma for one t."""
    R = 4*t - 7
    max_a = t - 8
    checked = 0

    for a in range(2, max_a+1):
        sa = single_good(R, a)
        ca = comb(R, a)
        for b in range(a, max_a+1):
            sb = single_good(R, b)
            cb = comb(R, b)
            for z in range(a+1):
                checked += 1
                G = good_count(R, a, b, z)
                M = total_categories(R, a, b, z)

                # Q_R(a,b,z) >= (1/2) q_R(a) q_R(b).
                assert 2*G*ca*cb >= M*sa*sb, ("pair-lower", t, a, b, z)

                # Q_{R-2}(a,b,z) <= Q_R(a,b,z).
                G2 = good_count(R-2, a, b, z)
                M2 = total_categories(R-2, a, b, z)
                assert G2*M <= G*M2, ("pair-monotone", t, a, b, z)

    return t, checked


# ---------------------------------------------------------------------------
# Uniform one-variable certificate
# ---------------------------------------------------------------------------

def certify_t(t):
    """Verify the compressed one-variable certificate used in the paper."""
    N = 4*t - 3
    R = 4*t - 7
    cs = list(range(4, t-5))      # 4,...,t-6

    def omega(j):
        return Fraction(2) * S[j+2] / (j+2)

    # The interpolation used below is valid because omega is decreasing and convex.
    for j in range(44):
        assert omega(j) >= omega(j+1), ("omega-monotone", t, j)
    for j in range(1, 44):
        assert omega(j-1) - 2*omega(j) + omega(j+1) >= 0, (
            "omega-convex", t, j
        )

    q = {}
    theta = {}
    D = {}
    a = {}

    for c in cs:
        q[c] = Fraction(single_good(R, c-2), comb(R, c-2))
        assert q[c] == z_distribution(R, c-2)[0]

        D[c] = comb(c, 2)
        a[c] = D[c] * q[c]

        lam = Fraction(comb(c-4, 2), R)
        k = lam.numerator // lam.denominator
        theta[c] = (
            (Fraction(k+1) - lam) * omega(k)
            + (lam-k) * omega(k+1)
        )

    gamma = Fraction(1, 4*N*(N-2))

    def Rcd(c, d):
        K = Phi(t, c, d) + Phi(t, d, c)
        return (
            K / N
            + gamma * (
                theta[c] * (2*a[c]*q[d] - (2*c-3)*q[d]*q[d])
                + theta[d] * (2*a[d]*q[c] - (2*d-3)*q[c]*q[c])
            )
        )

    # The edge kernel is bounded below by the average of its diagonal values.
    r = {c: Rcd(c, c) / 2 for c in cs}
    min_kernel_gap = None
    min_pair = None
    for ii, c in enumerate(cs):
        for d in cs[ii:]:
            gap = Rcd(c, d) - r[c] - r[d]
            assert gap >= 0, ("kernel", t, c, d, gap)
            if min_kernel_gap is None or gap < min_kernel_gap:
                min_kernel_gap = gap
                min_pair = (c, d)

    xi = {
        c: g_value(t, c)
           - gamma * theta[c] * a[c] * a[c]
           + D[c] * r[c]
        for c in cs
    }

    # Best affine lower bound through c=4 is obtained by the smallest secant slope.
    slopes = {
        c: (xi[c] - xi[4]) / (c-4)
        for c in cs if c > 4
    }
    b = min(slopes.values())
    witness = min(slopes, key=slopes.get)

    def ell(x):
        return xi[4] + b*(x-4)

    # This is automatic from the definition of b, but keep it as a guard.
    for c in cs:
        assert xi[c] >= ell(c), ("secant", t, c, xi[c]-ell(c))

    mu = Fraction((t-6)*(t-5), 3*(t+1))
    assert b < 0, ("slope", t, b)
    assert ell(mu) > 0, ("positive-line", t, ell(mu))

    bound = Fraction(3*(t+1), 2) * ell(mu)
    margin = bound - (t-1)
    assert margin > 0, ("final", t, margin)

    return {
        "t": t,
        "witness": witness,
        "bound": bound,
        "margin": margin,
        "min_kernel_gap": min_kernel_gap,
        "min_pair": min_pair,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the slower pairing-lemma check and verify only the final certificate",
    )
    args = parser.parse_args()

    if not args.fast:
        total = 0
        for t in range(23, 100):
            _, checked = check_pairing_t(t)
            total += checked
            # These caches contain the large coefficient tables for one value of t.
            # Clearing them keeps the full verification memory-light.
            paired_coeff.cache_clear()
            good_count.cache_clear()
            total_categories.cache_clear()
            single_good.cache_clear()
        print(
            f"Pairing lemma certified exactly for 23 <= t <= 99 "
            f"({total} admissible quadruples)."
        )

    certs = [certify_t(t) for t in range(23, 100)]
    worst = min(certs, key=lambda z: z["margin"])

    print("Uniform certificate certified exactly for 23 <= t <= 99.")
    print("Smallest final margin occurs at t =", worst["t"])
    print("Secant witness =", worst["witness"])
    print("Exact expected-alpha lower bound =", worst["bound"])
    print("Exact margin above t-1 =", worst["margin"])
    print("Decimal bound =", float(worst["bound"]))
    print("Decimal margin =", float(worst["margin"]))


if __name__ == "__main__":
    main()
