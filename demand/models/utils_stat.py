import scipy.special as sc
from numpy import floor
from numpy import log
from numpy import round
from numpy import sum
from numpy import exp
from numpy import sqrt
from numpy import pi
from numpy import abs
from numpy import mean
from numpy import maximum
from numpy.random import normal
from numpy.random import beta
from numpy.random import geometric
from scipy.stats import nbinom



ZERO_PLUS_EPSILON = 0 + (1e-10)
ONE_MINUS_EPSILON = 1 - (1e-10)


def gammapoisson_logpmf(k, a, b):
    """Log-pmf of a gamma-Poisson compound r.v. in shape-rate parameterization.

    This is equivalent to the negative binomial distribution, which models the
    number of trials k required to achieve a successes with probability p.

       f(k; a, b) = \int_0^\infty Poisson(k; L) * Gamma(L; a, b) dL

                     Gamma(k + a)    /    1    \ k    /        1    \ a
                  = -------------- * | ------- |    * | 1 - ------- |
                     k! Gamma(a)     \  b + 1  /      \      b + 1  /

    :param k: discrete value
    :param a: shape parameter (a > 0)
    :param b: rate parameter (b > 0)
    :return: the log pmf
    """
    # p = 1.0 / (b + 1.0)
    # binomln = sc.gammaln(k + a) - sc.gammaln(k + 1) - sc.gammaln(a)
    # return binomln + sc.xlogy(k, p) + sc.xlog1py(a, -p)

    # this version is empirically more stable, using the mean dispersion formulation
    # k = floor(k)

    mu = a * (1 / (b))
    d = (1 / a)

    term_1 = sc.gammaln((1/d + k))
    term_2 = - sc.gammaln((k + 1))
    term_3 = - sc.gammaln((1 / d))
    term_4 = k * log(d * mu)
    term_5 = - k * log(d * mu + 1)
    term_6 = (1/d) * log(1)
    term_7 = - (1/d) * log(d * mu + 1)

    return term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7


    # n = a
    # p = 1 / (1 + b)
    #
    # term1 = sc.gammaln(k + n)
    # term2 = - sc.gammaln(n)
    # term3 = - sc.gammaln(k + 1)
    # term4 = k * log(1 - p)
    # term5 = n * log(p)
    #
    # log_ret = term1 + term2 + term3 + term4 + term5
    #
    # return log_ret


def gammapoisson_logpmf_plotting(k, a, b):
    """Log-pmf of a gamma-Poisson compound r.v. in shape-rate parameterization.

    This is equivalent to the negative binomial distribution, which models the
    number of trials k required to achieve a successes with probability p.

       f(k; a, b) = \int_0^\infty Poisson(k; L) * Gamma(L; a, b) dL

                     Gamma(k + a)    /    1    \ k    /        1    \ a
                  = -------------- * | ------- |    * | 1 - ------- |
                     k! Gamma(a)     \  b + 1  /      \      b + 1  /

    :param k: discrete value
    :param a: shape parameter (a > 0)
    :param b: rate parameter (b > 0)
    :return: the log pmf
    """
    # p = 1.0 / (b + 1.0)
    # binomln = sc.gammaln(k + a) - sc.gammaln(k + 1) - sc.gammaln(a)
    # return binomln + sc.xlogy(k, p) + sc.xlog1py(a, -p)

    # this version is empirically more stable, using the mean dispersion formulation
    k = round(k)

    mu = a * (1 / (b))
    d = (1 / a)

    term_1 = sc.gammaln((1/d + k))
    term_2 = - sc.gammaln((k + 1))
    term_3 = - sc.gammaln((1 / d))
    term_4 = k * log(d * mu)
    term_5 = - k * log(d * mu + 1)
    term_6 = (1/d) * log(1)
    term_7 = - (1/d) * log(d * mu + 1)

    return term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7

    # n = a
    # p = 1 / (1 + b)
    #
    # term1 = sc.gammaln(k + n)
    # term2 = - sc.gammaln(n)
    # term3 = - sc.gammaln(k + 1)
    # term4 = k * log(1 - p)
    # term5 = n * log(p)
    #
    # log_ret = term1 + term2 + term3 + term4 + term5
    #
    # return log_ret


def gammapoisson_mean(a, b):
    """Mean of a gamma-Poisson compound r.v. in shape-rate parameterization.

    This is equivalent to the negative binomial distribution, which models the
    number of trials k required to achieve a successes with probability p.

       f(k; a, b) = \int_0^\infty Poisson(k; L) * Gamma(L; a, b) dL

                     Gamma(k + a)    /    1    \ k    /        1    \ a
                  = -------------- * | ------- |    * | 1 - ------- |
                     k! Gamma(a)     \  b + 1  /      \      b + 1  /

    :param k: discrete value
    :param a: shape parameter (a > 0)
    :param b: rate parameter (b > 0)
    :return: the mean rate
    """
    return a / b


def gammapoisson_mode(a, b):
    """Mode of a gamma-Poisson compound r.v. in shape-rate parameterization.

    This is equivalent to the negative binomial distribution, which models the
    number of trials k required to achieve a successes with probability p.

       f(k; a, b) = \int_0^\infty Poisson(k; L) * Gamma(L; a, b) dL

                     Gamma(k + a)    /    1    \ k    /        1    \ a
                  = -------------- * | ------- |    * | 1 - ------- |
                     k! Gamma(a)     \  b + 1  /      \      b + 1  /

    :param k: discrete value
    :param a: shape parameter (a > 0)
    :param b: rate parameter (b > 0)
    :return: the mode rate
    """
    return (a > 1) * floor((a - 1) / b) + 0.0  # the +0.0 ensures it's not -0.

def gammapoisson_var(a, b):
    """Variance of a gamma-Poisson compound r.v. in shape-rate parameterization.

    This is equivalent to the negative binomial distribution, which models the
    number of trials k required to achieve a successes with probability p.

       f(k; a, b) = \int_0^\infty Poisson(k; L) * Gamma(L; a, b) dL

                     Gamma(k + a)    /    1    \ k    /        1    \ a
                  = -------------- * | ------- |    * | 1 - ------- |
                     k! Gamma(a)     \  b + 1  /      \      b + 1  /

    :param k: discrete value
    :param a: shape parameter (a > 0)
    :param b: rate parameter (b > 0)
    :return: the variance
    """

    # # derivation of GaPo variance
    # p = 1 / (b + 1)
    # r = a
    #
    # return (p * r) / (1 - p) ** 2

    return (a * (b + 1)) / (b ** 2)


def poisson_logpmf(x, lam):
    """Log-pmf of a Poisson discrete random variable.

    Poisson random variables are interpreted as the number of occurrences (k)
    in a single interval with occurrence rate lam > 0.  The pmf of a Poisson
    discrete random variable is

                    L^k exp(-L)
        f(k | L) = -------------
                        k!

    where k = floor(x) and x in {0, 1, 2, ...}.  We compute
        log(k!) = gammaln(k+1).

    :param x: the number of occurrences
    :param lam: the rate parameter
    :return: log pmf
    """
    k = floor(x)
    return sc.xlogy(k, lam) - lam - sc.gammaln(k + 1)


def poisson_logpmf_plotting(x, lam):
    """Log-pmf of a Poisson discrete random variable.

    Poisson random variables are interpreted as the number of occurrences (k)
    in a single interval with occurrence rate lam > 0.  The pmf of a Poisson
    discrete random variable is

                    L^k exp(-L)
        f(k | L) = -------------
                        k!

    where k = floor(x) and x in {0, 1, 2, ...}.  We compute
        log(k!) = gammaln(k+1).

    :param x: the number of occurrences
    :param lam: the rate parameter
    :return: log pmf
    """
    k = round(x)
    return sc.xlogy(k, lam) - lam - sc.gammaln(k + 1)

def gamma_ab_logpdf(x, a, b):
    """Log-pdf of a gamma continuous r.v. in shape-rate parametrization.

    The probability density function the gamma distribution is:

                         b^a      a-1  -bx
        f(x | a, b) = ---------- x    e
                       Gamma(a)

    for x >= 0, a > 0, b > 0.  When a is an integer, the gamma distribution
    reduces to the Erlang distribution and for a=1 we have the exponential
    distribution.  Note that it is defined in terms of the shape and rate
    parametrization instead of the shape and scale parametrization.

    :param x: shape [n, m], the values of x >= 0 at which to evaluate the pdf
    :param a: shape parameter > 0
    :param b: scale parameter > 0
    :return: the log pdf evaluations
    """
    return sc.xlogy(a - 1.0, x) - b * x + sc.xlogy(a, b) - sc.gammaln(a)


def gamma_mean(a, b):
    """Mean of a gamma continuous r.v. in shape-rate parametrization.

    The probability density function the gamma distribution is:

                         b^a      a-1  -bx
        f(x | a, b) = ---------- x    e
                       Gamma(a)

    for x >= 0, a > 0, b > 0.  When a is an integer, the gamma distribution
    reduces to the Erlang distribution and for a=1 we have the exponential
    distribution.  Note that it is defined in terms of the shape and rate
    parametrization instead of the shape and scale parametrization.

    :param a: shape parameter > 0
    :param b: scale parameter > 0
    :return: the mean evaluations
    """
    return a / b


def cpoiss3_logpdf(a, b, lam):
    """Log-pdf of a three-parameter continuous extension of a Poisson r.v.

    THIS IS AN UNNORMALIZED FUNCTION, *NOT* A PROBABILITY DISTRIBUTION.

    This extension of Poisson random variables are interpreted as the number of
    occurrences (a > 0) in a number of intervals (b > 0) with occurrence rate
    lam > 0.  The pdf of this discrete random variable is

                       L^a exp(-b*L)
        f(a ; b, L) = -------------
                        Gamma(a+1)

    where a, b, L > 0.

    :param a: the number of occurrences
    :param b: the number of intervals
    :param lam: the rate parameter
    :return: log pmf
    """
    return sc.xlogy(a, lam) - b * lam - sc.gammaln(a + 1)



def cbinom_logpdf(x, n, p):
    """Log-pdf of a binomial continuous random variable.

    The probability mass function for the binomial distribution is:

       f(x | n, p) = choose(n, x) * p**x * (1-p)**(n-x)

                           gamma(n+1)
                   = ----------------------- * p**x * (1-p)**(n-x)
                    gamma(x+1)*gamma(n-x+1)

    for x in {0, 1,..., n}, n positive integer, and 0 <= p <= 1 where gamma(z)
    is the complete gamma function (scipy.special.gamma)

    :param x: shape [N, M], the values of x at which to evaluate the pmf
    :param n: shape [1] or [N, M], the number of total trials for x
    :param p: shape [1] or [N, M], the probability of success 0 <= p <= 1
    :return: the log pmf evaluations
    """
    lognck = sc.gammaln(n + 1) - sc.gammaln(x + 1) - sc.gammaln(n - x + 1)

    #        (n-x) * log(1-p)    +   x * log(p)   + log nCk
    return sc.xlog1py(n - x, -p) + sc.xlogy(x, p) + lognck


def binom_logpmf(x, n, p, no_combi=False):
    """Log-pmf of a binomial discrete random variable.

    The probability mass function for the binomial distribution is:

       f(k | n, p) = choose(n, k) * p**k * (1-p)**(n-k)

                           gamma(n+1)
                   = ----------------------- * p**k * (1-p)**(n-k)
                    gamma(k+1)*gamma(n-k+1)

    for k in {0, 1,..., n}, n positive integer, and 0 <= p <= 1 where gamma(z)
    is the complete gamma function (scipy.special.gamma)

    :param x: shape [N, M], the values of x at which to evaluate the pmf
    :param n: shape [1] or [N, M], the number of total trials for x
    :param p: shape [1] or [N, M], the probability of success 0 <= p <= 1
    :param no_combi: boolean; if true only return log( p**k * (1-p)**(n-k) )
    :return: the log pmf evaluations
    """
    k = floor(x)
    if no_combi:
        lognck = 0.0
    else:
        lognck = sc.gammaln(n + 1) - sc.gammaln(k + 1) - sc.gammaln(n - k + 1)

    #       (n-x) * log(1-p)   +   x * log(p)   + log nCk
    return sc.xlog1py(n - k, -p) + sc.xlogy(k, p) + lognck


def beta_logpdf(x, a, b):
    """Log-pdf of a beta continuous random variable.

    The probability density function for the beta distribution is:

                          gamma(a+b)
        f(x | a, b) = ------------------- * x**(a-1) * (1-x)**(b-1)
                       gamma(a)*gamma(b)

                       x**(a-1) * (1-x)**(b-1)
                    = -------------------------
                              beta(a, b)

    for 0 < x < 1, a > 0, b > 0, where gamma(z) is the complete gamma function
    (scipy.special.gamma) and beta(a, b) is the complete beta function.

    :param x: shape [N, M], the values of x at which to evaluate the pmf
    :param a: shape [1] or [N, M], the number of prior successes, a > 0
    :param b: shape [1] or [N, M], the number of prior failures, b > 0
    :return: the log pdf evaluations
    """
    logpdf = sc.xlog1py(b - 1.0, -x) + sc.xlogy(a - 1.0, x)
    logpdf -= sc.betaln(a, b)
    return logpdf


def gamma_logpdf(x, k, t=None):
    """Log-pdf of a gamma continuous random variable.

    The probability density function the gamma distribution is:

        f(x | k, t)     = x**(k-1) * exp(-x/t) / (gamma(k) * t**k)
        f(x | k, t = 1) = x**(k-1) * exp(-x) / gamma(k)

    for x >= 0, k > 0.  When k is an integer, the gamma distribution reduces to
    the Erlang distribution and for k=1 we have the exponential distribution.
    Note that it is defined in terms of the shape and scale parametrization
    instead of the shape and rate parametrization.

    :param x: shape [n, m], the values of x >= 0 at which to evaluate the pdf
    :param k: shape parameter > 0
    :param t: scale parameter > 0
    :return: the log pdf evaluations
    """
    if t is None:
        return sc.xlogy(k - 1.0, x) - x - sc.gammaln(k)
    else:
        return sc.xlogy(k - 1.0, x) - x / t - sc.gammaln(k) - sc.xlogy(k, t)


def betabinom_mean(a, b, n):
    """Mean of a beta-binomial discrete random variable

    :param a: the alpha parameter, number of prior successes, a > 0
    :param b: the beta parameter, number of prior failures, b > 0
    :param n: the number of total trials
    :return: the mean of the distribution(s)
    """
    return n * a / (a + b)


def betabinom_variance(a, b, n):
    """Variance of a beta-binomial discrete random variable

    :param a: the alpha parameter, number of prior successes, a > 0
    :param b: the beta parameter, number of prior failures, b > 0
    :param n: the number of total trials
    :return: the mean of the distribution(s)
    """
    return (n * a * b / (a + b) ** 2) * (a + b + n) / (a + b + 1)


def betabinom_logpmf(x, a, b, n):
    """Log-pmf of a beta-binomial discrete random variable.

    The probability mass function for the beta-binomial distribution is

       f(k | a, b, n) = \int_0^1 Bin(k | t; n) Beta(t | a, b) dt

                    gamma(n+1)        gamma(a+k)*gamma(b+n-k)    gamma(a+b)
            = ----------------------- ----------------------- -----------------
              gamma(k+1)*gamma(n-k+1)      gamma(a+b+n)       gamma(a)*gamma(b)

                     gamma(n+1)            beta(a+k, b+n-k)
            = ------------------------- * ------------------
               gamma(k+1)*gamma(n-k+1)        beta(a, b)

    for k in {0, ..., n}, a > 0, b > 0, where gamma(z) is the complete gamma
    function (scipy.special.gamma).

    :param x: shape [N, M], the values of x at which to evaluate the pmf
    :param a: shape [1] or [N, M], the number of prior successes, a > 0
    :param b: shape [1] or [N, M], the number of prior failures, b > 0
    :param n: shape [1] or [n, m], the number of total trials for x
    :return: the log pmf evaluations
    """
    return cbetabinom_logpdf(x, a, b, n)


def cbetabinom_logpdf(x, a, b, n):
    """Log-pmf of a beta-binomial continuous random variable.

    The probability density function for the beta-binomial distribution is

       f(x | a, b, n) = \int_0^1 Binc(x | t; n) Beta(t | a, b) dt

                    gamma(n+1)        gamma(a+k)*gamma(b+n-x)    gamma(a+b)
            = ----------------------- ----------------------- -----------------
              gamma(x+1)*gamma(n-x+1)      gamma(a+b+n)       gamma(a)*gamma(b)

                     gamma(n+1)            beta(a+x, b+n-x)
            = ------------------------- * ------------------
               gamma(x+1)*gamma(n-x+1)        beta(a, b)

    for k in {0, ..., n}, a > 0, b > 0, where gamma(z) is the complete gamma
    function (scipy.special.gamma).

    :param x: shape [N, M], the values of x at which to evaluate the pdf
    :param a: shape [1] or [N, M], the number of prior successes, a > 0
    :param b: shape [1] or [N, M], the number of prior failures, b > 0
    :param n: shape [1] or [n, m], the number of total trials for x
    :return: the log pmf evaluations
    """
    logpmf = sc.betaln(a + x, b + n - x) - sc.betaln(a, b)
    # logpmf = sc.gammaln(a + k) + sc.gammaln(b + n - k) - sc.gammaln(a + b + n)
    # logpmf += sc.gammaln(a + b) - sc.gammaln(a) - sc.gammaln(b)
    logpmf += sc.gammaln(n + 1) - sc.gammaln(x + 1) - sc.gammaln(n - x + 1)

    return logpmf


def nckln(n, k):
    """ Log of the n-choose-k, where any argument can be a real number.

                                  Gamma(n + 1)
    Computed via: nCk = ---------------------------------
                         Gamma(k + 1) * Gamma(n - k + 1)
    :param n:
    :param k:
    :return:
    """
    return sc.gammaln(n + 1) - sc.gammaln(k + 1) - sc.gammaln(n - k + 1)


def beta_kldiv(ap, bp, aq, bq):
    """Compute the KL divergence from Q = Beta(aq, bq) to P = Beta(ap, bp).

    This is KL(P || Q), the information lost when Q is used to approximate P.

    :param ap: the alpha parameter for distribution P
    :param bp: the beta parameter for distribution P
    :param aq: the alpha parameter for distribution Q
    :param bq: the beta parameter for distribution Q
    :return: KL(P || Q)
    """
    div = sc.gammaln(ap + bp) - sc.gammaln(ap) - sc.gammaln(bp)
    div += -(sc.gammaln(aq + bq) - sc.gammaln(aq) - sc.gammaln(bq))
    div += (ap - aq) * (sc.digamma(ap) - sc.digamma(ap + bp))
    div += (bp - bq) * (sc.digamma(bp) - sc.digamma(ap + bp))
    return div


def beta_mean(a, b):
    """Mean of a beta-distributed random variable

    :param a: the alpha parameter, number of prior successes
    :param b: the beta parameter, number of prior failures
    :return: the mean of the distribution(s)
    """
    return a / (a + b)


def beta_variance(a, b):
    """Variance of a beta-distributed random variable

    :param a: the alpha parameter, number of prior successes
    :param b: the beta parameter, number of prior failures
    :return: the variance of the distribution(s)
    """
    return (a * b) / ((a + b) ** 2 * (a + b + 1))


def beta_mode(a, b):
    """Mode of a beta-distributed random variable

    :param a: the alpha parameter, number of prior successes
    :param b: the beta parameter, number of prior failures
    :return: the mean of the distribution(s)
    """
    return (a - 1) / (a + b - 2)


def beta_entropy(a, b):
    """Entropy of a beta-distributed random variable

    :param a: the alpha parameter, number of prior successes
    :param b: the beta parameter, number of prior failures
    :return: the entropy of the distribution(s)
    """
    return sc.betaln(a, b) \
           - (a - 1) * sc.digamma(a) - (b - 1) * sc.digamma(b) \
           + (a + b - 2) * sc.digamma(a + b)


def randn_log10(log10mu, log10sigma, size):
    """ Sample from a normal distribution in log space

    :param log10mu: the mean of the distribution
    :param log10sigma: the standard deviation of the distribution
    :param size:
    :return:
    """
    return 10 ** (log10mu + log10sigma * normal(size=size))


def beta_random(a, b, size=None, safe=False):
    """

    :param a:
    :param b:
    :param size:
    :param safe:
    :return:
    """
    tmp = beta(a, b, size=size)
    if safe:
        # for numerical stability, ensure that samples aren't too close to 0, 1
        tmp[tmp < ZERO_PLUS_EPSILON] = ZERO_PLUS_EPSILON
        tmp[tmp > ONE_MINUS_EPSILON] = ONE_MINUS_EPSILON
    return tmp


def geometric_logpmf(x, p):
    """Log-pmf of a shifted geometric discrete random variable.

    The probability mass function for the geometric distribution is:

       f(k | p) = p * (1-p)**(k-1)

    for k in {1, 2, 3, ...} and 0 <= p <= 1.

    :param x: shape [N, M], the values of x at which to evaluate the pmf
    :param p: shape [1] or [N, M], the probability of success 0 < p <= 1
    """

    return sc.xlogy(x - 1.0, 1.0 - p) + log(p)


def geometric_random(p, size=None):
    """Random realizations of a shifted geometric discrete random variable.

    The probability mass function for the geometric distribution is:

       f(k   p) = p * (1-p)**(k-1)

    for k in {1, 2, 3, ...} and 0 <= p <= 1.  To specify a distribution with
    mean mu, set p = 1 / mu.

    :param p: shape [1] or [N, M], the probability of success 0 < p <= 1

    """
    return geometric(p, size=size)


def softplus(z):
    """The softplus function f(z) = np.log(1 + exp(z)).

    It is safe for z >> 0 by the following identity:
        log(1 + exp(z)) = log(1 + exp(z)) - log(exp(z)) + z
                        = log((1 + exp(z))/exp(z)) + z
                        = log(1 + exp(-z)) + z
    and since it's only required for z >> 0 we use
        log(1 + exp(z)) = log(1 + exp(-|z|)) + max(0, z).
    """
    return log(1 + exp(-abs(z))) + maximum(0, z)


def logistic(z):
    """The logistic sigmoid function."""
    return sc.expit(z)


def halfnormal_logpdf(x, tau):
    """Log-pdf of a half-normal random variable with precision tau.

    The p.d.f. for the half-normal distribution with precision tau:

                               /    2    2 \
                     2*tau     |   x  tau  |
       f(x | tau) = ------- exp|- -------- |
                       pi      \     pi    /

    for x >= 0 and tau > 0.  E[X] = 1 / tau.

    :param x: shape [N, M], the values of x to evaluate
    :param tau: shape [1] or [N, M], the precision
    :return: shape_of(x); the log pdf
    """

    return log(2) + log(tau) - log(pi) - x ** 2 * tau ** 2 / pi


def halfnormal_random(tau, size=1):
    """Random realizations of a half-normal random variable with precision tau.

    The p.d.f. for the half-normal distribution with precision tau:

                               /    2    2 \
                     2*tau     |   x  tau  |
       f(x | tau) = ------- exp|- -------- |
                       pi      \     pi    /

    for x >= 0 and tau > 0.  E[X] = 1 / tau.

    If X ~ N(0, sig^2), then |X| ~ HN(sig).
    We instead parameterize the distribution by
        tau = sqrt(pi) / (tau * sqrt(2)).

    :param x: shape [N, M], the values of x to evaluate
    :param tau: shape [1] or [N, M], the precision
    :return: shape [SIZE], pseudorandom samples from the distribut
    """
    sig = sqrt(pi) / (tau * sqrt(2))
    return sig * abs(normal(size=size))


def lognormal_logpdf(x, m=0, s=1):
    """Log-pdf of a log-normal random variable with parameters mu and sigma.

    If X ~ LN(m, s^2) then log(X) ~ N(m, s^2).  The p.d.f. for the log-normal
    random variable with parameters mu and sigma:

                                           /              2 \
                      1         1          |  (log(x) - m)  |
       f(x | m, s) = --- -------------- exp|- ------------- |
                      x   s sqrt(2\pi)     \       s^2      /

    for x > 0, -inf < mu < inf, and s > 0.  E[X] = exp(m + 0.5*s^2).

    :param x: shape [N, M], the values of x to evaluate
    :param m: shape [1] or [N, M], the mean of log(x)
    :param s: shape [1] or [N, M], the stddev of log(x)
    :return: shape_of(x); the log pdf
    """

    return -(log(x) - m) ** 2 / (2 * s ** 2) - log(x) - log(s) - 0.5*log(2*pi)


def lognormal_random(m=0, s=1, size=1):
    """Random realizations of a log-normal random variable with mean m and std. dev. s.

    If X ~ LN(m, s^2) then log(X) ~ N(m, s^2).  The p.d.f. for the log-normal
    random variable with parameters mu and sigma:

                                           /              2 \
                      1         1          |  (log(x) - m)  |
       f(x | m, s) = --- -------------- exp|- ------------- |
                      x   s sqrt(2\pi)     \       s^2      /

    for x > 0, -inf < mu < inf, and s > 0.  E[X] = exp(m + 0.5*s^2).

    :param m: shape [1] or [N, M], the mean of log(X)
    :param s: shape [1] or [N, M], the stddev of log(X)
    :return: shape [SIZE], pseudorandom samples from the distribution
    """

    return exp(m + s * normal(size=size))


def sample_from_distr_propto(x, f, ns, return_cdf=False):
    """Sample from a distribution over x that's proportional to f.

    :param x: shape (nx, )
    :param f: shape (nx, ) or (m, nx)
    :param ns: scalar int, number of samples
    :param return_cdf: default False
    :return: shape (ns, ) or (m, ns)
    """
    from scipy import integrate
    import numpy as np

    dx = x[1] - x[0]
    cdf = integrate.cumtrapz(f, x, initial=0)  # shape (nx, ) or (m, nx)

    if cdf.ndim == 1:  # i.e. cdf has shape (nx, )
        cdf /= cdf[-1]  # normalize to one
        bins = np.digitize(np.random.rand(ns), cdf, right=True)  # shape (ns, )
        samples = x[bins] - dx * np.random.rand(ns)

    else:  # cdf has shape (m, nx)
        samples = np.zeros((f.shape[0], ns))
        for i in range(f.shape[0]):
            cdf[i, :] /= cdf[i, -1]  # normalize to one
            bins = np.digitize(np.random.rand(ns), cdf[i, :],
                               right=True)  # shape (ns, )
            samples[i, :] = x[bins] - dx * np.random.rand(ns)

    if return_cdf:
        return (samples.squeeze(), cdf.squeeze())
    else:
        return samples.squeeze()


def rmse(y, yhat):
    """Compute the root-mean-squared error between the two inputs.

    :param y: shape (n0, n1, ...)
    :param yhat: shape (n0, n1, ...)
    :return: scalar
    """
    return sqrt(mean((y - yhat) ** 2))


def discrete_entropy(P, axis=-1, keepdims=False):
    """Compute the entropy of a discrete distribution along the specified axis.

    For example, for an array with shape (N, M) such that the rows sum to 1, the
    computes the entropy of the N different discrete distributions.
    :param P: shape (n0, n1, ... ), array of probabilities that sum to 1 along
                                    the last dimension
    :param axis: the axis over which to compute the distribution (default -1)
    :param keepdims: boolean, whether or not to keep the dimension of summation
    :return: -sum_i (P_i * log P_i), shape squeezes out the specified axis.
    """
    return -sum(P * log(P), axis=axis, keepdims=keepdims)


