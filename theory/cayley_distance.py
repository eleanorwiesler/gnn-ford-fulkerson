def cayley_distance(sigma, sigma_hat):
    from sympy.combinatorics import Permutation
    s = Permutation(sigma)
    s_hat = Permutation(sigma_hat)
    return len(s.domain) - (s**-1 * s_hat).cycles.__len__()


def weighted_cayley_distance(sigma, sigma_hat, weights):
    transpositions = 0
    for i in range(len(sigma)):
        if sigma[i] != sigma_hat[i]:
            transpositions += weights[i]
    return transpositions
