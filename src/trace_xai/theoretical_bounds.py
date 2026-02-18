"""Theoretical fidelity bounds for surrogate decision trees.

Provides PAC-learning, VC-dimension, and Rademacher complexity based bounds
on the fidelity of depth-D surrogate trees approximating black-box models.

References
----------
- Blanc, Lange, Malik, Tan, "Decision trees as partitioning machines" (NeurIPS 2020)
- Hanneke, "The Optimal Sample Complexity of PAC Learning" (JMLR 2016)
- Blumer, Ehrenfeucht, Haussler, Warmuth, "Learnability and the VC Dimension" (1989)
- Bartlett & Mendelson, "Rademacher and Gaussian Complexities" (JMLR 2002)
- Montufar, Pascanu, Cho, Bengio, "Number of Linear Regions of Deep NNs" (NeurIPS 2014)
"""

from __future__ import annotations

import math
from dataclasses import dataclass




@dataclass(frozen=True)
class FidelityBound:
    """Theoretical bound on surrogate fidelity.

    Attributes
    ----------
    vc_dimension : int
        VC-dimension of the depth-D tree hypothesis class.
    estimation_error_pac : float
        PAC-based upper bound on estimation error (agnostic setting).
    estimation_error_rademacher : float
        Rademacher-based upper bound on estimation error.
    min_fidelity_pac : float
        Lower bound on fidelity using PAC estimation error.
        ``fidelity >= 1 - empirical_infidelity - estimation_error_pac``
    min_fidelity_rademacher : float
        Lower bound on fidelity using Rademacher estimation error.
    sample_complexity_required : int
        Minimum N to guarantee estimation_error <= target_epsilon.
    depth : int
        Surrogate tree depth used for computation.
    n_features : int
        Number of input features.
    n_samples : int
        Number of training samples.
    confidence : float
        Confidence level (1 - delta).
    max_decision_regions : int
        Maximum number of decision regions the tree can represent (2^D).
    """

    vc_dimension: int
    estimation_error_pac: float
    estimation_error_rademacher: float
    min_fidelity_pac: float
    min_fidelity_rademacher: float
    sample_complexity_required: int
    depth: int
    n_features: int
    n_samples: int
    confidence: float
    max_decision_regions: int

    def __str__(self) -> str:
        lines = [
            "=== Theoretical Fidelity Bounds ===",
            f"  Surrogate depth D = {self.depth}, features d = {self.n_features}",
            f"  Training samples N = {self.n_samples}",
            f"  Confidence level: {self.confidence:.0%}",
            "",
            f"  VC-dimension of tree class: {self.vc_dimension}",
            f"  Max decision regions (2^D): {self.max_decision_regions}",
            "",
            f"  Estimation error (PAC/VC):        <= {self.estimation_error_pac:.4f}",
            f"  Estimation error (Rademacher):     <= {self.estimation_error_rademacher:.4f}",
            "",
            f"  Min fidelity guarantee (PAC):      >= {self.min_fidelity_pac:.4f}",
            f"  Min fidelity guarantee (Rademacher):>= {self.min_fidelity_rademacher:.4f}",
            "",
            f"  Sample complexity for eps=0.05:    N >= {self.sample_complexity_required}",
        ]
        return "\n".join(lines)


def vc_dimension_decision_tree(depth: int, n_features: int) -> int:
    """Compute the VC-dimension bound for depth-D decision trees.

    Uses the result from Blanc et al. (NeurIPS 2020):
        VCdim(T_{D,d}) = O(2^D * (D + log2(d)))

    Parameters
    ----------
    depth : int
        Maximum depth of the decision tree.
    n_features : int
        Number of input features.

    Returns
    -------
    int
        Upper bound on the VC-dimension.
    """
    if depth <= 0 or n_features <= 0:
        return 0
    n_internal = 2**depth - 1
    # Blanc et al.: V = O(N_internal * log(N_internal * d))
    # We use the tighter form: V ≈ 2^D * (D + log2(d))
    log_d = max(math.log2(n_features), 1.0)
    return int(math.ceil(n_internal * (depth + log_d)))


def estimation_error_pac(
    vc_dim: int,
    n_samples: int,
    delta: float = 0.05,
) -> float:
    """PAC-based upper bound on estimation error (agnostic setting).

    From Blumer et al. (1989), agnostic PAC bound:
        epsilon_est <= sqrt((V * ln(2*N/V) + ln(4/delta)) / (2*N))

    This bounds the gap between empirical and true disagreement rate.

    Parameters
    ----------
    vc_dim : int
        VC-dimension of the hypothesis class.
    n_samples : int
        Number of training samples.
    delta : float
        Failure probability (bound holds with prob >= 1 - delta).

    Returns
    -------
    float
        Upper bound on estimation error.
    """
    if n_samples <= 0 or vc_dim <= 0:
        return 1.0
    v = vc_dim
    n = n_samples
    # Vapnik-Chervonenkis inequality (agnostic form)
    ratio = max(2 * n / v, 1.0)
    numerator = v * math.log(ratio) + math.log(4.0 / delta)
    return math.sqrt(numerator / (2 * n))


def estimation_error_rademacher(
    depth: int,
    n_features: int,
    n_samples: int,
    delta: float = 0.05,
) -> float:
    """Rademacher complexity-based bound on estimation error.

    From Bartlett & Mendelson (JMLR 2002):
        R_n(T_{D,d}) <= c * sqrt(2^D * log(d) / n)

    Generalization bound:
        err <= emp_err + 2*R_n + sqrt(ln(1/delta) / (2*n))

    Parameters
    ----------
    depth : int
        Surrogate tree depth.
    n_features : int
        Number of features.
    n_samples : int
        Number of training samples.
    delta : float
        Failure probability.

    Returns
    -------
    float
        Upper bound on estimation error via Rademacher complexity.
    """
    if n_samples <= 0 or depth <= 0:
        return 1.0
    n = n_samples
    n_leaves = 2**depth
    log_d = max(math.log(n_features), 1.0)

    # Empirical Rademacher complexity
    rademacher = math.sqrt(n_leaves * log_d / n)

    # Full generalization bound
    confidence_term = math.sqrt(math.log(1.0 / delta) / (2 * n))
    return 2 * rademacher + confidence_term


def sample_complexity(
    vc_dim: int,
    epsilon: float = 0.05,
    delta: float = 0.05,
) -> int:
    """Minimum samples to guarantee estimation error <= epsilon.

    Agnostic PAC bound (Blumer et al. 1989):
        N >= (V + ln(1/delta)) / epsilon^2

    Parameters
    ----------
    vc_dim : int
        VC-dimension of the tree hypothesis class.
    epsilon : float
        Target estimation error bound.
    delta : float
        Failure probability.

    Returns
    -------
    int
        Minimum number of samples required.
    """
    if epsilon <= 0:
        return 0
    return int(math.ceil((vc_dim + math.log(1.0 / delta)) / epsilon**2))


def relu_network_regions(
    n_layers: int,
    width: int,
    input_dim: int,
) -> int:
    """Upper bound on linear regions of a ReLU network.

    From Montufar et al. (NeurIPS 2014):
        R(f) <= prod_{i=1}^{L} sum_{j=0}^{min(n0, ni)} C(ni, j)

    For equal width W >= n0:
        R(f) <= (eW/n0)^{n0*L}

    Parameters
    ----------
    n_layers : int
        Number of hidden layers.
    width : int
        Neurons per hidden layer (assumes equal width).
    input_dim : int
        Input dimensionality.

    Returns
    -------
    int
        Upper bound on the number of linear regions.
    """
    if n_layers <= 0 or width <= 0 or input_dim <= 0:
        return 1

    if width >= input_dim:
        # Montufar et al. simplified bound
        ratio = math.e * width / input_dim
        exponent = input_dim * n_layers
        # Cap to avoid overflow
        log_regions = exponent * math.log2(ratio)
        if log_regions > 63:
            return 2**63  # cap at ~9.2e18
        return int(math.ceil(ratio**exponent))
    else:
        # General case: compute layer by layer
        total = 1
        for _ in range(n_layers):
            k = min(input_dim, width)
            layer_sum = sum(
                math.comb(width, j) for j in range(k + 1)
            )
            total *= layer_sum
            if total > 2**63:
                return 2**63
        return total


def min_depth_for_regions(n_regions: int) -> int:
    """Minimum tree depth to represent n_regions decision regions.

    D >= ceil(log2(K))

    Parameters
    ----------
    n_regions : int
        Number of decision regions to represent.

    Returns
    -------
    int
        Minimum required tree depth.
    """
    if n_regions <= 1:
        return 0
    return int(math.ceil(math.log2(n_regions)))


def optimal_depth_bound(
    n_samples: int,
    n_features: int,
) -> int:
    """Theoretical optimal depth minimizing total error bound.

    D* = O(log(N / (d * log N)))

    This balances approximation error (decreasing in D) against
    estimation error (increasing in D). Derived from minimizing:
        epsilon_total = c1 * 2^{-D/d} + c2 * sqrt(2^D * (D + log d) / N)

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features.

    Returns
    -------
    int
        Theoretically optimal depth.
    """
    if n_samples <= 1 or n_features <= 0:
        return 1
    log_n = math.log2(max(n_samples, 2))
    log_d = max(math.log2(n_features), 1.0)
    # D* ≈ log2(N / (d * log2(N)))
    denominator = max(log_d * log_n, 1.0)
    d_star = math.log2(max(n_samples / denominator, 2.0))
    return max(1, int(round(d_star)))


def compute_fidelity_bounds(
    depth: int,
    n_features: int,
    n_samples: int,
    empirical_infidelity: float = 0.0,
    delta: float = 0.05,
) -> FidelityBound:
    """Compute all theoretical fidelity bounds for a surrogate tree.

    Decomposes total infidelity as:
        Infidelity(T_D, f) <= empirical_infidelity + estimation_error

    where estimation_error is bounded by both PAC/VC and Rademacher methods.

    Parameters
    ----------
    depth : int
        Fitted surrogate tree depth.
    n_features : int
        Number of features in the dataset.
    n_samples : int
        Number of samples used for training.
    empirical_infidelity : float
        Observed disagreement rate (1 - fidelity) on training data.
    delta : float
        Failure probability. Bounds hold with prob >= 1 - delta.

    Returns
    -------
    FidelityBound
        Complete theoretical analysis.
    """
    confidence = 1.0 - delta

    vc_dim = vc_dimension_decision_tree(depth, n_features)
    eps_pac = estimation_error_pac(vc_dim, n_samples, delta)
    eps_rad = estimation_error_rademacher(depth, n_features, n_samples, delta)
    n_required = sample_complexity(vc_dim, epsilon=0.05, delta=delta)

    # Fidelity lower bounds (clamped to [0, 1])
    fidelity_pac = max(0.0, 1.0 - empirical_infidelity - eps_pac)
    fidelity_rad = max(0.0, 1.0 - empirical_infidelity - eps_rad)

    return FidelityBound(
        vc_dimension=vc_dim,
        estimation_error_pac=min(eps_pac, 1.0),
        estimation_error_rademacher=min(eps_rad, 1.0),
        min_fidelity_pac=fidelity_pac,
        min_fidelity_rademacher=fidelity_rad,
        sample_complexity_required=n_required,
        depth=depth,
        n_features=n_features,
        n_samples=n_samples,
        confidence=confidence,
        max_decision_regions=2**depth,
    )
