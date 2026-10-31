import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from datetime import date
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multivariate VAR datasets."
    )

    # Dataset dimensions
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1000,
        help="Number of independent sequences to generate (default: 1000).",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1000,
        help="Number of retained time steps per sequence (default: 1000).",
    )

    parser.add_argument(
        "--num-variates",
        type=int,
        default=8,
        help="Number of variates in the VAR process (default: 8).",
    )

    # Simulation
    parser.add_argument(
        "--burn-in",
        type=int,
        default=200,
        help="Number of initial time steps discarded as burn-in (default: 200).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42).",
    )

    parser.add_argument(
        "--num-replicates",
        type=int,
        default=1,
        help="Number of independently generated DGP replicates (default: 1).",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Standard deviation of the VAR innovation process.",
    )

    parser.add_argument(
        "--max-lag",
        type=int,
        default=1,
       help=(
        "Maximum active lag. "
        "Active lags are generated as [1, 2, 4, ..., max_lag]."
        ),
    )

    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Lambda values to simulate.",
    )

    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Gamma values to simulate.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. If omitted, datasets are not written.",
    )

    parser.add_argument(
        "--interaction_type",
        type=str,
        default="sqrt",
        help="Non-linear interaction function.",
    )

    parser.add_argument(
        "--persistences",
        type=float,
        nargs="+",
        default=[0.99, 0.95, 0.9],
        help="Probability of remaining in the current hidden regime.",
    )

    parser.add_argument(
        "--regime-switching",
        action="store_true",
    )

    return parser.parse_args()

def make_D_stack(
    num_lags: int,
    num_variates: int,
    rng: np.random.Generator,
    low: float = -0.8,
    high: float = 0.8,
) -> np.ndarray:
    """Generate the diagonal autoregressive component of a VAR(1) process."""
    if not -1 < low < high < 1:
        raise ValueError("D bounds must satisfy -1 < low < high < 1.")
    D = np.zeros((num_lags,num_variates, num_variates), dtype=np.float64)
    for i in range(num_lags):
        coefficients = rng.uniform(low, high, size=num_variates)
        D[i] = np.diag(coefficients)

    assert D.shape == (
        num_lags,
        num_variates,
        num_variates,
    )
    return D

def make_B_stack(
    num_lags: int,
    num_variates: int,
    rng: np.random.Generator,
    density: float = 0.7,
    low: float = -0.5,
    high: float = 0.5,
) -> np.ndarray:
    """
    Generate the cross-variate interaction component of a VAR(1) process.
    """
    if not 0 <= density <= 1:
        raise ValueError("density must be between 0 and 1.")

    if low >= high:
        raise ValueError("low must be less than high.")
    # begin with everything set to 0
    B = np.zeros((num_lags, num_variates, num_variates), dtype=np.float64)
    
    # construct co-ordinate map of possible locations excluding the diagonal
    rows, cols = np.where(~np.eye(num_variates, dtype=bool))
    num_slots = len(rows) # n^2 - n
    num_selected = round(density * num_slots)
    for i in range(num_lags):
        selected = rng.choice(
            num_slots,
            size=num_selected,
            replace=False,
        )

        # Generate coefficients only for the selected interactions.
        B[i, rows[selected], cols[selected]] = rng.uniform(
            low,
            high,
            size=num_selected,
        )

        assert np.all(np.diag(B[i]) == 0.0), "B diagonal must be zero."

    assert B.shape == (
        num_lags,
        num_variates,
        num_variates,
    )
    return B

def make_A_stack(
    D: np.ndarray,
    B: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """
    Construct VAR(p) coefficient matrices.
 
    A_i(lambda) = D_i + lambda * c * B_i
    Returns:
        shape = (num_lags, num_variates, num_variates)
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be between 0 and 1.")
    
    if D.shape != B.shape:
        raise ValueError("D and B must have the same shape.")

    return D + lambda_ * B

def matrix_spectral_radius(A: np.ndarray) -> float:
    """
    Return the spectral radius of A.
    r(A) = max_i | mu_i(A) |
    A VAR(1) process is stable when r(A) < 1.
    """

    eigenvalues = np.linalg.eigvals(A)
    return float(np.max(np.abs(eigenvalues)))

def companion_matrix(A: np.ndarray) -> np.ndarray:
    """
    Construct the VAR(p) companion matrix.

    A shape:
        (num_lags, num_variates, num_variates)
    """

    num_lags, num_variates, _ = A.shape
    size = num_lags * num_variates

    C = np.zeros((size, size), dtype=np.float64)
    # First block row
    C[:num_variates, :] = np.hstack(A)

    # Lower identity blocks
    for lag in range(1, num_lags):
        start_row = lag * num_variates
        start_col = (lag - 1) * num_variates

        C[
            start_row:start_row + num_variates,
            start_col:start_col + num_variates,
        ] = np.eye(num_variates)

    return C


def scale_B_to_radius(D, B, lambdas, max_radius=0.95):
    
    A = make_A_stack(D, B * 0.0, lambda_=0.0)
    radius = matrix_spectral_radius(companion_matrix(A))

    print(radius)
    
    c = 1.0
    while c > 0.0:
        stable = True
        for lambda_ in lambdas:
            A = make_A_stack(D, B * c, lambda_)
            companion = companion_matrix(A)
            r = matrix_spectral_radius(companion)
            if r > max_radius:
                stable = False
                break
        if stable:
            print(
                f"Selected c={c:.4f}, "
                f"max radius={r:.4f}"
            )
            return B * c
        else:
            c = c - 0.05
    raise ValueError("No stable c found")

def nonlinear_transform(
    x: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "sign":
        return np.sign(x)
    if mode == "tanh":
        return np.tanh(x)
    if mode == "sqrt":
        return np.sign(x) * np.sqrt(np.abs(x))
    if mode == "clip":
        return np.clip(x, -1.0, 1.0)
    if mode == "flip":
        return -np.sign(x) * np.abs(x)

    raise ValueError(f"Unknown mode: {mode}")


def save_var(
    args: argparse.Namespace,
    X: np.ndarray,
    lambda_: float,
    gamma: float,
    persistence: float,
    D: np.ndarray,
    B: np.ndarray,
    lags: list[int],
    burn_in: int,
    replicate_number: int,
) -> Path:
    output_dir = (args.outdir / date.today().isoformat() / f"var{args.max_lag}_{args.interaction_type}")
    
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        output_dir / (
            f"rep{replicate_number:03d}_"
            f"lambda{lambda_:.2f}_"
            f"persistence{persistence:.2f}_"
            f"gamma{gamma:.2f}.npz"
        )
    )
    
    np.savez_compressed(
        filename,
        max_lag=args.max_lag,
        X = X,
        lambda_= lambda_,
        gamma = gamma,
        persistence = persistence,
        D = D,
        B = B,
        lags = np.array(lags, dtype=np.int32),
        burn_in = burn_in,
        interaction_type = args.interaction_type,
        replicate_number = replicate_number,
    )

    return filename

def simulate_var(
    args: argparse.Namespace,
    rng: np.random.Generator,
    D: np.ndarray,
    B: np.ndarray,
    lambda_: float,
    lags: list[int],
    innovations: np.ndarray,
    burn_in: int,
    gamma: float,
    persistence: float,
    replicate_number: int
) -> np.ndarray:
    """
    Simulate independent VAR(1) sequences.
    x_t = A @ x_(t-1) + epsilon_t
    epsilon_t ~ N(0, sigma^2 I)

    Returns
    -------
    X : np.ndarray
    Shape: (num_sequences, sequence_length, num_variates)
    """
    
    num_sequences, total_length, num_variates = innovations.shape

    X = np.zeros(
        (num_sequences, total_length, num_variates),
        dtype=np.float64,
    )

    max_lag = max(lags)
    hidden_state = 0
    hidden_states = np.zeros(total_length, dtype=np.int8)

    if args.regime_switching:
        for t in range(total_length):
            hidden_states[t] = hidden_state

            if rng.random() > persistence:
                hidden_state = 1 - hidden_state


    for t in range(max_lag, total_length):
        state = np.zeros((num_sequences, num_variates), dtype=np.float64)
        if args.regime_switching:
            gamma_ = float(hidden_states[t])
            lambda__ = float(hidden_states[t])
        else:
            gamma_ = gamma
            lambda__ = lambda_
            

        for matrix_idx, lag in enumerate(lags):
            x_lag = X[:, t - lag, :]
            memory = (x_lag @ D[matrix_idx].T)

            interaction_input = ((1.0 - gamma_) * x_lag + gamma_ * nonlinear_transform(x_lag, args.interaction_type))
            interaction = (interaction_input @ B[matrix_idx].T)
            state += (memory + lambda__ * interaction)

        X[:, t, :] = (
        state
        + innovations[:, t, :]
        )
    

        # Remove artificial initialisation period
    if False:
        x = X[0, :, 0]
        x0 = x[hidden_states == 0]
        x1 = x[hidden_states == 1]

        print(x0.mean(), x0.std())
        print(x1.mean(), x1.std())
        print(stats.kurtosis(x0))
        print(stats.kurtosis(x1))
    
        plt.figure(figsize=(8, 5))

        plt.hist(
            x0,
            bins=100,
            density=True,
            alpha=0.5,
            label="State 0 (Linear)",
            color="steelblue",
        )

        plt.hist(
            x1,
            bins=100,
            density=True,
            alpha=0.5,
            label="State 1 (Nonlinear)",
            color="tomato",
        )

        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Distribution by Hidden Regime")
        plt.legend()

        plt.show()

    
    X = X[:, burn_in:, :]
    if args.outdir is not None:
            filename = save_var(args, X, lambda_, gamma, persistence, D, B, lags, burn_in, replicate_number)
            print(f"Saved {filename}")
    return X

def plot_sequence(
    X: np.ndarray,
    lambda_: float,
    gamma_: float,
    persistence_: float,
    max_lag: int,
    sequence: int = 0,
):
    """Plot all variates for one simulated sequence."""

    num_variates = X.shape[2]
    fig, axes = plt.subplots(
        num_variates,
        1,
        figsize=(12, 10),
        sharex=True,
    )
    for i, ax in enumerate(axes):
        ax.plot(X[sequence, :, i])
        ax.set_ylabel(f"x{i + 1}")
        
    axes[-1].set_xlabel("Time")
    
    fig.suptitle(
        f"VAR({max_lag}) sequence {sequence}, lambda={lambda_:.2f}, gamma={gamma_:.2f}, persistence={persistence_:.2f}"
    )
    
    plt.tight_layout()
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt


def plot_histogram(
    X: np.ndarray,
    lambda_: float,
    gamma: float,
    persistence: float,
    variate: int = 1,
):
    """
    Plot the distribution of a single variate.
    """

    values = X[:, :, variate].ravel()

    plt.figure(figsize=(8, 5))

    sns.histplot(
        values,
        bins=100,
        stat="density",
        kde=True,
    )

    plt.title(
        f"Variate {variate + 1} distribution "
        f"(lambda={lambda_:.2f}, gamma={gamma:.2f}, persistence={persistence:.2f})"
    )

    plt.xlabel("Value")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()

def plot_heatmap(data):
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="viridis",
    )
    plt.title("heatmap")
    plt.tight_layout()
    plt.show()


def active_lags(max_lag: int) -> list[int]:
    lags = [1]
    lag = 2
    while lag <= max_lag:
        lags.append(lag)
        lag *= 2
    return lags

def scale_D_to_radius(
    D: np.ndarray,
    target_radius: float = 0.8,
) -> np.ndarray:
    """
    Scale a D stack so the companion matrix has the
    desired spectral radius.
    """
    scale = 1.0
    while scale > 0:    
        D_scaled = D * scale
        radius = matrix_spectral_radius(companion_matrix(D_scaled))
        if radius <= target_radius:
            print(f"D spectral radius={radius:.4f}")
            return D_scaled
        scale -= 0.01
    raise ValueError("Could not stabilize D.")


def make_replicate(args, rng_dgp, rng_noise, replicate_number):
    lags = active_lags(args.max_lag)
    D = make_D_stack(len(lags), args.num_variates, rng_dgp)
    # need to scale D
    D = scale_D_to_radius(D)
    B = make_B_stack(len(lags), args.num_variates, rng_dgp)
    B = scale_B_to_radius(D, B, args.lambdas)

    innovations = rng_noise.normal(
        loc=0.0,
        scale=args.sigma,
        size=(
            args.num_sequences,
            args.sequence_length + args.burn_in,
            args.num_variates,
        ),
    )
    
    results = []
    for lambda_ in args.lambdas:
        for persistence_ in args.persistences:
            for gamma_ in args.gammas:
                A = make_A_stack(D, B, lambda_)
                spectral_radius = matrix_spectral_radius(companion_matrix(A))
                assert (spectral_radius < 1.0), "Generated Matrix A is not stable."

                #print(
                #    f"replicate={replicate_number:.2f}, "
                #    f"lambda={lambda_:.2f}, "
                #    f"gamma={gamma_:.2f}, "
                #    f"persistence_={persistence_:.2f}, "
                #    f"spectral radius={spectral_radius:.4f}"
                #)

                X = simulate_var(args, rng_dgp, D, B, lambda_, lags, innovations, args.burn_in, gamma_, persistence_, replicate_number)
                if gamma_== 1.0 and lambda_ == 1.0 and persistence_ == 0.9:
                    plot_sequence(X, lambda_, gamma_, persistence_, args.max_lag)
                    plot_histogram(X, lambda_, gamma_, persistence_)
                kurt = stats.kurtosis(
                    X,
                    axis=None,
                    fisher=True,
                    bias=False,
                )

                results.append({
                    "replicate": replicate_number,
                    "lambda": lambda_,
                    "gamma": gamma_,
                    "persistence": persistence_,
                    "mean": X.mean(),
                    "std": X.std(),
                    "min": X.min(),
                    "max": X.max(),
                    "kurtosis": kurt,
                })
                print(
                    f"replicate={replicate_number:.2f}, "
                    f"lambda={lambda_:.2f}, "
                    f"gamma={gamma_:.2f}, "
                    f"persistence={persistence_:.2f}, "
                    f"spectral radius={spectral_radius:.4f}",
                    f"kurtosis={kurt:.4f}"
                )
    df = pd.DataFrame(results)
    if args.regime_switching:
        std_matrix = df.pivot(
            index="lambda",
            columns="persistence",
            values="std",
        )
    else:
        std_matrix = df.pivot(
            index="lambda",
            columns="gamma",
            values="std",
        )
    plot_heatmap(std_matrix)



if __name__ == "__main__":
    args = parse_args()

    print(args)
    rng_dgp = np.random.default_rng(args.seed)
    rng_noise = np.random.default_rng(args.seed + 1)
    
    print(f"Running regime switching: {args.regime_switching}")
    
    if args.regime_switching:
        args.gammas = [1.0]
        args.lambdas = [1.0]    
    else:
        args.persistences = [1.0]
        
    print(f"Persistences: {args.persistences}")
    print(f"Gammas: {args.gammas}")
    print(f"Lambdas: {args.lambdas}")

    for replicate_number in range(args.num_replicates):
        make_replicate(args, rng_dgp, rng_noise, replicate_number)
