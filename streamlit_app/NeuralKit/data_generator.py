import numpy as np
from typing import Tuple, Optional, List, Dict 

class SpiralDataset:
    """
    Generate a multi-class 2D spiral dataset.

    Geometry (per class k):
        We sample a radius r ∈ [r_min, r_max] and an angle θ that sweeps
        a sector of the spiral assigned to class k, then map (r, θ) to Cartesian:

            x = r * cos(θ)
            y = r * sin(θ)

        The angle is perturbed with Gaussian noise for non-linearity:
            θ_noise = θ + N(0, σ^2)

        If you prefer the Archimedean spiral form r = a + bθ, note that this
        implementation is equivalent after rescaling because r increases
        linearly while θ sweeps linearly:
            b ≈ (r_max - r_min) / (θ_max - θ_min)

    Parameters
    ----------
    points_per_class : int
        Number of points in each class.
    n_classes : int
        Number of spiral arms (classes).
    noise : float, default=0.2
        Standard deviation of Gaussian noise added to the angle θ.
    turns : float, default=2.0
        How many full turns the whole spiral makes across all classes.
        (2.0 → ~4π radians total span)
    radius : tuple[float, float], default=(0.0, 1.0)
        Inclusive range for radius sampling per arm.
    random_state : int | None, default=None
        Seed for reproducibility.

    Returns (via .generate())
    -------------------------
    X : np.ndarray, shape (points_per_class * n_classes, 2)
        2D coordinates.
    y : np.ndarray, shape (points_per_class * n_classes,)
        Integer labels in [0, n_classes-1].
    """

    def __init__(self, points_per_class: int, n_classes: int,
                    noise: float = 0.2, turns: float = 2.0, 
                    radius: Tuple[float, float] = (0.0, 1.0),
                    random_state: Optional[int] = None) -> None:
        if points_per_class <= 0:
            raise ValueError("points_per_class must be > 0.")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0.")
        if radius[1] <= radius[0]:
            raise ValueError("radius must satisfy radius[1] > radius[0].")

        self.points_per_class = points_per_class
        self.n_classes = n_classes
        self.noise = noise
        self.turns = turns
        self.radius = radius
        self.rng = np.random.default_rng(random_state)

        n_total = points_per_class * n_classes

        self.X = np.zeros((n_total, 2), dtype=float) # 2D coordinates
        self.y = np.zeros(n_total, dtype=np.uint8) # Labels in [0, n_classes-1]

        # Total angle span (in radians) across all classes
        theta_total = 1.0 * np.pi * turns

        for k in range(n_classes):
            # Slice for class k
            idx = slice(k * points_per_class, (k + 1) * points_per_class)

            # Radius increases linearly along each arm
            r = np.linspace(self.radius[0], self.radius[1], points_per_class)

            # Angle sector for class k
            theta_start = (k / n_classes) * theta_total
            theta_end   = ((k + 1) / n_classes) * theta_total
            theta = np.linspace(theta_start, theta_end, points_per_class)

            # Add angular noise
            theta += self.rng.normal(0.0, self.noise, points_per_class)

            # Polar → Cartesian (x = r cosθ, y = r sinθ)
            self.X[idx, 0] = r * np.cos(theta)
            self.X[idx, 1] = r * np.sin(theta)
            self.y[idx] = k

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y)."""
        return self.X, self.y

class CircleDataset:
    """
    Generate a 2D dataset where each class corresponds to a noisy circle.

    For class k (k = 0..C-1):
        • Center is fixed at the origin (0,0).
        • Radius = r_k (user-defined sequence or default linear scaling).
        • Sample N angular positions uniformly in [0, 2π).
        • Add Gaussian noise to radius for non-perfect circles.

    Mathematical form:
        θ_i ~ Uniform(0, 2π)
        ρ_i = r_k + N(0, σ^2)
        x_i = ρ_i cos θ_i
        y_i = ρ_i sin θ_i

    Parameters
    ----------
    points_per_class : int
        Number of points per circle (N).
    n_classes : int
        Number of concentric circles (C).
    base_radius : float, default=1.0
        Radius of the innermost circle.
    radius_step : float, default=1.0
        Increase in radius per class (so r_k = base_radius + k*radius_step).
    noise_std : float, default=0.05
        Standard deviation of Gaussian radial noise.
    random_state : int | None, default=None
        Seed for reproducibility.

    Returns (via .generate())
    -------------------------
    X : np.ndarray, shape (points_per_class * n_classes, 2)
        2D coordinates.
    y : np.ndarray, shape (points_per_class * n_classes,)
        Integer labels in [0, n_classes-1].
    """

    def __init__(
        self,
        points_per_class: int,
        n_classes: int,
        base_radius: float = 1.0,
        radius_step: float = 1.0,
        noise_std: float = 0.05,
        random_state: Optional[int] = None,
    ) -> None:
        if points_per_class <= 0:
            raise ValueError("points_per_class must be > 0.")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0.")
        if base_radius <= 0 or radius_step <= 0:
            raise ValueError("base_radius and radius_step must be > 0.")

        self.points_per_class = points_per_class
        self.n_classes = n_classes
        self.base_radius = base_radius
        self.radius_step = radius_step
        self.noise_std = noise_std
        self.rng = np.random.default_rng(random_state)

        n_total = points_per_class * n_classes
        self.X = np.zeros((n_total, 2), dtype=float)
        self.y = np.zeros(n_total, dtype=np.uint8)

        for k in range(n_classes):
            idx = slice(k * points_per_class, (k + 1) * points_per_class)

            # Angles equally spaced
            theta = np.linspace(0, 2 * np.pi, points_per_class, endpoint=False)

            # Circle radius with Gaussian noise
            radius = self.base_radius + k * self.radius_step
            rho = radius + self.rng.normal(0.0, self.noise_std, points_per_class)

            # Polar → Cartesian
            self.X[idx, 0] = rho * np.cos(theta)
            self.X[idx, 1] = rho * np.sin(theta)
            self.y[idx] = k

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y)."""
        return self.X, self.y

class LineDataset:
    """
    Generate a multi-class line dataset in 2D.

    Each class corresponds to a line of the form:
        y = a_k * x + b
    where
        - a_k is the slope assigned to class k
        - b is a Gaussian random offset

    Construction:
        • x is sampled uniformly from [-10, 10] using linspace
        • y is generated using the line equation with noise
        • Labels y[i] = k for all points of class k

    Parameters
    ----------
    points_per_class : int
        Number of points per class.
    n_classes : int
        Number of lines (classes).
    n_dimensions : int
        Dimensionality of the dataset (currently only supports 2).
    noise_scale : float, default=2.0
        Standard deviation of Gaussian noise added to the intercept b.
    random_state : int | None, default=None
        Seed for reproducibility.

    Returns (via .generate())
    -------------------------
    X : np.ndarray, shape (points_per_class * n_classes, 2)
        Data points (x, y).
    y : np.ndarray, shape (points_per_class * n_classes,)
        Integer labels in [0, n_classes-1].
    """

    def __init__(self, points_per_class: int, n_classes: int,
                    n_dimensions: int = 2, noise_scale: float = 2.0,
                    random_state: Optional[int] = None
    ) -> None:
        if n_dimensions != 2:
            raise ValueError("Currently only 2D datasets are supported.")

        self.points_per_class = points_per_class
        self.n_classes = n_classes
        self.n_dimensions = n_dimensions
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(random_state)

        n_total = points_per_class * n_classes
        self.X = np.zeros((n_total, n_dimensions), dtype=float)
        self.y = np.zeros(n_total, dtype=np.uint8)

        # Generate data for each class
        for k in range(n_classes):
            # Range of indices for class k
            idx = slice(k * points_per_class, (k + 1) * points_per_class)

            # X-axis values evenly spaced
            x = np.linspace(-10, 10, points_per_class)

            # Slope for class k
            slope = 2 * k - 2   # e.g. -2, 0, 2, 4, ...
            # Random Gaussian intercept
            intercept = self.rng.normal(0.0, noise_scale, points_per_class)

            # Compute y = slope * x + intercept
            y_vals = slope * x + intercept

            # Store in dataset
            self.X[idx, 0] = x
            self.X[idx, 1] = y_vals
            self.y[idx] = k

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y)."""
        return self.X, self.y

class ZoneDataset:
    """
    Generate a 2D multi-class 'zone' dataset: each class is a small noisy ring
    centered on points uniformly placed on a big circle.

    Construction
    ------------
    For class k (k = 0..C-1):
        Center on the unit circle (scaled by R_center):
            θ_k = 2π k / C
            c_k = (R_center cos θ_k, R_center sin θ_k)

        Sample a ring around c_k with angular and radial noise:
            t_i = linspace(0, 2π, N) + 𝒩(0, σ_θ²)
            ρ_i = R_ring + 𝒩(0, σ_ρ²)

        Cartesian mapping:
            x_i = c_k.x + ρ_i cos(t_i)
            y_i = c_k.y + ρ_i sin(t_i)

    Parameters
    ----------
    points_per_class : int
        Number of points per class (N).
    n_classes : int
        Number of zones / rings (C).
    center_radius : float, default=1.0
        Radius of the big circle where class centers lie (R_center).
    ring_radius : float, default=0.5
        Mean radius of each local ring around its center (R_ring).
    angular_noise : float, default=0.2
        Std-dev of Gaussian noise added to angle t (σ_θ).
    radial_noise : float, default=0.5
        Std-dev of Gaussian noise added to radius ρ (σ_ρ).
    random_state : int | None, default=None
        Seed for reproducibility.

    Returns (via .generate())
    -------------------------
    X : np.ndarray, shape (points_per_class * n_classes, 2)
        2D data points.
    y : np.ndarray, shape (points_per_class * n_classes,)
        Integer labels in [0, n_classes-1].
    """

    def __init__(self, points_per_class: int,
                    n_classes: int, center_radius: float = 1.0,
                    ring_radius: float = 0.5, angular_noise: float = 0.2,
                    radial_noise: float = 0.5, 
                    random_state: Optional[int] = None,) -> None:
        if points_per_class <= 0:
            raise ValueError("points_per_class must be > 0.")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0.")
        if center_radius <= 0:
            raise ValueError("center_radius must be > 0.")
        if ring_radius <= 0:
            raise ValueError("ring_radius must be > 0.")

        self.points_per_class = points_per_class
        self.n_classes = n_classes
        self.center_radius = center_radius
        self.ring_radius = ring_radius
        self.angular_noise = angular_noise
        self.radial_noise = radial_noise
        self.rng = np.random.default_rng(random_state)

        n_total = points_per_class * n_classes
        self.X = np.zeros((n_total, 2), dtype=float)
        self.y = np.zeros(n_total, dtype=np.uint8)

        # Precompute a clean angular sweep (shared shape) then add noise per class
        base_t = np.linspace(0.0, 2.0 * np.pi, points_per_class)

        for k in range(n_classes):
            idx = slice(k * points_per_class, (k + 1) * points_per_class)

            # Center of class k on the big circle
            theta_k = 2.0 * np.pi * k / n_classes
            cx = center_radius * np.cos(theta_k)
            cy = center_radius * np.sin(theta_k)

            # Noisy angle and radius for the ring
            t = base_t + self.rng.normal(0.0, self.angular_noise, points_per_class)
            rho = self.ring_radius + self.rng.normal(0.0, self.radial_noise, points_per_class)

            # Polar → Cartesian around the class center
            self.X[idx, 0] = cx + rho * np.cos(t)
            self.X[idx, 1] = cy + rho * np.sin(t)
            self.y[idx] = k

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y)."""
        return self.X, self.y

# class Zone_3D:
#     def __init__(self, n_points, n_classes, n_dimensions, centers):
#         self.N = n_points
#         self.D = n_dimensions
#         self.K = n_classes
#         self.centers = centers
#         self.P = np.zeros((self.N * self.K, self.D))
#         self.L = np.zeros(self.N * self.K, dtype='uint8')
#         for j in range(self.K):
#             center = np.array(self.centers[j])
#             ix = range(self.N * j, self.N * (j + 1))
#             R = np.random.randn(self.N) * 2
#             gamma = np.random.uniform(0, np.pi, self.N)
#             theta = np.random.uniform(0, 2 * np.pi, self.N)
#             x = R * np.sin(gamma) * np.cos(theta)
#             y = R * np.sin(gamma) * np.sin(theta)
#             z = R * np.cos(gamma)
#             self.P[ix] = np.c_[center[0] + x, center[1] + y, center[2] + z]
#             self.L[ix] = j

#     def generate(self):
#         return self.P, self.L

class PolynomialDataset:
    """
    Generate synthetic 1D data from a polynomial with additive Gaussian noise.

    Mathematical form:
        y(x) = c_0 + c_1 x + c_2 x^2 + ... + c_d x^d + ε
    where
        • coefficients = [c_0, c_1, ..., c_d]
        • ε ~ N(0, σ^2)

    Parameters
    ----------
    n_points : int, default=200
        Number of sampled points along the x-axis.
    coefficients : list[float], default=[1, -2, 3]
        Polynomial coefficients in ascending order
        [c_0, c_1, ..., c_d].
    noise_std : float, default=5.0
        Standard deviation of Gaussian noise ε.

    Methods
    -------
    generate() -> dict
        Return {"x": np.ndarray, "y": np.ndarray} synthetic data.
    save_to_csv(file_name: str) -> None
        Save generated data to a CSV file with header "x,y".
    print_equation() -> None
        Print the actual polynomial equation in human-readable form.
    """

    def __init__(self, n_points: int = 200, 
                 coefficients: List[float] = [1, -2, 3], 
                 noise_std: float = 5.0) -> None:
        self.n_points = n_points
        self.coefficients = coefficients
        self.noise_std = noise_std
        self.X = None
        self.y = None

    def generate(self) -> Dict[str, np.ndarray]:
        """Generate noisy polynomial data."""
        x = np.linspace(-10, 10, self.n_points)

        # Evaluate polynomial y = Σ c_i * x^i
        y = np.zeros_like(x, dtype=float)
        for i, c in enumerate(self.coefficients):
            y += c * x**i

        # Add Gaussian noise
        y += np.random.normal(0, self.noise_std, self.n_points)

        self.X = x.reshape(-1, 1)  # Reshape for consistency
        self.y = y
        return self.X, self.y

    def save_to_csv(self, file_name: str = "polynomial_data.csv") -> None:
        """Save generated data to CSV with columns (x, y)."""
        data = self.generate()
        combined = np.column_stack((data["x"], data["y"]))
        np.savetxt(file_name, combined, delimiter=",", header="x,y", comments="")

    def print_equation(self) -> None:
        """Pretty-print the actual polynomial equation."""
        terms = []
        for i, c in enumerate(self.coefficients):
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{c}*x")
            else:
                terms.append(f"{c}*x^{i}")
        equation = " + ".join(terms)
        print(f"Actual polynomial: y(x) = {equation}")
