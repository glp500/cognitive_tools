from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
ANIMATIONS_DIR = RESULTS_DIR / "animations"
FRAMES_DIR = RESULTS_DIR / "frames"


# ---------------------------------------------------------------------
# Experimental design
# ---------------------------------------------------------------------

# High and low abundance differ in mean carrying capacity.
# The spatial pattern itself is kept comparable across abundance levels.
ABUNDANCE_LEVELS = {
    "high": 0.75,
    "low": 0.35,
}

# These families differ in spatial organization and/or heterogeneity.
PATTERN_FAMILIES = {
    "uniform": {
        "label": "Uniform",
        "heterogeneity_strength": 0.00,
    },
    "weak_patchy": {
        "label": "Weak patchy",
        "heterogeneity_strength": 0.12,
    },
    "patchy": {
        "label": "Patchy",
        "heterogeneity_strength": 0.28,
    },
    "centralized": {
        "label": "Centralized",
        "heterogeneity_strength": 0.35,
    },
    "decentralized": {
        "label": "Decentralized",
        "heterogeneity_strength": 0.35,
    },
    "fragmented": {
        "label": "Fragmented",
        "heterogeneity_strength": 0.42,
    },
}


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------


def ensure_directories() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path}")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def gini(field: np.ndarray) -> float:
    """Gini coefficient of non-negative tile values."""

    values = np.sort(np.asarray(field, dtype=float).ravel())
    total = values.sum()

    if total <= 0:
        return 0.0

    n = len(values)
    index = np.arange(1, n + 1)

    value = 2 * np.sum(index * values) / (n * total) - (n + 1) / n

    if abs(value) < 1e-12:
        return 0.0

    return float(value)


def morans_i(field: np.ndarray) -> float:
    """
    Moran's I using four-neighbour adjacency and non-wrapping boundaries.

    Positive values mean similar resource values tend to be next to each
    other. Negative values mean neighbouring values tend to be dissimilar.

    Moran's I is undefined for a perfectly uniform field because the field
    has zero variance. In that case this function returns np.nan.
    """

    values = np.asarray(field, dtype=float)
    mean = float(values.mean())
    deviations = values - mean

    denominator = float(np.sum(deviations**2))

    if denominator <= 1e-15:
        return float("nan")

    height, width = values.shape
    numerator = 0.0
    weight_sum = 0.0

    neighbours = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    for y in range(height):
        for x in range(width):
            for dy, dx in neighbours:
                yy = y + dy
                xx = x + dx

                if 0 <= yy < height and 0 <= xx < width:
                    numerator += deviations[y, x] * deviations[yy, xx]
                    weight_sum += 1.0

    n = values.size

    return float(
        (n / weight_sum)
        * (numerator / denominator)
    )


def scarcity_fraction(
    resource: np.ndarray,
    capacity: np.ndarray,
    threshold: float = 0.20,
) -> float:
    return float(
        np.mean(
            resource
            < threshold * capacity
        )
    )


def resource_cv(field: np.ndarray) -> float:
    mean = float(np.mean(field))

    if mean <= 0:
        return 0.0

    return float(
        np.std(field) / mean
    )


# ---------------------------------------------------------------------
# Spatial pattern helpers
# ---------------------------------------------------------------------


def smooth_field(
    field: np.ndarray,
    passes: int,
) -> np.ndarray:
    """Neighbour smoothing used to produce spatially autocorrelated patches."""

    out = field.copy()

    for _ in range(passes):
        padded = np.pad(
            out,
            1,
            mode="edge",
        )

        centre = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]

        out = (
            4 * centre
            + up
            + down
            + left
            + right
        ) / 8.0

    return out


def rescale_pattern(
    raw: np.ndarray,
    mean_capacity: float,
    heterogeneity_strength: float,
    min_capacity: float = 0.05,
    max_capacity: float = 1.0,
) -> np.ndarray:
    """
    Scale a spatial pattern around the requested mean carrying capacity.
    """

    if np.allclose(raw, raw.flat[0]):
        return np.full_like(
            raw,
            mean_capacity,
            dtype=float,
        )

    centered = raw - raw.mean()
    max_abs = np.max(np.abs(centered))

    if max_abs > 0:
        centered = centered / max_abs

    amplitude = heterogeneity_strength * mean_capacity
    capacity = mean_capacity + amplitude * centered
    capacity = np.clip(
        capacity,
        min_capacity,
        max_capacity,
    )

    # Re-centre after clipping so the requested abundance level remains
    # approximately comparable across pattern families.
    capacity += mean_capacity - capacity.mean()
    capacity = np.clip(
        capacity,
        min_capacity,
        max_capacity,
    )

    return capacity


def gaussian_hotspot(
    width: int,
    height: int,
    centre_x: float,
    centre_y: float,
    sigma: float,
) -> np.ndarray:
    y, x = np.indices((height, width))
    distance_squared = (
        (x - centre_x) ** 2
        + (y - centre_y) ** 2
    )

    return np.exp(
        -distance_squared
        / (2.0 * sigma**2)
    )


def make_capacity_map(
    family: str,
    mean_capacity: float,
    width: int,
    height: int,
    seed: int,
) -> np.ndarray:
    """Create one carrying-capacity pattern family."""

    rng = np.random.default_rng(seed)
    strength = PATTERN_FAMILIES[family]["heterogeneity_strength"]

    if family == "uniform":
        raw = np.ones(
            (height, width),
            dtype=float,
        )

    elif family == "weak_patchy":
        raw = smooth_field(
            rng.random((height, width)),
            passes=10,
        )

    elif family == "patchy":
        raw = smooth_field(
            rng.random((height, width)),
            passes=5,
        )

    elif family == "fragmented":
        raw = smooth_field(
            rng.random((height, width)),
            passes=2,
        )

    elif family == "centralized":
        raw = gaussian_hotspot(
            width=width,
            height=height,
            centre_x=(width - 1) / 2.0,
            centre_y=(height - 1) / 2.0,
            sigma=min(width, height) / 4.0,
        )

    elif family == "decentralized":
        raw = (
            gaussian_hotspot(
                width,
                height,
                width * 0.25,
                height * 0.25,
                sigma=1.6,
            )
            + gaussian_hotspot(
                width,
                height,
                width * 0.75,
                height * 0.25,
                sigma=1.6,
            )
            + gaussian_hotspot(
                width,
                height,
                width * 0.25,
                height * 0.75,
                sigma=1.6,
            )
            + gaussian_hotspot(
                width,
                height,
                width * 0.75,
                height * 0.75,
                sigma=1.6,
            )
        )

    else:
        raise ValueError(
            f"Unknown family: {family}"
        )

    return rescale_pattern(
        raw=raw,
        mean_capacity=mean_capacity,
        heterogeneity_strength=strength,
    )


# ---------------------------------------------------------------------
# Ecology-only simulation
# ---------------------------------------------------------------------


class EcologySimulation:
    """Small ecology-only simulator used by this pattern comparison."""

    def __init__(
        self,
        capacity: np.ndarray,
        *,
        equilibrium_fraction: float,
        regeneration_rate: float,
        coupling_rate: float,
        initial_fraction: float,
    ):
        self.capacity = np.asarray(
            capacity,
            dtype=float,
        )

        self.height, self.width = self.capacity.shape

        self.equilibrium_fraction = equilibrium_fraction
        self.regeneration_rate = regeneration_rate
        self.coupling_rate = coupling_rate

        self.depletion_rate = (
            regeneration_rate
            * (1.0 - equilibrium_fraction)
        )

        self.resource = (
            initial_fraction
            * self.capacity
        )

        self.time = 0

    def _neighbor_mean(
        self,
        field: np.ndarray,
    ) -> np.ndarray:
        padded = np.pad(
            field,
            1,
            mode="edge",
        )

        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]

        return (
            up
            + down
            + left
            + right
        ) / 4.0

    def step(self) -> None:
        r = self.regeneration_rate
        d = self.depletion_rate
        c = self.coupling_rate

        resource = self.resource
        capacity = self.capacity
        safe_capacity = np.maximum(
            capacity,
            1e-12,
        )

        regeneration = (
            r
            * resource
            * (1.0 - resource / safe_capacity)
        )

        depletion = (
            d * resource
        )

        diffusion = (
            c
            * (
                self._neighbor_mean(resource)
                - resource
            )
        )

        next_resource = (
            resource
            + regeneration
            - depletion
            + diffusion
        )

        # Preserve the current scan model's 0-1 resource bounds.
        next_resource = np.clip(
            next_resource,
            0.0,
            1.0,
        )

        self.resource = next_resource
        self.time += 1

    def measure(self) -> dict:
        return {
            "time": self.time,
            "mean_capacity": float(
                np.mean(self.capacity)
            ),
            "capacity_gini": gini(
                self.capacity
            ),
            "capacity_morans_i": morans_i(
                self.capacity
            ),
            "mean_resource": float(
                np.mean(self.resource)
            ),
            "resource_fraction": float(
                np.mean(
                    self.resource
                    / np.maximum(
                        self.capacity,
                        1e-12,
                    )
                )
            ),
            "resource_gini": gini(
                self.resource
            ),
            "resource_morans_i": morans_i(
                self.resource
            ),
            "resource_cv": resource_cv(
                self.resource
            ),
            "scarcity_fraction": scarcity_fraction(
                self.resource,
                self.capacity,
            ),
        }


# ---------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------


def build_scenarios() -> list[dict]:
    scenarios = []

    for abundance_name, mean_capacity in ABUNDANCE_LEVELS.items():
        for family_name, family_spec in PATTERN_FAMILIES.items():
            scenarios.append(
                {
                    "scenario": f"{family_name}_{abundance_name}",
                    "family": family_name,
                    "family_label": family_spec["label"],
                    "abundance": abundance_name,
                    "mean_capacity": mean_capacity,
                }
            )

    return scenarios


def scenario_seed(
    family: str,
    base_seed: int,
) -> int:
    """
    Use the same spatial realization for high and low abundance.

    This makes abundance comparisons cleaner: patchy_high and patchy_low
    have the same pattern geometry, scaled around different mean capacities.
    """

    family_offset = (
        list(PATTERN_FAMILIES.keys())
        .index(family)
        * 1000
    )

    return base_seed + family_offset


# ---------------------------------------------------------------------
# Running scenarios
# ---------------------------------------------------------------------


def make_simulation(
    spec: dict,
    *,
    width: int,
    height: int,
    seed: int,
    equilibrium_fraction: float,
    regeneration_rate: float,
    coupling_rate: float,
    initial_fraction: float,
) -> EcologySimulation:
    capacity = make_capacity_map(
        family=spec["family"],
        mean_capacity=spec["mean_capacity"],
        width=width,
        height=height,
        seed=scenario_seed(
            spec["family"],
            seed,
        ),
    )

    return EcologySimulation(
        capacity=capacity,
        equilibrium_fraction=equilibrium_fraction,
        regeneration_rate=regeneration_rate,
        coupling_rate=coupling_rate,
        initial_fraction=initial_fraction,
    )


def run_scenario(
    spec: dict,
    *,
    width: int,
    height: int,
    steps: int,
    seed: int,
    equilibrium_fraction: float,
    regeneration_rate: float,
    coupling_rate: float,
    initial_fraction: float,
):
    sim = make_simulation(
        spec,
        width=width,
        height=height,
        seed=seed,
        equilibrium_fraction=equilibrium_fraction,
        regeneration_rate=regeneration_rate,
        coupling_rate=coupling_rate,
        initial_fraction=initial_fraction,
    )

    timeseries = [
        {
            **spec,
            **sim.measure(),
        }
    ]

    for _ in range(steps):
        sim.step()
        timeseries.append(
            {
                **spec,
                **sim.measure(),
            }
        )

    summary = {
        **spec,
        **sim.measure(),
    }

    return sim, timeseries, summary


def run_full_scan(
    *,
    width: int,
    height: int,
    steps: int,
    seed: int,
    equilibrium_fraction: float,
    regeneration_rate: float,
    coupling_rate: float,
    initial_fraction: float,
):
    final_models: dict[str, EcologySimulation] = {}
    all_timeseries: list[dict] = []
    all_summaries: list[dict] = []

    for spec in build_scenarios():
        sim, timeseries, summary = run_scenario(
            spec,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            equilibrium_fraction=equilibrium_fraction,
            regeneration_rate=regeneration_rate,
            coupling_rate=coupling_rate,
            initial_fraction=initial_fraction,
        )

        final_models[spec["scenario"]] = sim
        all_timeseries.extend(timeseries)
        all_summaries.append(summary)

    return (
        final_models,
        all_timeseries,
        all_summaries,
    )


# ---------------------------------------------------------------------
# Static figure helpers
# ---------------------------------------------------------------------


def plot_catalog(
    models: dict[str, EcologySimulation],
    *,
    field_name: str,
    filename: str,
    title: str,
    colorbar_label: str,
) -> None:
    families = list(PATTERN_FAMILIES.keys())
    abundances = ["high", "low"]

    fig, axes = plt.subplots(
        len(abundances),
        len(families),
        figsize=(2.8 * len(families), 5.8),
        constrained_layout=True,
    )

    image = None

    for row_i, abundance in enumerate(abundances):
        for col_i, family in enumerate(families):
            scenario = f"{family}_{abundance}"
            sim = models[scenario]
            field = getattr(sim, field_name)

            ax = axes[row_i, col_i]
            image = ax.imshow(
                field,
                origin="upper",
                vmin=0.0,
                vmax=1.0,
            )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{PATTERN_FAMILIES[family]['label']}\n"
                f"{abundance} abundance",
                fontsize=10,
            )

    fig.suptitle(
        title,
        fontsize=17,
    )

    if image is not None:
        fig.colorbar(
            image,
            ax=axes,
            label=colorbar_label,
            shrink=0.82,
        )

    path = RESULTS_DIR / filename
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved {path}")


def plot_factor_trajectories(
    rows: list[dict],
    *,
    metric: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Two-panel trajectory figure: high abundance vs low abundance."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
        constrained_layout=True,
    )

    for ax, abundance in zip(
        axes,
        ["high", "low"],
    ):
        for family in PATTERN_FAMILIES.keys():
            subset = [
                row
                for row in rows
                if (
                    row["family"] == family
                    and row["abundance"] == abundance
                )
            ]

            subset = sorted(
                subset,
                key=lambda row: row["time"],
            )

            x_values = [
                row["time"]
                for row in subset
            ]

            y_values = np.asarray(
                [
                    row[metric]
                    for row in subset
                ],
                dtype=float,
            )

            # Uniform landscapes have no variance, so Moran's I is undefined.
            # Matplotlib naturally leaves that line blank when values are NaN.
            ax.plot(
                x_values,
                y_values,
                label=PATTERN_FAMILIES[family]["label"],
            )

        ax.set_title(
            f"{abundance.capitalize()} abundance"
        )
        ax.set_xlabel("Time step")
        ax.grid(alpha=0.25)
        ax.margins(x=0)

        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(ylabel)

    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Pattern family",
    )

    path = RESULTS_DIR / filename
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved {path}")


def plot_three_factor_state_space(
    summary_rows: list[dict],
) -> None:
    """
    Three linked 2D views of abundance, inequality, and spatial organization.
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        constrained_layout=True,
    )

    for row in summary_rows:
        label = row["scenario"]
        abundance = row["mean_resource"]
        inequality = row["resource_gini"]
        spatial = row["resource_morans_i"]

        axes[0].scatter(
            inequality,
            abundance,
            s=65,
        )
        axes[0].annotate(
            label,
            (inequality, abundance),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

        if np.isfinite(spatial):
            axes[1].scatter(
                spatial,
                abundance,
                s=65,
            )
            axes[1].annotate(
                label,
                (spatial, abundance),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

            axes[2].scatter(
                spatial,
                inequality,
                s=65,
            )
            axes[2].annotate(
                label,
                (spatial, inequality),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    axes[0].set_xlabel("Resource Gini")
    axes[0].set_ylabel("Mean resource")
    axes[0].set_title("Abundance vs inequality")

    axes[1].set_xlabel("Resource Moran's I")
    axes[1].set_ylabel("Mean resource")
    axes[1].set_title("Abundance vs spatial clustering")

    axes[2].set_xlabel("Resource Moran's I")
    axes[2].set_ylabel("Resource Gini")
    axes[2].set_title("Inequality vs spatial clustering")

    for ax in axes:
        ax.grid(alpha=0.25)

    path = RESULTS_DIR / "06_three_factor_state_space.png"
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved {path}")


def plot_centralized_vs_decentralized(
    models: dict[str, EcologySimulation],
) -> None:
    scenarios = [
        "centralized_high",
        "decentralized_high",
        "centralized_low",
        "decentralized_low",
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7),
        constrained_layout=True,
    )

    image = None

    for row_index, pair in enumerate(
        [
            scenarios[:2],
            scenarios[2:],
        ]
    ):
        for pair_index, scenario in enumerate(pair):
            sim = models[scenario]
            col = pair_index * 2

            axes[row_index, col].imshow(
                sim.capacity,
                origin="upper",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row_index, col].set_title(
                f"{scenario}\ncapacity"
            )

            image = axes[row_index, col + 1].imshow(
                sim.resource,
                origin="upper",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row_index, col + 1].set_title(
                f"{scenario}\nfinal resource"
            )

            for current_ax in (
                axes[row_index, col],
                axes[row_index, col + 1],
            ):
                current_ax.set_xticks([])
                current_ax.set_yticks([])

    fig.suptitle(
        "Centralized vs decentralized ecological patterns",
        fontsize=17,
    )

    if image is not None:
        fig.colorbar(
            image,
            ax=axes,
            label="Capacity / resource level",
            shrink=0.82,
        )

    path = RESULTS_DIR / "07_centralized_vs_decentralized.png"
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------


FACTOR_GIFS = {
    "abundance": {
        "metric": "mean_resource",
        "ylabel": "Mean resource",
        "title": "Abundance",
    },
    "inequality": {
        "metric": "resource_gini",
        "ylabel": "Resource Gini",
        "title": "Resource inequality",
    },
    "spatial_organization": {
        "metric": "resource_morans_i",
        "ylabel": "Moran's I",
        "title": "Spatial organization",
    },
}


def apply_central_shock(
    sim: EcologySimulation,
    *,
    shock_loss: float,
    shock_radius: float,
) -> None:
    y, x = np.indices(
        sim.resource.shape
    )

    centre_x = (
        sim.width - 1
    ) / 2.0

    centre_y = (
        sim.height - 1
    ) / 2.0

    mask = (
        (x - centre_x) ** 2
        + (y - centre_y) ** 2
        <= shock_radius**2
    )

    sim.resource[mask] *= (
        1.0 - shock_loss
    )


def collect_animation_history(
    spec: dict,
    *,
    width: int,
    height: int,
    steps: int,
    seed: int,
    equilibrium_fraction: float,
    regeneration_rate: float,
    coupling_rate: float,
    initial_fraction: float,
    animation_mode: str,
    warmup_steps: int,
    shock_loss: float,
    shock_radius: float,
):
    sim = make_simulation(
        spec,
        width=width,
        height=height,
        seed=seed,
        equilibrium_fraction=equilibrium_fraction,
        regeneration_rate=regeneration_rate,
        coupling_rate=coupling_rate,
        initial_fraction=initial_fraction,
    )

    if animation_mode == "shock":
        for _ in range(warmup_steps):
            sim.step()

        apply_central_shock(
            sim,
            shock_loss=shock_loss,
            shock_radius=shock_radius,
        )

        # Animation time is reset so frame 0 means immediately after shock.
        sim.time = 0

    history = [
        {
            "time": 0,
            "resource": sim.resource.copy(),
            **sim.measure(),
        }
    ]

    for step in range(1, steps + 1):
        sim.step()

        history.append(
            {
                "time": step,
                "resource": sim.resource.copy(),
                **sim.measure(),
            }
        )

    return sim.capacity.copy(), history


def metric_limits(
    history: list[dict],
    metric: str,
) -> tuple[float, float]:
    values = np.asarray(
        [
            row[metric]
            for row in history
        ],
        dtype=float,
    )

    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        return (-0.1, 0.1)

    minimum = float(finite.min())
    maximum = float(finite.max())

    if metric in {
        "mean_resource",
        "resource_gini",
    }:
        minimum = 0.0

    if minimum == maximum:
        pad = max(
            0.05,
            abs(maximum) * 0.10,
        )
    else:
        pad = (
            maximum - minimum
        ) * 0.12

    return (
        minimum - (0.0 if minimum == 0.0 else pad),
        maximum + pad,
    )


def render_factor_frame(
    *,
    capacity: np.ndarray,
    history: list[dict],
    frame_index: int,
    spec: dict,
    factor_name: str,
    animation_mode: str,
) -> np.ndarray:
    factor = FACTOR_GIFS[factor_name]
    metric = factor["metric"]
    current = history[frame_index]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.8),
        constrained_layout=True,
    )

    resource_image = axes[0].imshow(
        current["resource"],
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    axes[0].set_title(
        "Current resource field"
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    fig.colorbar(
        resource_image,
        ax=axes[0],
        label="Resource stock",
        fraction=0.046,
        pad=0.04,
    )

    x_values = [
        row["time"]
        for row in history[: frame_index + 1]
    ]

    y_values = [
        row[metric]
        for row in history[: frame_index + 1]
    ]

    axes[1].plot(
        x_values,
        y_values,
    )

    if np.isfinite(current[metric]):
        axes[1].scatter(
            [current["time"]],
            [current[metric]],
            s=45,
        )

    axes[1].set_xlim(
        0,
        max(1, history[-1]["time"]),
    )

    lower, upper = metric_limits(
        history,
        metric,
    )
    axes[1].set_ylim(
        lower,
        upper,
    )

    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel(factor["ylabel"])
    axes[1].set_title(factor["title"])
    axes[1].grid(alpha=0.25)

    value = current[metric]

    if np.isfinite(value):
        value_text = f"{value:.3f}"
    else:
        value_text = "undefined (uniform field)"

    fig.suptitle(
        f"{spec['scenario']} | {animation_mode} | step {current['time']}\n"
        f"{factor['title']}: {value_text}",
        fontsize=14,
    )

    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(
        fig.canvas.buffer_rgba(),
        dtype=np.uint8,
    )
    frame = frame.reshape(
        height,
        width,
        4,
    )[..., :3]

    plt.close(fig)

    return frame


def save_factor_gif(
    *,
    capacity: np.ndarray,
    history: list[dict],
    spec: dict,
    factor_name: str,
    animation_mode: str,
    frame_every: int,
    fps: int,
    save_frames: bool,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "GIF export requires imageio and pillow. Install them with:\n"
            "pip install imageio pillow"
        ) from exc

    output_name = (
        f"{spec['scenario']}"
        f"_{animation_mode}"
        f"_{factor_name}"
    )

    frame_dir = (
        FRAMES_DIR
        / output_name
    )

    if save_frames:
        frame_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    frame_indexes = list(
        range(
            0,
            len(history),
            max(1, frame_every),
        )
    )

    if frame_indexes[-1] != len(history) - 1:
        frame_indexes.append(
            len(history) - 1
        )

    frames = []

    for frame_number, history_index in enumerate(frame_indexes):
        frame = render_factor_frame(
            capacity=capacity,
            history=history,
            frame_index=history_index,
            spec=spec,
            factor_name=factor_name,
            animation_mode=animation_mode,
        )

        frames.append(frame)

        if save_frames:
            frame_path = (
                frame_dir
                / f"frame_{frame_number:04d}.png"
            )

            imageio.imwrite(
                frame_path,
                frame,
            )

    gif_path = (
        ANIMATIONS_DIR
        / f"{output_name}.gif"
    )

    imageio.mimsave(
        gif_path,
        frames,
        fps=fps,
    )

    print(f"Saved {gif_path}")

    if save_frames:
        print(
            f"Saved PNG frames to {frame_dir}"
        )


def make_three_factor_gifs(
    spec: dict,
    *,
    width: int,
    height: int,
    steps: int,
    seed: int,
    equilibrium_fraction: float,
    regeneration_rate: float,
    coupling_rate: float,
    initial_fraction: float,
    animation_mode: str,
    warmup_steps: int,
    shock_loss: float,
    shock_radius: float,
    frame_every: int,
    fps: int,
    save_frames: bool,
) -> None:
    capacity, history = collect_animation_history(
        spec,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        equilibrium_fraction=equilibrium_fraction,
        regeneration_rate=regeneration_rate,
        coupling_rate=coupling_rate,
        initial_fraction=initial_fraction,
        animation_mode=animation_mode,
        warmup_steps=warmup_steps,
        shock_loss=shock_loss,
        shock_radius=shock_radius,
    )

    for factor_name in FACTOR_GIFS:
        save_factor_gif(
            capacity=capacity,
            history=history,
            spec=spec,
            factor_name=factor_name,
            animation_mode=animation_mode,
            frame_every=frame_every,
            fps=fps,
            save_frames=save_frames,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ecological abundance, inequality, and spatial "
            "organization across multiple patch structures."
        )
    )

    parser.add_argument(
        "--width",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--equilibrium-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--recovery-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--coupling",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--make-factor-gifs",
        type=str,
        default="",
        help=(
            "Scenario name, e.g. patchy_low, centralized_high, "
            "or 'all'. Creates abundance, inequality, and spatial-"
            "organization GIFs."
        ),
    )
    parser.add_argument(
        "--animation-steps",
        type=int,
        default=200,
        help="Number of timesteps shown in each GIF.",
    )
    parser.add_argument(
        "--animation-mode",
        choices=[
            "settling",
            "shock",
        ],
        default="settling",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=400,
        help=(
            "For shock animations, number of steps allowed to reach a "
            "stable state before the shock is applied."
        ),
    )
    parser.add_argument(
        "--shock-loss",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--shock-radius",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--frame-every",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
    )

    args = parser.parse_args()

    ensure_directories()

    # -----------------------------------------------------------------
    # Full static experiment
    # -----------------------------------------------------------------

    (
        models,
        timeseries_rows,
        summary_rows,
    ) = run_full_scan(
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        equilibrium_fraction=args.equilibrium_fraction,
        regeneration_rate=args.recovery_rate,
        coupling_rate=args.coupling,
        initial_fraction=args.initial_fraction,
    )

    write_csv(
        RESULTS_DIR / "patch_scan_timeseries.csv",
        timeseries_rows,
    )

    write_csv(
        RESULTS_DIR / "patch_scan_summary.csv",
        summary_rows,
    )

    plot_catalog(
        models,
        field_name="capacity",
        filename="01_capacity_pattern_catalog.png",
        title="Capacity-pattern families",
        colorbar_label="Carrying capacity",
    )

    plot_catalog(
        models,
        field_name="resource",
        filename="02_final_resource_catalog.png",
        title="Final resource stock after ecological dynamics",
        colorbar_label="Resource stock",
    )

    # Factor 1: abundance
    plot_factor_trajectories(
        timeseries_rows,
        metric="mean_resource",
        ylabel="Mean resource",
        filename="03_abundance_trajectories.png",
        ylim=(0.0, 1.0),
    )

    # Factor 2: inequality
    plot_factor_trajectories(
        timeseries_rows,
        metric="resource_gini",
        ylabel="Resource Gini",
        filename="04_resource_gini_trajectories.png",
        ylim=(0.0, 0.25),
    )

    # Factor 3: spatial organization
    plot_factor_trajectories(
        timeseries_rows,
        metric="resource_morans_i",
        ylabel="Resource Moran's I",
        filename="05_spatial_organization_trajectories.png",
        ylim=(-0.25, 1.0),
    )

    plot_three_factor_state_space(
        summary_rows
    )

    plot_centralized_vs_decentralized(
        models
    )

    # -----------------------------------------------------------------
    # Optional GIF creation
    # -----------------------------------------------------------------

    if args.make_factor_gifs:
        lookup = {
            spec["scenario"]: spec
            for spec in build_scenarios()
        }

        if args.make_factor_gifs == "all":
            selected_specs = list(
                lookup.values()
            )
        else:
            if args.make_factor_gifs not in lookup:
                valid = ", ".join(
                    sorted(lookup.keys())
                )

                raise ValueError(
                    f"Unknown scenario '{args.make_factor_gifs}'. "
                    f"Valid choices: {valid}, all"
                )

            selected_specs = [
                lookup[
                    args.make_factor_gifs
                ]
            ]

        for spec in selected_specs:
            make_three_factor_gifs(
                spec,
                width=args.width,
                height=args.height,
                steps=args.animation_steps,
                seed=args.seed,
                equilibrium_fraction=args.equilibrium_fraction,
                regeneration_rate=args.recovery_rate,
                coupling_rate=args.coupling,
                initial_fraction=args.initial_fraction,
                animation_mode=args.animation_mode,
                warmup_steps=args.warmup_steps,
                shock_loss=args.shock_loss,
                shock_radius=args.shock_radius,
                frame_every=args.frame_every,
                fps=args.fps,
                save_frames=args.save_frames,
            )

    print("\nDone.")
    print(
        f"Results written to: {RESULTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()