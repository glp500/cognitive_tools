from __future__ import annotations

import mesa
import numpy as np


class EcologyModel(mesa.Model):
    """
    Agent-free renewable spatial environment.

    Each tile has:
        resource  - current resource stock
        capacity  - long-run ecological carrying capacity

    Resources change through:
        1. local regeneration,
        2. natural depletion,
        3. exchange with neighbouring tiles.

    There are deliberately no agents in this model.
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        distribution: str = "uniform",
        mean_capacity: float = 0.6,
        heterogeneity: float = 0.0,
        initial_fraction: float = 0.5,
        regeneration_rate: float = 0.05,
        depletion_rate: float = 0.01,
        coupling_rate: float = 0.10,
        seed: int | None = None,
    ):
        super().__init__(rng=seed)

        if width <= 0 or height <= 0:
            raise ValueError(
                "width and height must be positive."
            )

        if not 0.0 < mean_capacity <= 1.0:
            raise ValueError(
                "mean_capacity must be between 0 and 1."
            )

        if not 0.0 <= heterogeneity <= 1.0:
            raise ValueError(
                "heterogeneity must be between 0 and 1."
            )

        if not 0.0 <= initial_fraction <= 1.0:
            raise ValueError(
                "initial_fraction must be between 0 and 1."
            )

        if regeneration_rate < 0.0:
            raise ValueError(
                "regeneration_rate must be non-negative."
            )

        if depletion_rate < 0.0:
            raise ValueError(
                "depletion_rate must be non-negative."
            )

        if not 0.0 <= coupling_rate <= 1.0:
            raise ValueError(
                "coupling_rate must be between 0 and 1."
            )

        self.width = width
        self.height = height

        self.distribution = distribution
        self.mean_capacity = mean_capacity
        self.heterogeneity = heterogeneity

        self.initial_fraction = initial_fraction

        self.regeneration_rate = regeneration_rate
        self.depletion_rate = depletion_rate
        self.coupling_rate = coupling_rate

        # Persistent ecological opportunity structure.
        self.capacity = self._make_capacity()

        # Initial resource stock.
        self.resource = (
            self.initial_fraction
            * self.capacity
        )

    # ------------------------------------------------------------------
    # Environmental dynamics
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        Advance the ecological environment by one timestep.
        """

        # Logistic local regeneration.
        growth = (
            self.regeneration_rate
            * self.resource
            * (
                1.0
                - self.resource
                / self.capacity
            )
        )

        # Background endogenous resource loss.
        depletion = (
            self.depletion_rate
            * self.resource
        )

        # Resource relationship with neighbouring tiles.
        neighbour_mean = (
            self._neighbour_mean(
                self.resource
            )
        )

        exchange = (
            self.coupling_rate
            * (
                neighbour_mean
                - self.resource
            )
        )

        self.resource += (
            growth
            - depletion
            + exchange
        )

        self.resource = np.clip(
            self.resource,
            0.0,
            self.capacity,
        )

    # ------------------------------------------------------------------
    # Spatial distributions
    # ------------------------------------------------------------------

    def _make_capacity(
        self,
    ) -> np.ndarray:
        """
        Create the persistent carrying-capacity landscape.

        The distributions are centred around the same
        mean carrying capacity. Their spatial organisation
        changes, rather than simply changing total abundance.
        """

        if self.distribution == "uniform":

            pattern = np.zeros(
                (
                    self.height,
                    self.width,
                ),
                dtype=float,
            )

        elif self.distribution == "gradient":

            pattern = (
                self._gradient_pattern()
            )

        elif self.distribution == "patchy":

            pattern = (
                self._patchy_pattern()
            )

        elif self.distribution == "concentrated":

            pattern = (
                self._concentrated_pattern()
            )

        else:

            raise ValueError(
                "distribution must be one of: "
                "'uniform', "
                "'gradient', "
                "'patchy', "
                "'concentrated'"
            )

        # Prevent heterogeneity from producing values
        # above 1 or below 0.
        max_amplitude = min(
            self.mean_capacity,
            1.0 - self.mean_capacity,
        )

        amplitude = (
            self.heterogeneity
            * max_amplitude
        )

        capacity = (
            self.mean_capacity
            + amplitude * pattern
        )

        return np.clip(
            capacity,
            0.001,
            1.0,
        )

    def _gradient_pattern(
        self,
    ) -> np.ndarray:
        """
        Smooth low-to-high gradient from left to right.
        """

        x = np.linspace(
            -1.0,
            1.0,
            self.width,
        )

        return np.tile(
            x,
            (
                self.height,
                1,
            ),
        )

    def _patchy_pattern(
        self,
    ) -> np.ndarray:
        """
        Spatially autocorrelated random resource potential.

        Random variation is repeatedly averaged with
        neighbouring cells, producing contiguous regions.
        """

        field = self.rng.normal(
            size=(
                self.height,
                self.width,
            )
        )

        for _ in range(6):

            neighbours = (
                self._neighbour_mean(
                    field
                )
            )

            field = (
                0.5 * field
                + 0.5 * neighbours
            )

        return self._standardize_pattern(
            field
        )

    def _concentrated_pattern(
        self,
    ) -> np.ndarray:
        """
        One highly productive central region surrounded
        by lower-productivity territory.
        """

        y, x = np.indices(
            (
                self.height,
                self.width,
            )
        )

        centre_x = (
            self.width - 1
        ) / 2.0

        centre_y = (
            self.height - 1
        ) / 2.0

        scale = (
            min(
                self.width,
                self.height,
            )
            / 4.0
        )

        distance_squared = (
            (x - centre_x) ** 2
            + (y - centre_y) ** 2
        )

        field = np.exp(
            -distance_squared
            / (
                2.0
                * scale**2
            )
        )

        return self._standardize_pattern(
            field
        )

    @staticmethod
    def _standardize_pattern(
        field: np.ndarray,
    ) -> np.ndarray:
        """
        Centre a spatial pattern on zero and scale it
        to approximately [-1, 1].
        """

        field = (
            field
            - field.mean()
        )

        maximum = np.max(
            np.abs(field)
        )

        if maximum == 0:

            return np.zeros_like(
                field
            )

        return (
            field
            / maximum
        )

    # ------------------------------------------------------------------
    # Neighbour relationship
    # ------------------------------------------------------------------

    @staticmethod
    def _neighbour_mean(
        field: np.ndarray,
    ) -> np.ndarray:
        """
        Return the mean value of the four adjacent tiles.

        The environment does not wrap from one edge
        to the opposite edge.
        """

        padded = np.pad(
            field,
            pad_width=1,
            mode="edge",
        )

        up = padded[
            :-2,
            1:-1,
        ]

        down = padded[
            2:,
            1:-1,
        ]

        left = padded[
            1:-1,
            :-2,
        ]

        right = padded[
            1:-1,
            2:,
        ]

        return (
            up
            + down
            + left
            + right
        ) / 4.0

    # ------------------------------------------------------------------
    # Environmental metrics
    # ------------------------------------------------------------------

    @property
    def total_resource(
        self,
    ) -> float:

        return float(
            self.resource.sum()
        )

    @property
    def mean_resource(
        self,
    ) -> float:

        return float(
            self.resource.mean()
        )

    @property
    def resource_fraction(
        self,
    ) -> float:
        """
        Mean resource abundance relative to each
        tile's local carrying capacity.
        """

        return float(
            np.mean(
                self.resource
                / self.capacity
            )
        )

    @property
    def resource_gini(
        self,
    ) -> float:
        """
        Inequality in current resource stock
        across tiles.
        """

        return self._gini(
            self.resource
        )

    @property
    def capacity_gini(
        self,
    ) -> float:
        """
        Structural inequality in ecological
        opportunity across tiles.
        """

        return self._gini(
            self.capacity
        )

    def scarcity_fraction(
        self,
        threshold: float = 0.20,
    ) -> float:
        """
        Fraction of tiles with resource levels below
        a proportion of their own carrying capacity.

        Default threshold = 20%.
        """

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                "threshold must be between 0 and 1."
            )

        scarce = (
            self.resource
            <
            threshold
            * self.capacity
        )

        return float(
            np.mean(scarce)
        )

    @staticmethod
    def _gini(
        field: np.ndarray,
    ) -> float:

        values = np.sort(
            field.ravel()
        )

        total = values.sum()

        if total == 0:

            return 0.0

        n = len(values)

        index = np.arange(
            1,
            n + 1,
        )

        return float(
            2
            * np.sum(
                index * values
            )
            / (
                n * total
            )
            - (
                n + 1
            ) / n
        )