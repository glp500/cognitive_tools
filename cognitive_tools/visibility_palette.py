"""Shared, print-friendly colors for visibility study figures."""

ECOLOGY_COLORS = {
    "uniform_high": "#197A73",
    "patchy_high": "#B06435",
    "split_high_low": "#6654A4",
    "balanced_uniform": "#197A73",
    "balanced_dispersed": "#B06435",
    "balanced_segregated": "#6654A4",
}
ECOLOGY_MARKERS = {
    "uniform_high": "o",
    "patchy_high": "s",
    "split_high_low": "^",
    "balanced_uniform": "o",
    "balanced_dispersed": "s",
    "balanced_segregated": "^",
}
DYNAMICS_COLORS = {"fixed": "#245B78", "adaptive_bounded": "#C56A32"}
DYNAMICS_MARKERS = {"fixed": "o", "adaptive_bounded": "^"}
INK = "#24323B"
MUTED = "#5B6972"
GRID = "#E4E9EA"
INITIAL = "#547B91"
FINAL = DYNAMICS_COLORS["adaptive_bounded"]
