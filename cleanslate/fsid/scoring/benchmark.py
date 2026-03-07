"""Post-unlearning scoring: compare pre/post memorization metrics."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cleanslate.fsid.datasets.io import load_local_records


def _resolve_song_id(row: dict[str, Any]) -> str | None:
    raw = row.get("song_id") or row.get("content_id")
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw)


def _resolve_song_title(row: dict[str, Any]) -> str:
    return str(row.get("song_title") or row.get("content_title") or "")


@dataclass
class WindowDelta:
    """Per-window change in memorization after unlearning."""

    idx: int
    pre_p_z: float
    post_p_z: float
    delta_p_z: float  # pre - post (positive = less memorized)
    pre_is_memorized: bool
    post_is_memorized: bool
    transition: (
        str  # "forgotten" | "retained_memorized" | "retained_clean" | "regressed"
    )


@dataclass
class SongDelta:
    """Per-song change in memorization after unlearning."""

    song_id: str
    song_title: str
    pre_memorized_fraction: float
    post_memorized_fraction: float
    delta_memorized_fraction: float
    pre_max_p_z: float
    post_max_p_z: float
    delta_max_p_z: float
    in_forget_set: bool
    num_windows: int | None = None
    windows_forgotten: int | None = None
    windows_retained_memorized: int | None = None
    windows_retained_clean: int | None = None
    windows_regressed: int | None = None
    window_deltas: list[WindowDelta] | None = None


@dataclass
class UnlearningScore:
    """Aggregate unlearning benchmark score."""

    forget_count: int
    forget_mean_delta_frac: float
    forget_mean_delta_pz: float
    forget_fraction_improved: float

    retain_count: int
    retain_mean_abs_delta_frac: float
    retain_mean_abs_delta_pz: float

    forget_score: float
    retain_penalty: float

    per_song: list[SongDelta]

    forget_total_windows: int | None = None
    forget_windows_forgotten: int | None = None
    forget_windows_regressed: int | None = None
    forget_window_forget_rate: float | None = None

    retain_total_windows: int | None = None
    retain_windows_regressed: int | None = None
    retain_window_stability: float | None = None

    # Optional validation benchmark deltas (populated when --pre-validation / --post-validation provided)
    validation_deltas: list[dict[str, Any]] | None = None
    utility_retained: float | None = None  # avg(post_acc/pre_acc) across benchmarks

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for song in d.get("per_song", []):
            song.pop("window_deltas", None)
        return d


def _compute_window_deltas(
    pre_windows: list[dict[str, Any]] | None,
    post_windows: list[dict[str, Any]] | None,
) -> tuple[int, int, int, int, int, list[WindowDelta]] | None:
    """Match pre/post windows by idx and classify transitions.

    Returns (num_windows, forgotten, retained_memorized, retained_clean, regressed, deltas)
    or None if window data is absent.
    """
    if not pre_windows or not post_windows:
        return None

    pre_by_idx: dict[int, dict[str, Any]] = {int(w["idx"]): w for w in pre_windows}
    post_by_idx: dict[int, dict[str, Any]] = {int(w["idx"]): w for w in post_windows}

    common_idxs = sorted(set(pre_by_idx) & set(post_by_idx))
    if not common_idxs:
        return None

    forgotten = 0
    retained_memorized = 0
    retained_clean = 0
    regressed = 0
    deltas: list[WindowDelta] = []

    for idx in common_idxs:
        pre_w = pre_by_idx[idx]
        post_w = post_by_idx[idx]

        pre_pz = float(pre_w.get("p_z", 0.0))
        post_pz = float(post_w.get("p_z", 0.0))
        pre_mem = bool(pre_w.get("is_memorized", False))
        post_mem = bool(post_w.get("is_memorized", False))

        if pre_mem and not post_mem:
            transition = "forgotten"
            forgotten += 1
        elif pre_mem and post_mem:
            transition = "retained_memorized"
            retained_memorized += 1
        elif not pre_mem and not post_mem:
            transition = "retained_clean"
            retained_clean += 1
        else:
            transition = "regressed"
            regressed += 1

        deltas.append(
            WindowDelta(
                idx=idx,
                pre_p_z=pre_pz,
                post_p_z=post_pz,
                delta_p_z=pre_pz - post_pz,
                pre_is_memorized=pre_mem,
                post_is_memorized=post_mem,
                transition=transition,
            )
        )

    return (
        len(common_idxs),
        forgotten,
        retained_memorized,
        retained_clean,
        regressed,
        deltas,
    )


def _index_unique_by_song_id(
    records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    missing = 0
    for row in records:
        song_id = _resolve_song_id(row)
        if song_id is None:
            missing += 1
            continue
        if song_id in indexed:
            duplicates.add(song_id)
            continue
        indexed[song_id] = row

    if missing:
        raise ValueError(
            f"{label} contains {missing} row(s) without song_id/content_id"
        )
    if duplicates:
        sample = sorted(duplicates)[:5]
        raise ValueError(
            f"{label} contains duplicate song_id values: {sample}. "
            "Evaluation universes must be unique by song_id."
        )
    return indexed


def compute_score(
    *,
    forget_set_path: str | Path,
    pre_eval_path: str | Path,
    post_eval_path: str | Path,
) -> UnlearningScore:
    """Compute a benchmark score from local pre/post eval artifacts."""
    forget_rows = load_local_records(forget_set_path)
    forget_by_id = _index_unique_by_song_id(forget_rows, label="forget set")
    forget_ids = set(forget_by_id.keys())

    pre_records = load_local_records(pre_eval_path)
    post_records = load_local_records(post_eval_path)

    pre_by_id = _index_unique_by_song_id(pre_records, label="pre-eval")
    post_by_id = _index_unique_by_song_id(post_records, label="post-eval")

    pre_ids = set(pre_by_id.keys())
    post_ids = set(post_by_id.keys())
    if pre_ids != post_ids:
        missing_from_post = sorted(pre_ids - post_ids)[:5]
        missing_from_pre = sorted(post_ids - pre_ids)[:5]
        raise ValueError(
            "pre-eval and post-eval must cover the exact same song universe. "
            f"Missing from post: {missing_from_post}; missing from pre: {missing_from_pre}"
        )

    missing_forget = forget_ids - pre_ids
    if missing_forget:
        raise ValueError(
            f"{len(missing_forget)} forget-set song(s) missing from eval universe: "
            f"{sorted(missing_forget)[:5]}"
        )

    deltas: list[SongDelta] = []
    for song_id in sorted(pre_ids):
        pre = pre_by_id[song_id]
        post = post_by_id[song_id]

        pre_frac_raw = pre.get("memorized_fraction")
        post_frac_raw = post.get("memorized_fraction")
        pre_pz_raw = pre.get("max_p_z")
        post_pz_raw = post.get("max_p_z")

        if pre_frac_raw is None:
            raise KeyError(
                f"Pre-eval row for song_id={song_id!r} missing 'memorized_fraction'"
            )
        if post_frac_raw is None:
            raise KeyError(
                f"Post-eval row for song_id={song_id!r} missing 'memorized_fraction'"
            )
        if pre_pz_raw is None:
            raise KeyError(f"Pre-eval row for song_id={song_id!r} missing 'max_p_z'")
        if post_pz_raw is None:
            raise KeyError(f"Post-eval row for song_id={song_id!r} missing 'max_p_z'")

        pre_frac = float(pre_frac_raw)
        post_frac = float(post_frac_raw)
        pre_pz = float(pre_pz_raw)
        post_pz = float(post_pz_raw)

        window_result = _compute_window_deltas(pre.get("windows"), post.get("windows"))
        window_kwargs: dict[str, Any] = {}
        if window_result is not None:
            n_win, w_forg, w_ret_mem, w_ret_clean, w_reg, w_deltas = window_result
            window_kwargs = dict(
                num_windows=n_win,
                windows_forgotten=w_forg,
                windows_retained_memorized=w_ret_mem,
                windows_retained_clean=w_ret_clean,
                windows_regressed=w_reg,
                window_deltas=w_deltas,
            )

        deltas.append(
            SongDelta(
                song_id=song_id,
                song_title=_resolve_song_title(pre),
                pre_memorized_fraction=pre_frac,
                post_memorized_fraction=post_frac,
                delta_memorized_fraction=pre_frac - post_frac,
                pre_max_p_z=pre_pz,
                post_max_p_z=post_pz,
                delta_max_p_z=pre_pz - post_pz,
                in_forget_set=(song_id in forget_ids),
                **window_kwargs,
            )
        )

    forget_deltas = [delta for delta in deltas if delta.in_forget_set]
    retain_deltas = [delta for delta in deltas if not delta.in_forget_set]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    forget_count = len(forget_deltas)
    forget_mean_delta_frac = _mean(
        [delta.delta_memorized_fraction for delta in forget_deltas]
    )
    forget_mean_delta_pz = _mean([delta.delta_max_p_z for delta in forget_deltas])
    forget_fraction_improved = (
        sum(1 for delta in forget_deltas if delta.delta_memorized_fraction > 0)
        / forget_count
        if forget_count > 0
        else 0.0
    )

    retain_count = len(retain_deltas)
    retain_mean_abs_delta_frac = _mean(
        [abs(delta.delta_memorized_fraction) for delta in retain_deltas]
    )
    retain_mean_abs_delta_pz = _mean(
        [abs(delta.delta_max_p_z) for delta in retain_deltas]
    )

    forget_score = forget_fraction_improved
    pre_retain_mean = _mean([delta.pre_memorized_fraction for delta in retain_deltas])
    retain_penalty = (
        retain_mean_abs_delta_frac / pre_retain_mean
        if pre_retain_mean > 0
        else retain_mean_abs_delta_frac
    )
    retain_penalty = min(retain_penalty, 1.0)

    # Aggregate window-level metrics across forget/retain sets.
    window_agg_kwargs: dict[str, Any] = {}
    has_forget_windows = any(d.num_windows is not None for d in forget_deltas)
    has_retain_windows = any(d.num_windows is not None for d in retain_deltas)

    if has_forget_windows:
        f_total = sum(d.num_windows for d in forget_deltas if d.num_windows is not None)
        f_forgotten = sum(
            d.windows_forgotten
            for d in forget_deltas
            if d.windows_forgotten is not None
        )
        f_regressed = sum(
            d.windows_regressed
            for d in forget_deltas
            if d.windows_regressed is not None
        )
        f_retained_mem = sum(
            d.windows_retained_memorized
            for d in forget_deltas
            if d.windows_retained_memorized is not None
        )
        f_pre_memorized = f_forgotten + f_retained_mem
        window_agg_kwargs["forget_total_windows"] = f_total
        window_agg_kwargs["forget_windows_forgotten"] = f_forgotten
        window_agg_kwargs["forget_windows_regressed"] = f_regressed
        window_agg_kwargs["forget_window_forget_rate"] = (
            f_forgotten / f_pre_memorized if f_pre_memorized > 0 else 0.0
        )

    if has_retain_windows:
        r_total = sum(d.num_windows for d in retain_deltas if d.num_windows is not None)
        r_regressed = sum(
            d.windows_regressed
            for d in retain_deltas
            if d.windows_regressed is not None
        )
        window_agg_kwargs["retain_total_windows"] = r_total
        window_agg_kwargs["retain_windows_regressed"] = r_regressed
        window_agg_kwargs["retain_window_stability"] = (
            1.0 - (r_regressed / r_total) if r_total > 0 else 1.0
        )

    return UnlearningScore(
        forget_count=forget_count,
        forget_mean_delta_frac=forget_mean_delta_frac,
        forget_mean_delta_pz=forget_mean_delta_pz,
        forget_fraction_improved=forget_fraction_improved,
        retain_count=retain_count,
        retain_mean_abs_delta_frac=retain_mean_abs_delta_frac,
        retain_mean_abs_delta_pz=retain_mean_abs_delta_pz,
        forget_score=forget_score,
        retain_penalty=retain_penalty,
        per_song=deltas,
        **window_agg_kwargs,
    )


def save_score(score: UnlearningScore, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(score.to_dict(), f, indent=2)
    return output_path
