from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import superconductivity as sc
from scipy.optimize import (
    LinearConstraint,
    OptimizeResult,
    differential_evolution,
    minimize_scalar,
)
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

# source /Users/oliver/Documents/cryolab/.venv/bin/activate

tau_grid = np.linspace(0.0, 1.0, 101)
Vgrid_mV = np.linspace(-0.9, 0.9, 1801)
_, Imar_nA = sc.get_Imar_nA(
    V_mV=Vgrid_mV,
    tau=tau_grid,
    tau_resolved=True,
    Delta_meV=0.1895,
    gamma_meV=1e-6,
    T_K=0.0,
    show_progress=True,
)
Imar_nA = Imar_nA.T

N_PHI_CPR = 2048
phi_cpr_grid = (np.arange(N_PHI_CPR, dtype=float) + 0.5) * (2 * np.pi / N_PHI_CPR)
Isc_abs_table_nA = np.stack(
    [
        sc.get_cpr_abs_nA(phi_cpr_grid, tau=float(transmission))
        for transmission in tau_grid
    ]
)


def get_cached_cpr_abs_nA(phi, tau):
    """Return an ABS CPR from the precomputed transmission-phase table."""
    phase = np.asarray(phi, dtype=float)
    tau_index = int(np.rint(float(tau) * (tau_grid.size - 1)))
    tau_index = int(np.clip(tau_index, 0, tau_grid.size - 1))
    cpr = Isc_abs_table_nA[tau_index]
    if phase.shape == phi_cpr_grid.shape and np.array_equal(
        phase,
        phi_cpr_grid,
    ):
        return cpr
    return np.interp(
        phase,
        phi_cpr_grid,
        cpr,
        period=2 * np.pi,
    )


@lru_cache(maxsize=256)
def _get_total_mar_inverse(tau_indices, multiplicities):
    """Return the current and voltage axes for a parallel PIN code."""
    indices = np.asarray(tau_indices, dtype=np.intp)
    counts = np.asarray(multiplicities, dtype=np.int64)

    # Imar_nA is the single cached MAR bank with shape (tau, voltage).
    # Parallel channels share voltage, so their currents add directly.
    total_current = np.sum(counts[:, None] * Imar_nA[indices], axis=0)

    # The total MAR curve is assumed to be monotonic, so no per-candidate
    # sorting or duplicate removal is needed before interpolation.
    return total_current, Vgrid_mV


def _group_parameters(tau, multiplicities):
    """Validate grouped transmissions and return their grid indices."""
    transmissions = np.asarray(tau, dtype=float)
    if transmissions.ndim == 0:
        transmissions = transmissions.reshape(1)
    if transmissions.ndim != 1 or transmissions.size == 0:
        raise ValueError("tau must be a scalar or a nonempty 1D sequence.")
    if np.any(~np.isfinite(transmissions)) or np.any(
        (transmissions < 0.0) | (transmissions > 1.0)
    ):
        raise ValueError("all transmissions must lie in [0, 1].")

    if multiplicities is None:
        counts = np.ones(transmissions.size, dtype=np.int64)
    else:
        values = np.asarray(multiplicities, dtype=float)
        if values.shape != transmissions.shape:
            raise ValueError("multiplicities must have the same shape as tau.")
        counts = values.astype(np.int64)
        if np.any(values != counts) or np.any(counts < 1):
            raise ValueError("multiplicities must contain positive integers.")

    indices = np.rint(transmissions * (tau_grid.size - 1)).astype(np.intp)
    indices = np.clip(indices, 0, tau_grid.size - 1)
    return transmissions, counts, indices


def get_Vmar_mV(I_nA, tau, multiplicities=None):
    """Return MAR voltage for one channel or a parallel PIN code."""
    I_nA = np.asarray(I_nA, dtype=float)
    _, counts, indices = _group_parameters(tau, multiplicities)
    current, voltage = _get_total_mar_inverse(
        tuple(int(value) for value in indices),
        tuple(int(value) for value in counts),
    )
    return np.interp(
        I_nA,
        current,
        voltage,
        left=np.nan,
        right=np.nan,
    )


def get_rsj_Vj_mV(
    Ibias_nA,
    tau,
    alpha,
    get_Vmar_mV,
    get_Isc_nA,
    n_phi=2048,
    multiplicities=None,
):
    """Return nonlinear-RSJ junction voltage for a scalar or PIN code."""
    Ibias_nA = np.asarray(Ibias_nA, dtype=float)
    transmissions, counts, _ = _group_parameters(tau, multiplicities)

    # Midpoints reduce the chance of evaluating exactly at V = 0.
    phi = (np.arange(n_phi, dtype=float) + 0.5) * (2 * np.pi / n_phi)

    Is_nA = alpha * np.sum(
        [
            count * get_Isc_nA(phi, float(transmission))
            for transmission, count in zip(
                transmissions,
                counts,
                strict=True,
            )
        ],
        axis=0,
    )

    # Assuming Imar(0) = 0.
    lower_static_nA = np.min(Is_nA)
    upper_static_nA = np.max(Is_nA)

    Vj_mV = np.zeros_like(Ibias_nA)

    running = (Ibias_nA < lower_static_nA) | (Ibias_nA > upper_static_nA)

    running_indices = np.flatnonzero(running)
    if running_indices.size:
        Iqp_nA = Ibias_nA[running_indices, None] - Is_nA[None, :]
        Vinst_mV = get_Vmar_mV(
            Iqp_nA,
            transmissions,
            multiplicities=counts,
        )
        usable = np.all(np.isfinite(Vinst_mV), axis=1) & np.all(
            np.abs(Vinst_mV) > np.finfo(float).eps,
            axis=1,
        )
        running_voltage = np.full(running_indices.size, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            running_voltage[usable] = 1.0 / np.mean(
                1.0 / Vinst_mV[usable],
                axis=1,
            )
        Vj_mV[running_indices] = running_voltage

    return Vj_mV


def get_group_multiplicities(n_channels, n_groups=6):
    """Distribute channels as evenly as possible over groups."""
    if n_groups < 1 or n_channels < n_groups:
        raise ValueError("n_channels must be at least n_groups >= 1.")
    counts = np.full(n_groups, n_channels // n_groups, dtype=np.int64)
    counts[: n_channels % n_groups] += 1
    return counts


def fit_grouped_rsj(
    Ibias_nA,
    Vexp_mV,
    *,
    n_channels,
    conductance_bounds,
    alpha_bounds=(0.0, 1.5),
    n_groups=6,
    multiplicities=None,
    weights=None,
    n_phi=512,
    seed=0,
    maxiter=200,
    popsize=12,
    restarts=4,
    progress=True,
):
    """Fit dimensionless CPR scale and grouped transmission PIN code."""
    current = np.asarray(Ibias_nA, dtype=float)
    voltage = np.asarray(Vexp_mV, dtype=float)
    if current.shape != voltage.shape:
        raise ValueError("Ibias_nA and Vexp_mV must have the same shape.")
    fit_weights = (
        np.ones_like(voltage)
        if weights is None
        else np.broadcast_to(np.asarray(weights, dtype=float), voltage.shape)
    )
    valid = (
        np.isfinite(current)
        & np.isfinite(voltage)
        & np.isfinite(fit_weights)
        & (fit_weights > 0.0)
    )
    if np.count_nonzero(valid) < n_groups + 1:
        raise ValueError("not enough finite, positively weighted samples.")

    lower_G0, upper_G0 = map(float, conductance_bounds)
    if not 0.0 <= lower_G0 <= upper_G0 <= n_channels:
        raise ValueError("conductance_bounds conflict with n_channels.")
    if multiplicities is None:
        multiplicities = get_group_multiplicities(n_channels, n_groups)
    else:
        _, multiplicities, _ = _group_parameters(
            np.zeros(n_groups),
            multiplicities,
        )
        if int(np.sum(multiplicities)) != n_channels:
            raise ValueError("n_channels must equal the sum of multiplicities.")
    fit_current = current[valid]
    fit_voltage = voltage[valid]
    fit_weight = fit_weights[valid]

    def objective(parameters):
        alpha = float(parameters[0])
        tau_indices = np.rint(parameters[1:]).astype(np.intp)
        transmissions = tau_grid[tau_indices]
        model = get_rsj_Vj_mV(
            fit_current,
            tau=transmissions,
            alpha=alpha,
            get_Vmar_mV=get_Vmar_mV,
            get_Isc_nA=get_cached_cpr_abs_nA,
            n_phi=n_phi,
            multiplicities=multiplicities,
        )
        if np.any(~np.isfinite(model)):
            return np.inf
        residual = model - fit_voltage
        return float(np.sum(fit_weight * residual**2))

    if int(restarts) != restarts or restarts < 1:
        raise ValueError("restarts must be a positive integer.")

    # The optimized transmission parameters are integer indices on tau_grid.
    # Scaling the multiplicities converts their constrained sum back to G0.
    coefficients = np.concatenate(
        (
            [0.0],
            multiplicities.astype(float) / (tau_grid.size - 1),
        )
    )
    conductance_constraint = LinearConstraint(
        coefficients,
        lower_G0,
        upper_G0,
    )
    seed_sequence = np.random.SeedSequence(seed)
    restart_results = []
    for restart, child in enumerate(seed_sequence.spawn(int(restarts)), 1):
        with tqdm(
            total=maxiter,
            desc=f"RSJ restart {restart}/{restarts}",
            unit="generation",
            disable=not progress,
        ) as fit_progress:

            def update_progress(intermediate_result):
                fit_progress.update(1)
                if np.isfinite(intermediate_result.fun):
                    fit_progress.set_postfix(chi2=f"{intermediate_result.fun:.5g}")
                return False

            restart_results.append(
                differential_evolution(
                    objective,
                    bounds=[tuple(map(float, alpha_bounds))]
                    + [(0, tau_grid.size - 1)] * n_groups,
                    constraints=(conductance_constraint,),
                    integrality=[False] + [True] * n_groups,
                    seed=int(child.generate_state(1)[0]),
                    maxiter=maxiter,
                    popsize=popsize,
                    polish=False,
                    updating="immediate",
                    workers=1,
                    callback=update_progress,
                    disp=False,
                )
            )
    result = min(restart_results, key=lambda candidate: candidate.fun)
    result.alpha = float(result.x[0])
    result.tau_indices = np.rint(result.x[1:]).astype(np.intp)
    result.tau = tau_grid[result.tau_indices]
    result.multiplicities = multiplicities
    result.GN_G0 = float(np.dot(multiplicities, result.tau))
    result.restart_results = restart_results
    result.Vfit_mV = get_rsj_Vj_mV(
        current,
        tau=result.tau,
        alpha=result.alpha,
        get_Vmar_mV=get_Vmar_mV,
        get_Isc_nA=get_cached_cpr_abs_nA,
        n_phi=n_phi,
        multiplicities=multiplicities,
    )
    return result


@dataclass(frozen=True)
class GroupedCandidate:
    """Canonical grouped PIN code and dimensionless CPR suppression."""

    alpha: float
    groups: tuple[tuple[int, int], ...]


def _canonical_groups(groups, max_groups):
    """Merge equal transmissions, remove empty groups, and sort."""
    merged = {}
    for tau_index, count in groups:
        index = int(np.clip(np.rint(tau_index), 1, tau_grid.size - 1))
        multiplicity = int(np.rint(count))
        if multiplicity > 0:
            merged[index] = merged.get(index, 0) + multiplicity
    canonical = sorted(merged.items(), reverse=True)
    while len(canonical) > max_groups:
        separation = [
            abs(canonical[index][0] - canonical[index + 1][0])
            for index in range(len(canonical) - 1)
        ]
        index = int(np.argmin(separation))
        first_tau, first_count = canonical[index]
        second_tau, second_count = canonical[index + 1]
        count = first_count + second_count
        transmission = int(
            np.rint((first_tau * first_count + second_tau * second_count) / count)
        )
        canonical[index : index + 2] = [(transmission, count)]
        canonical = sorted(canonical, reverse=True)
    return tuple(canonical)


def fit_grouped_rsj_memetic(
    Ibias_nA,
    Vexp_mV,
    *,
    conductance_bounds,
    alpha_bounds=(0.0, 0.2),
    max_groups=6,
    channel_bounds=None,
    weights=None,
    n_phi=512,
    islands=4,
    population_size=20,
    generations=75,
    migration_interval=10,
    seed=0,
    progress=True,
):
    """Fit grouped transmissions and multiplicities with an island search."""
    current = np.asarray(Ibias_nA, dtype=float)
    voltage = np.asarray(Vexp_mV, dtype=float)
    if current.shape != voltage.shape or current.ndim != 1:
        raise ValueError("Ibias_nA and Vexp_mV must be same-length 1D arrays.")
    fit_weights = (
        np.ones_like(voltage)
        if weights is None
        else np.broadcast_to(np.asarray(weights, dtype=float), voltage.shape)
    )
    valid = (
        np.isfinite(current)
        & np.isfinite(voltage)
        & np.isfinite(fit_weights)
        & (fit_weights > 0.0)
    )
    if np.count_nonzero(valid) < 2 * max_groups + 1:
        raise ValueError("not enough finite, positively weighted samples.")
    fit_current = current[valid]
    fit_voltage = voltage[valid]
    fit_weight = fit_weights[valid]
    n_data = fit_current.size

    lower_G0, upper_G0 = map(float, conductance_bounds)
    if not 0.0 < lower_G0 <= upper_G0:
        raise ValueError("invalid conductance_bounds.")
    if channel_bounds is None:
        channel_bounds = (
            int(np.ceil(upper_G0)),
            int(np.ceil(2 * upper_G0)),
        )
    min_channels, max_channels = map(int, channel_bounds)
    if min_channels < 1 or max_channels < min_channels:
        raise ValueError("invalid channel_bounds.")
    if max_groups < 1 or islands < 1 or population_size < 4:
        raise ValueError("invalid population configuration.")

    lower_numerator = int(np.ceil(lower_G0 * (tau_grid.size - 1)))
    upper_numerator = int(np.floor(upper_G0 * (tau_grid.size - 1)))
    alpha_low, alpha_high = map(float, alpha_bounds)
    rng = np.random.default_rng(seed)
    score_cache = {}

    def repair(groups):
        groups = _canonical_groups(groups, max_groups)
        if not groups:
            return None
        n_channels = sum(count for _, count in groups)
        if not min_channels <= n_channels <= max_channels:
            return None
        for _ in range(4 * max_groups + 4):
            numerator = sum(index * count for index, count in groups)
            if lower_numerator <= numerator <= upper_numerator:
                return groups
            target = lower_numerator if numerator < lower_numerator else upper_numerator
            candidates = []
            for group_index, (index, count) in enumerate(groups):
                ideal = index + (target - numerator) / count
                for proposed in (np.floor(ideal), np.ceil(ideal)):
                    proposed = int(np.clip(proposed, 1, tau_grid.size - 1))
                    changed = list(groups)
                    changed[group_index] = (proposed, count)
                    changed = _canonical_groups(changed, max_groups)
                    changed_numerator = sum(i * n for i, n in changed)
                    distance = max(
                        lower_numerator - changed_numerator,
                        changed_numerator - upper_numerator,
                        0,
                    )
                    candidates.append((distance, changed))
            if not candidates:
                return None
            groups = min(candidates, key=lambda item: item[0])[1]
        return None

    def evaluate(candidate, *, use_bic=True):
        key = (candidate.groups, round(candidate.alpha, 10), n_phi)
        cached = score_cache.get(key)
        if cached is None:
            tau_indices, multiplicities = zip(*candidate.groups, strict=True)
            model = get_rsj_Vj_mV(
                fit_current,
                tau=tau_grid[np.asarray(tau_indices, dtype=np.intp)],
                alpha=candidate.alpha,
                get_Vmar_mV=get_Vmar_mV,
                get_Isc_nA=get_cached_cpr_abs_nA,
                n_phi=n_phi,
                multiplicities=np.asarray(multiplicities),
            )
            if np.any(~np.isfinite(model)):
                chi2 = np.inf
            else:
                residual = model - fit_voltage
                chi2 = float(np.sum(fit_weight * residual**2))
            parameters = 2 * len(candidate.groups) + 1
            bic = (
                n_data * np.log(max(chi2 / n_data, np.finfo(float).tiny))
                + parameters * np.log(n_data)
                if np.isfinite(chi2)
                else np.inf
            )
            cached = (bic, chi2)
            score_cache[key] = cached
        return cached[0] if use_bic else cached[1]

    def initialize():
        for _ in range(1000):
            n_channels = int(rng.integers(min_channels, max_channels + 1))
            center = 0.5 * (lower_G0 + upper_G0) / n_channels
            transmission = int(
                np.rint(1000 * np.clip(rng.normal(center, 0.08), 0.001, 1.0))
            )
            groups = repair(((transmission, n_channels),))
            if groups is not None:
                return GroupedCandidate(
                    alpha=float(rng.uniform(alpha_low, alpha_high)),
                    groups=groups,
                )
        raise RuntimeError("could not initialize a feasible PIN code.")

    def mutate(candidate, tau_step, allowed_groups, force_split=False):
        groups = [list(group) for group in candidate.groups]
        alpha = candidate.alpha
        move = (
            "split"
            if force_split
            else rng.choice(
                ("tau", "count", "transfer", "split", "merge", "alpha"),
                p=(0.28, 0.16, 0.14, 0.20, 0.06, 0.16),
            )
        )
        if move == "tau":
            index = int(rng.integers(len(groups)))
            groups[index][0] += int(rng.choice((-5, -2, -1, 1, 2, 5))) * tau_step
        elif move == "count":
            index = int(rng.integers(len(groups)))
            groups[index][1] += int(rng.choice((-1, 1)))
        elif move == "transfer" and len(groups) > 1:
            source, target = rng.choice(len(groups), size=2, replace=False)
            if groups[source][1] > 1:
                groups[source][1] -= 1
                groups[target][1] += 1
        elif move == "split" and len(groups) < allowed_groups:
            options = [index for index, group in enumerate(groups) if group[1] > 1]
            if options:
                index = int(rng.choice(options))
                moved = int(rng.integers(1, groups[index][1]))
                groups[index][1] -= moved
                offset = (
                    int(rng.choice((-1, 1)))
                    * int(rng.choice((2, 5, 10)))
                    * max(tau_step, 1)
                )
                groups.append([groups[index][0] + offset, moved])
        elif move == "merge" and len(groups) > 1:
            first = int(rng.integers(len(groups)))
            second = int(rng.integers(len(groups) - 1))
            if second >= first:
                second += 1
            total = groups[first][1] + groups[second][1]
            merged_tau = int(
                np.rint(
                    (
                        groups[first][0] * groups[first][1]
                        + groups[second][0] * groups[second][1]
                    )
                    / total
                )
            )
            groups[first] = [merged_tau, total]
            groups.pop(second)
        else:
            scale = 0.08 * (alpha_high - alpha_low)
            alpha = float(
                np.clip(alpha + rng.normal(0.0, scale), alpha_low, alpha_high)
            )

        repaired = repair(groups)
        if repaired is None:
            return candidate
        if rng.random() < 0.35:
            scale = 0.03 * (alpha_high - alpha_low)
            alpha = float(
                np.clip(alpha + rng.normal(0.0, scale), alpha_low, alpha_high)
            )
        return GroupedCandidate(alpha=alpha, groups=repaired)

    def tournament(population, scores):
        choices = rng.choice(len(population), size=3, replace=False)
        return population[min(choices, key=lambda index: scores[index])]

    def evolution_score(candidate):
        """Use raw chi squared during structural evolution."""
        return evaluate(candidate, use_bic=False)

    populations = [
        [initialize() for _ in range(population_size)] for _ in range(islands)
    ]
    history = []
    archive_by_group_count = {}
    previous_allowed_groups = 1
    with tqdm(
        total=generations,
        desc="Memetic RSJ fit",
        unit="generation",
        disable=not progress,
    ) as fit_progress:
        for generation in range(generations):
            fraction = generation / max(generations - 1, 1)
            tau_step = 20 if fraction < 0.4 else 5 if fraction < 0.75 else 1
            allowed_groups = min(max_groups, 1 + int(fraction * max_groups))

            # Explicitly open each structural stage by splitting the best
            # candidate from the preceding group count on every island.
            if allowed_groups > previous_allowed_groups:
                for population in populations:
                    sources = [
                        candidate
                        for candidate in population
                        if len(candidate.groups) == previous_allowed_groups
                    ]
                    if not sources:
                        sources = population
                    source = min(sources, key=evolution_score)
                    split = mutate(
                        source,
                        tau_step,
                        allowed_groups,
                        force_split=True,
                    )
                    scores = [evolution_score(candidate) for candidate in population]
                    population[int(np.argmax(scores))] = split
                previous_allowed_groups = allowed_groups

            island_scores = []
            for island, population in enumerate(populations):
                scores = [evolution_score(candidate) for candidate in population]

                # Keep the best candidate at every represented group count.
                # Split candidates thus survive long enough to differentiate.
                best_by_group_count = {}
                for candidate, score in zip(population, scores, strict=True):
                    group_count = len(candidate.groups)
                    previous = best_by_group_count.get(group_count)
                    if previous is None or score < previous[0]:
                        best_by_group_count[group_count] = (score, candidate)
                next_population = [
                    item[1]
                    for item in sorted(
                        best_by_group_count.values(),
                        key=lambda item: item[0],
                    )[:population_size]
                ]
                while len(next_population) < population_size:
                    parent = tournament(population, scores)
                    next_population.append(mutate(parent, tau_step, allowed_groups))
                populations[island] = next_population
                island_scores.append(
                    [evolution_score(item) for item in next_population]
                )

            if migration_interval and (generation + 1) % migration_interval == 0:
                migrants = [
                    populations[index][int(np.argmin(island_scores[index]))]
                    for index in range(islands)
                ]
                for island in range(islands):
                    worst = int(np.argmax(island_scores[(island + 1) % islands]))
                    populations[(island + 1) % islands][worst] = migrants[island]

            best = min(
                (candidate for population in populations for candidate in population),
                key=evolution_score,
            )
            for candidate in (
                candidate for population in populations for candidate in population
            ):
                group_count = len(candidate.groups)
                previous = archive_by_group_count.get(group_count)
                if previous is None or evolution_score(candidate) < evolution_score(
                    previous
                ):
                    archive_by_group_count[group_count] = candidate
            best_bic = evaluate(best)
            history.append(best_bic)
            group_counts = np.bincount(
                [
                    len(candidate.groups)
                    for population in populations
                    for candidate in population
                ],
                minlength=max_groups + 1,
            )
            group_mix = ",".join(
                f"{count}:{group_counts[count]}"
                for count in range(1, max_groups + 1)
                if group_counts[count]
            )
            fit_progress.update(1)
            fit_progress.set_postfix(
                BIC=f"{best_bic:.5g}",
                groups=len(best.groups),
                allowed=allowed_groups,
                mix=group_mix,
            )

    # Refine alpha independently for every explored structural model before
    # allowing BIC to choose the final number of active groups.
    refined_archive = {}
    for group_count, candidate in archive_by_group_count.items():

        def archived_alpha_chi2(alpha):
            return evaluate(
                GroupedCandidate(float(alpha), candidate.groups),
                use_bic=False,
            )

        alpha_result = minimize_scalar(
            archived_alpha_chi2,
            bounds=(alpha_low, alpha_high),
            method="bounded",
            options={"xatol": 1e-5},
        )
        refined_archive[group_count] = GroupedCandidate(
            float(alpha_result.x),
            candidate.groups,
        )
    best = min(refined_archive.values(), key=evaluate)

    # One exact-grid local descent around the best global candidate.
    improved = True
    while improved:
        improved = False
        neighbours = []
        for group_index, (tau_index, count) in enumerate(best.groups):
            for direction in (-1, 1):
                changed = list(best.groups)
                changed[group_index] = (tau_index + direction, count)
                repaired = repair(changed)
                if repaired is not None:
                    neighbours.append(GroupedCandidate(best.alpha, repaired))
                changed = list(best.groups)
                changed[group_index] = (tau_index, count + direction)
                repaired = repair(changed)
                if repaired is not None:
                    neighbours.append(GroupedCandidate(best.alpha, repaired))
        for source in range(len(best.groups)):
            if best.groups[source][1] <= 1:
                continue
            for target in range(len(best.groups)):
                if source == target:
                    continue
                changed = list(best.groups)
                changed[source] = (changed[source][0], changed[source][1] - 1)
                changed[target] = (changed[target][0], changed[target][1] + 1)
                repaired = repair(changed)
                if repaired is not None:
                    neighbours.append(GroupedCandidate(best.alpha, repaired))
        if neighbours:
            candidate = min(neighbours, key=evaluate)
            if evaluate(candidate) < evaluate(best):
                best = candidate
                improved = True

    def final_alpha_chi2(alpha):
        return evaluate(GroupedCandidate(float(alpha), best.groups), use_bic=False)

    alpha_result = minimize_scalar(
        final_alpha_chi2,
        bounds=(alpha_low, alpha_high),
        method="bounded",
        options={"xatol": 1e-5},
    )
    best = GroupedCandidate(float(alpha_result.x), best.groups)

    tau_indices, multiplicities = zip(*best.groups, strict=True)
    tau_indices = np.asarray(tau_indices, dtype=np.intp)
    multiplicities = np.asarray(multiplicities, dtype=np.int64)
    transmissions = tau_grid[tau_indices]
    model = get_rsj_Vj_mV(
        current,
        tau=transmissions,
        alpha=best.alpha,
        get_Vmar_mV=get_Vmar_mV,
        get_Isc_nA=get_cached_cpr_abs_nA,
        n_phi=n_phi,
        multiplicities=multiplicities,
    )
    return OptimizeResult(
        success=True,
        message="Memetic island search completed.",
        fun=evaluate(best, use_bic=False),
        bic=evaluate(best),
        alpha=best.alpha,
        tau=transmissions,
        tau_indices=tau_indices,
        multiplicities=multiplicities,
        n_channels=int(np.sum(multiplicities)),
        GN_G0=float(np.dot(multiplicities, transmissions)),
        Vfit_mV=model,
        history=np.asarray(history),
        candidates_cached=len(score_cache),
        group_chi2={
            count: evaluate(candidate, use_bic=False)
            for count, candidate in refined_archive.items()
        },
        group_bic={
            count: evaluate(candidate) for count, candidate in refined_archive.items()
        },
    )


path = "43.0G0/15.0GHz_stripline"
data = np.load(f"atomic_contact/{path}/Rs_data.npz")
Rs_Ohm = data["Rs_Ohm"]
Voff_mV = data["Voff_mV"]
data = np.load(f"atomic_contact/{path}/eva.npz")
Ibias_nA = data["Ibias_nA"]
Vexp_mV = data["Vexp_mV"][0, :] - Voff_mV - Ibias_nA * (Rs_Ohm * 1e-6)

mask = np.isfinite(Vexp_mV)

Ibins_nA = np.linspace(-1700, 1700, 501)
Vtofit_mV = sc.bin_y_over_x(Vexp_mV[mask], Ibias_nA[mask], Ibins_nA)

result = fit_grouped_rsj_memetic(
    Ibins_nA,
    Vtofit_mV,
    conductance_bounds=(28.0, 30.0),
    alpha_bounds=(0.0, 1.0),
    max_groups=30,
    channel_bounds=(30, 200),
    n_phi=512,
    islands=5,
    population_size=10,
    generations=10000,
    migration_interval=10,
    seed=0,
    progress=True,
)

# Recalculate the optimized model on every experimental point at higher phase
# resolution. The optimizer itself uses the downsampled trace above for speed.
Vfit_mV = get_rsj_Vj_mV(
    Ibias_nA,
    tau=result.tau,
    alpha=result.alpha,
    get_Vmar_mV=get_Vmar_mV,
    get_Isc_nA=get_cached_cpr_abs_nA,
    n_phi=2048,
    multiplicities=result.multiplicities,
)

print(f"alpha = {result.alpha:.6g}")
print(f"tau = {result.tau}")
print(f"tau indices = {result.tau_indices}")
print(f"multiplicities = {result.multiplicities}")
print(f"channels = {result.n_channels}")
print(f"GN = {result.GN_G0:.6g} G0")
print(f"chi2 = {result.fun:.6g}")
print(f"BIC = {result.bic:.6g}")
print(f"chi2 by group count = {result.group_chi2}")
print(f"BIC by group count = {result.group_bic}")
print(f"cached candidates = {result.candidates_cached}")

Isc_abs_nA = np.sum(
    [
        count * get_cached_cpr_abs_nA(phi_cpr_grid, transmission)
        for transmission, count in zip(
            result.tau,
            result.multiplicities,
            strict=True,
        )
    ],
    axis=0,
)
Ic_abs_nA = float(np.max(Isc_abs_nA))
switching_current_nA = result.alpha * Ic_abs_nA
print(f"Ic_abs = {Ic_abs_nA:.6g} nA")
print(f"alpha * Ic_abs = {switching_current_nA:.6g} nA")

plt.plot(Ibias_nA, Vexp_mV, ".", color="grey", label="experiment")
plt.plot(Ibias_nA, Vfit_mV, color="tab:red", label="RSJ + MAR fit")
plt.xlabel("Bias current (nA)")
plt.ylabel("Junction voltage (mV)")
plt.legend()
plt.tight_layout()
plt.show()


def get_didv_branches_G0(
    I_nA,
    V_mV,
    *,
    voltage_cutoff_mV=0.01,
    n_points=300,
    smooth_window=None,
):
    """Return branch-wise differential conductance on uniform voltage grids."""
    current = np.asarray(I_nA, dtype=float)
    voltage = np.asarray(V_mV, dtype=float)
    if current.shape != voltage.shape or current.ndim != 1:
        raise ValueError("I_nA and V_mV must be same-length 1D arrays.")

    branches = []
    finite = np.isfinite(current) & np.isfinite(voltage)
    for polarity in (-1, 1):
        selected = finite & (polarity * voltage > voltage_cutoff_mV)
        if np.count_nonzero(selected) < 5:
            continue

        branch_voltage = voltage[selected]
        branch_current = current[selected]
        order = np.argsort(branch_voltage)
        branch_voltage = branch_voltage[order]
        branch_current = branch_current[order]
        branch_voltage, unique = np.unique(
            branch_voltage,
            return_index=True,
        )
        branch_current = branch_current[unique]

        uniform_voltage = np.linspace(
            branch_voltage[0],
            branch_voltage[-1],
            n_points,
        )
        uniform_current = np.interp(
            uniform_voltage,
            branch_voltage,
            branch_current,
        )
        voltage_step = uniform_voltage[1] - uniform_voltage[0]

        if smooth_window is not None:
            window = min(int(smooth_window), uniform_current.size)
            window -= 1 - window % 2
            if window >= 5:
                didv_uS = savgol_filter(
                    uniform_current,
                    window_length=window,
                    polyorder=min(3, window - 1),
                    deriv=1,
                    delta=voltage_step,
                )
                # Polynomial derivatives are unreliable where the filter
                # lacks a complete centered window.
                margin = window // 2
                didv_uS[:margin] = np.nan
                didv_uS[-margin:] = np.nan
            else:
                didv_uS = np.gradient(
                    uniform_current,
                    uniform_voltage,
                    edge_order=2,
                )
        else:
            didv_uS = np.gradient(
                uniform_current,
                uniform_voltage,
                edge_order=2,
            )
        branches.append((uniform_voltage, didv_uS / sc.G0_muS))
    return branches


didv_exp = get_didv_branches_G0(
    Ibias_nA,
    Vexp_mV,
    smooth_window=15,
)
didv_fit = get_didv_branches_G0(
    Ibias_nA,
    Vfit_mV,
    smooth_window=9,
)

for index, (voltage, conductance) in enumerate(didv_exp):
    plt.plot(
        voltage,
        conductance,
        color="grey",
        label="experiment" if index == 0 else None,
    )
for index, (voltage, conductance) in enumerate(didv_fit):
    plt.plot(
        voltage,
        conductance,
        color="tab:red",
        label="RSJ + MAR fit" if index == 0 else None,
    )
plt.xlabel("Junction voltage (mV)")
plt.ylabel(r"Differential conductance ($G_0$)")
plt.legend()
plt.tight_layout()
plt.show()
