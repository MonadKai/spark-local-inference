# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Calculate the frozen five-block A/B/A interval, without qualifying a run.

Callers must first reconcile complete natural transcription, actual engine
observations, warmup, workload, cache conditions and isolation. These arithmetic
results cannot establish those properties or independence of the time windows.
"""

import math
from decimal import Decimal, localcontext


def paired_interval(blocks):
    """Exponentiate the two-sided t interval on five independent log ratios.

    Each block contains an ID and positive scalar measurements named
    ``vllm_before``, ``flare`` and ``vllm_after``. Units must match. Higher ratios
    mean better throughput and worse latency. No request/token pseudo-replicates
    or additional optional-stopping blocks belong in this calculation.
    """
    if len(blocks) != 5:
        raise ValueError("the frozen experiment requires exactly five blocks")
    identifiers, logs = set(), []
    for block in blocks:
        identity = block["id"]
        if not isinstance(identity, str) or not identity.strip() or identity in identifiers:
            raise ValueError("blocks require distinct nonempty IDs")
        identifiers.add(identity)
        values = [block[name] for name in ("vllm_before", "flare", "vllm_after")]
        try:
            valid = all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("every paired measurement must be finite and positive")
        # Decimal preserves exact input floats and avoids a rounded square root
        # turning an exactly equal geometric reference into a strict >1 win.
        with localcontext() as context:
            context.prec = 80
            before, candidate, after = map(Decimal, values)
            logs.append((candidate * candidate / (before * after)).ln() / 2)
    with localcontext() as context:
        context.prec = 80
        mean = sum(logs) / 5
        variance = sum((value - mean) ** 2 for value in logs) / 4
        # performance-protocol.json freezes df=4 and this two-sided 95% value.
        critical = Decimal("2.7764451051977987")
        half_width = critical * (variance / 5).sqrt()
        exact_estimate, exact_lower, exact_upper = (
            value.exp() for value in (mean, mean - half_width, mean + half_width))
        estimate, lower, upper = map(float, (exact_estimate, exact_lower, exact_upper))
        # Round binary interval endpoints outward, so conversion cannot turn a
        # boundary into a throughput win or hide a latency-budget violation.
        if Decimal(lower) > exact_lower:
            lower = math.nextafter(lower, -math.inf)
        if Decimal(upper) < exact_upper:
            upper = math.nextafter(upper, math.inf)
    if not all(math.isfinite(value) and value > 0 for value in (estimate, lower, upper)):
        raise ValueError("paired interval exceeds the representable range")
    return dict(block_ids=[block["id"] for block in blocks], block_log_ratios=list(map(float, logs)),
        block_count=5, degrees_of_freedom=4, t_critical=float(critical),
        geometric_mean_ratio=estimate, lower_95=lower, upper_95=upper)
