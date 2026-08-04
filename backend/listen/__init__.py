"""Blinded listening-test tooling.

THE HARNESS THAT GATES EVERY QUALITY CLAIM THIS PROJECT MAKES. The plan's top
risk is proxy-metric capture — the team optimising whichever number moves
easiest — and its stated rule is that no automated metric may gate CI until it
correlates with human MOS at rho > 0.6. Nothing can be calibrated against human
judgement until there is a way to collect human judgement, so this is the
blocker for that entire chain.

Running a panel needs people. Building the harness does not, and until it
exists the panel cannot be run at all.

Five pieces, in the order they are used:

    match       loudness matching, VERIFIED on the written files
    session     the blinded, seeded, randomised session and its private key
    screen      the plan's listener screen, rendered by the real engine
    score       two rating scales, listener-clustered intervals, inconsistency
    correlate   Spearman rho against automated proxies, honest about n

``serve`` and ``interface.html`` are the local interface for taking the test.
"""
