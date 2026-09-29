# Held-out endpoint validation protocol

Frozen after the 10-seed pilot and before generating any held-out result.

The pilot ran 780 physical episodes in 453.85 seconds with four workers.
Its discounted collective-advantage contrasts were negative with simultaneous
intervals below zero in all three ecologies. A full proposed 48,600-episode
composition sweep would take roughly 7.9 hours at that observed throughput;
parallel scaling to additional workers is not guaranteed.

Use a staged validation design: first test the necessary endpoint conditions on
100 new ecological/position seeds (replicate indices 1000–1099, base 20260928),
N=64, H=1000, C=000, D=111, one uniformly sampled focal agent, k={0,63}.
This requires 1200 physical episodes. All-C and all-D population returns use all
64 agents, so reducing focal sampling does not reduce precision of the primary
collective-return contrast. Focal-based exploitation/greed/fear are less precise
than in the originally proposed eight-focal design; report that limitation.

Primary family: four endpoint contrasts across three ecologies, gamma=0.95,
95% simultaneous centered max-standardized bootstrap intervals, 5000 resamples,
bootstrap seed 1729. Secondary family: the same contrasts for undiscounted
1000-step harvest. Do not combine families into a single optimized verdict.
Zero is the prespecified theoretical threshold, not a practical equivalence
margin. For planning, pilot projected collective-gap standard errors at 100
replicates were about 0.0010–0.0013 discounted and 0.051–0.073 undiscounted.
The fixed sample size remains 100 even if an interim result appears decisive.

If collective advantage is contradicted for an ecology, stop testing this pair
in that ecology: a failed necessary condition cannot be repaired by adding
interior composition points. If an ecology remains potentially supportive,
treat fuller composition characterization as a separately planned follow-up.
The main 11-composition/eight-focal configuration remains available; it is not
claimed to have been executed by the endpoint run.

The pilot supplies exploratory Schelling curves; the held-out run supplies
endpoint validation. This is a prospective refinement of the provisional main
configuration, based on pilot cost and a failed necessary condition, not
post-selection on held-out outcomes. No incentives, policies, discount or horizon
are revised. No model change is authorized by a failure alone.
