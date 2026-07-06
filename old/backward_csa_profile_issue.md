# Why the Current Backward CSA Candidate Enumeration Is Not Optimal

## Context

The project uses a scheduled journey planner based on the Connection Scan Algorithm (CSA). The planner currently has a method called `plan_candidates(...)` that tries to return multiple route candidates sorted by latest departure time before a requested arrival deadline.

The final project requirement is not just to find *one* route. It needs a list of valid routes between origin and destination, sorted from latest departure to earliest departure, where each route arrives before the deadline and can later be evaluated for robustness/confidence.

In other words, before adding delay risk, the scheduled route candidate generation should approximate or match the scheduled Pareto frontier:

```text
maximize departure time
minimize arrival time
```

A route `(dep1, arr1)` dominates another route `(dep2, arr2)` if:

```text
dep1 >= dep2 and arr1 <= arr2
```

with at least one strict improvement.

So if one route departs later and arrives no later, the other route should not be kept.

---

## Validation Setup

A simple forward earliest-arrival CSA was used as an oracle. For a given departure time, it computes the earliest possible arrival time. Sampling many departure times over a bounded window gives a scheduled Pareto frontier.

The validation query was:

```python
day = planner._day_from_travel_date("2026-05-18")
deadline = 9 * 3600

gold = forward_profile(
    planner,
    8579238,
    8501214,
    deadline - 3 * 3600,
    deadline,
    day,
    max_walk_m=500,
    step=60,
    deadline=deadline,
)

mine = planner.plan_candidates(
    8579238,
    8501214,
    "2026-05-18",
    "09:00",
    max_routes=100,
    max_walk_m=500,
)
```

The important detail is that the forward oracle was corrected so that:

- initial boarding at the origin does **not** require `min_transfer_secs`;
- transfers after alighting **do** require `min_transfer_secs`;
- only arrivals before the deadline are kept.

---

## First Observed Problem: Fixed Arrival Deadline Returned Dominated Routes

Before the fix, `plan_candidates()` repeatedly called backward CSA with the original deadline `09:00`. It only decreased the departure cutoff.

Simplified version of the old logic:

```python
current_deadline = deadline_secs
# but this was not actually passed to route()

departure_cutoff = deadline_secs

while len(routes) < max_routes:
    result = self.route(
        start_stop_id,
        end_stop_id,
        deadline_secs,       # BUG: always original 09:00 deadline
        departure_cutoff,
        ...
    )

    # after finding one route
    departure_cutoff = departure_secs - 1
```

This asks a sequence of questions like:

```text
1. Find latest departure that arrives before 09:00.
2. Find latest departure before previous departure that arrives before 09:00.
3. Find latest departure before previous departure that arrives before 09:00.
```

That does **not** enumerate the Pareto frontier. It just finds many earlier alternatives that still satisfy the same loose deadline.

Example output:

```text
Backward CSA returned:
  dep 08:35:00  arr 08:58:00
  dep 08:34:00  arr 08:58:00
  dep 08:33:00  arr 08:58:00
  dep 08:30:00  arr 08:58:00
  dep 08:29:00  arr 08:58:00
  dep 08:28:00  arr 08:58:00
  dep 08:26:00  arr 08:58:00
  dep 08:18:28  arr 08:58:00
```

After Pareto filtering, all of these collapse to just:

```text
08:35:00 -> 08:58:00
```

because `08:35 -> 08:58` dominates all earlier departures that arrive at the same time.

Meanwhile, the forward oracle found many useful earlier-arriving alternatives:

```text
08:35:00 -> 08:58:00
08:30:00 -> 08:53:00
08:26:00 -> 08:50:00
08:24:00 -> 08:46:00
08:20:00 -> 08:43:00
...
```

### Root Cause

Only the departure cutoff was tightened. The arrival deadline was not tightened.

### First Fix

After each accepted route, also tighten the current arrival deadline:

```python
current_deadline = route["arrival_secs"] - 1
departure_cutoff = departure_secs - 1
```

and make sure the backward route call uses `current_deadline`:

```python
result = self.route(
    start_stop_id,
    end_stop_id,
    current_deadline,      # not deadline_secs
    departure_cutoff,
    connections,
    dep_secs_list,
    max_walk_m,
    earliest_dep=earliest_dep,
    end_idx=end_idx,
    start_idx=start_idx,
    trip_reachable=trip_reachable,
)
```

Also, the validity check must compare against `current_deadline`, not the original deadline:

```python
if route is None or route["arrival_secs"] > current_deadline:
    departure_cutoff = departure_secs - 1
    continue
```

---

## Second Observed Problem: Backward CSA Still Misses Better Same-Departure Routes

After correctly tightening the arrival deadline, the output improved a lot:

```text
Backward plan_candidates raw routes:
  dep 08:35:00  arr 08:58:00
  dep 08:30:00  arr 08:54:00
  dep 08:26:00  arr 08:50:00
  dep 08:24:00  arr 08:46:00
  dep 08:20:00  arr 08:43:00
  dep 08:17:00  arr 08:39:00
  ...
```

But it still did not fully match the forward oracle.

Example mismatch:

```text
Forward oracle / gold:
  dep 08:30:00  arr 08:53:00

Backward result:
  dep 08:30:00  arr 08:54:00
```

The debugged backward route was:

```text
Route dep 08:30:00 arr 08:54:00

Step 1:
  ride m2
  from 8579238 to 8591818
  departure 08:30:00
  arrival   08:38:00

Step 2:
  ride m1
  from 8591818 to 8501214
  departure 08:42:00
  arrival   08:54:00
```

This route is valid, but it is not optimal, because the forward oracle found another route with the same departure time and earlier arrival:

```text
08:30:00 -> 08:53:00
```

### Why This Happens

The backward search optimizes this objective:

```text
latest departure subject to arriving before current_deadline
```

It does **not** necessarily optimize this:

```text
earliest arrival for that same departure
```

So when the backward scan finds:

```text
08:30 -> 08:54
```

it considers that acceptable for the current deadline. Then the loop tightens both limits:

```python
current_deadline = 08:53:59
departure_cutoff = 08:29:59
```

This blocks the planner from later discovering:

```text
08:30 -> 08:53
```

because that better route has the **same departure time**, but the departure cutoff has already moved below `08:30`.

### Key Insight

The backward loop can find valid routes, but it is not guaranteed to find the exact scheduled Pareto frontier unless it is implemented as a proper CSA profile algorithm or enhanced with extra same-departure refinement.

---

## Why This Matters for the Robust Planner

The final project adds delay/confidence estimation on top of scheduled routes.

If the scheduled candidate generator misses good routes or keeps slightly worse versions, the confidence layer is built on a weak base.

For example:

```text
Correct scheduled candidate:
  08:30 -> 08:53

Returned scheduled candidate:
  08:30 -> 08:54
```

This may seem like only one minute, but it affects:

- slack time before the deadline;
- transfer feasibility;
- confidence calculation;
- route ranking;
- scientific validation.

For a robust planner, correctness and explainability are more important than having a clever but fragile enumeration method.

---

## Recommended Solution: Forward CSA Over a Bounded Window

Use forward earliest-arrival CSA as the reliable candidate generator.

Pipeline:

```text
1. Choose a bounded departure window, e.g. deadline - 3h to deadline.
2. Sample departure times in that window.
3. For each sampled departure time, run forward earliest-arrival CSA.
4. Keep only routes arriving before the requested deadline.
5. Pareto-filter by latest departure and earliest arrival.
6. Reconstruct only the final Pareto routes.
7. Apply delay/confidence evaluation.
8. Sort by latest departure first.
```

Pseudo-code:

```python
candidates = []

for dep_time in range(t_min, deadline_secs + 1, step_seconds):
    arr_time = forward_eat(
        planner,
        start_id,
        end_id,
        dep_time,
        day,
        max_walk_m,
    )

    if arr_time <= deadline_secs:
        candidates.append((dep_time, arr_time))

frontier = pareto_filter_latest_dep_earliest_arr(candidates)

routes = []
for dep_time, arr_time in frontier:
    route = forward_route_with_reconstruction(
        planner,
        start_id,
        end_id,
        dep_time,
        day,
        max_walk_m,
    )
    routes.append(route)
```

---

## Optimization: Coarse-to-Fine Forward Scan

A naive forward scan every minute over three hours requires:

```text
181 forward scans
```

That may still be acceptable in a small region, but it can be optimized without sacrificing correctness too much.

Recommended approach:

```text
Stage 1: coarse scan every 5 minutes
Stage 2: refine around candidate frontier changes every 1 minute
Stage 3: Pareto-filter again
Stage 4: reconstruct only final routes
```

Pseudo-code:

```python
coarse_step = 300      # 5 minutes
fine_step = 60         # 1 minute
refine_radius = 300    # +/- 5 minutes

coarse_pairs = []

for dep_time in range(t_min, deadline_secs + 1, coarse_step):
    arr_time = forward_eat(planner, start_id, end_id, dep_time, day, max_walk_m)
    if arr_time <= deadline_secs:
        coarse_pairs.append((dep_time, arr_time))

coarse_frontier = pareto_filter_latest_dep_earliest_arr(coarse_pairs)

fine_times = set()
for dep_time, arr_time in coarse_frontier:
    for t in range(dep_time - refine_radius, dep_time + refine_radius + 1, fine_step):
        if t_min <= t <= deadline_secs:
            fine_times.add(t)

fine_pairs = []
for dep_time in sorted(fine_times):
    arr_time = forward_eat(planner, start_id, end_id, dep_time, day, max_walk_m)
    if arr_time <= deadline_secs:
        fine_pairs.append((dep_time, arr_time))

final_frontier = pareto_filter_latest_dep_earliest_arr(coarse_pairs + fine_pairs)
```

This keeps the method simple and defensible while reducing runtime.

---

## Important Performance Optimization: Arrival-Only First, Reconstruction Later

Do **not** reconstruct the full route for every sampled departure time.

Use two functions:

```python
forward_eat(...)
```

returns only the earliest arrival time.

```python
forward_route(...)
```

tracks predecessors and reconstructs the actual route.

Recommended flow:

```text
many cheap forward_eat scans
-> Pareto filtering
-> few expensive forward_route reconstructions
```

This is a strong practical optimization because most sampled departure times will not survive the Pareto filter.

---

## Suggested Explanation for Report / Video

A clear and honest explanation would be:

```text
We initially implemented an iterative backward CSA candidate generator to quickly find latest-departure routes before a deadline. During validation, we compared it against a simple forward earliest-arrival CSA oracle. This revealed that the backward loop could return valid routes, but it sometimes missed scheduled Pareto-optimal routes or returned a later-arriving route for the same departure time.

The issue is that the backward loop optimizes latest departure under a deadline, but does not necessarily minimize arrival time for each departure. Tightening the deadline helped remove dominated routes, but did not fully solve same-departure optimality issues.

For correctness and scientific validation, we therefore use forward CSA over a bounded departure window, followed by Pareto filtering. To keep runtime reasonable, we use arrival-only scans first and reconstruct full route details only for the final Pareto candidates. Optionally, we optimize the scan using a coarse-to-fine schedule.
```



# Bacward scan problem


## Why the backward CSA search is not symmetric with forward CSA

At first, the backward scan looks like the exact reverse of forward CSA:

```text
Forward CSA:
Given a departure time τ, find the earliest arrival.

Backward CSA:
Given an arrival deadline T, find the latest departure.

However, these are not fully symmetric for our planner, because the route we need is not only “any route that leaves as late as possible.” We need the best scheduled route according to the project objective:

primary objective: leave as late as possible
secondary objective: arrive as early as possible / keep more slack

So even for a single returned route, a route like this is not ideal:

08:30 -> 08:54

if another route exists with the same departure but earlier arrival:

08:30 -> 08:53

The backward scan may return the first route because it only proves that 08:30 is a feasible latest departure before the deadline. It does not necessarily prove that the selected route is the earliest-arriving route among all journeys departing at 08:30.

This is the key asymmetry.

Forward CSA stores, for each stop:

earliest known arrival time

That is safe because if we reach a stop earlier, that state dominates reaching the same stop later: from an earlier time we can always wait and still catch everything the later state could catch. The CSA paper describes the earliest-arrival variant exactly this way: connections are scanned in increasing departure time, and the stop array stores tentative earliest arrival times.

But our backward implementation stored, for each stop:

latest known feasible departure time

This is not enough to reconstruct the best journey. A later feasible departure does not tell us whether the suffix route after that point gives the earliest possible arrival, the best transfer slack, or the best route structure. It only tells us that reaching the target before the deadline is possible.

In the CSA paper, the backward scan used for profile queries is not a simple reverse earliest-arrival scan. The paper scans connections decreasing by departure time, but it computes an arrival value for starting in each connection and incorporates that into stop profiles. The profile algorithm stores departure/arrival pairs, not only one latest-departure value.

So the correct paper-style backward method is closer to:

S[stop] = [
    (departure_time, arrival_time),
    (departure_time, arrival_time),
    ...
]

not:

S[stop] = latest_departure_time

That is why our simplified backward scan can fail even for one selected route.

What we tried

We implemented an iterative backward CSA candidate generator. The idea was:

1. Start from the arrival deadline.
2. Scan connections backward.
3. Find the latest departure that can still reach the target.
4. Reconstruct that route.
5. Tighten the cutoff and repeat.

In the first version, only the departure cutoff was tightened. This produced dominated routes such as:

08:35 -> 08:58
08:34 -> 08:58
08:33 -> 08:58
08:30 -> 08:58

These are not useful because 08:35 -> 08:58 dominates all earlier departures with the same arrival time. Our notes identify this as the first problem: the arrival deadline remained too loose while only the departure cutoff changed.

We then fixed that by also tightening the arrival deadline after each accepted route:

current_deadline = route["arrival_secs"] - 1
departure_cutoff = departure_secs - 1

This improved the results, but it still did not guarantee optimality. The validation found this mismatch:

Forward oracle:
08:30 -> 08:53

Backward result:
08:30 -> 08:54

The backward result is valid, but it is not the best route for that departure. The reason is that the backward scan optimized:

latest departure subject to arriving before the current deadline

but it did not optimize:

earliest arrival among all routes with that same latest departure

Once the scan accepted 08:30 -> 08:54, the loop moved the cutoff below 08:30, which prevented it from later finding the better 08:30 -> 08:53 route. This exact failure is documented in our validation notes.

Why we cannot just use this backward search as the final method

The backward search from the CSA paper is a profile algorithm, not the simplified latest-departure scan we implemented.

The paper’s backward profile scan works because it keeps multiple non-dominated labels:

(departure_time, arrival_time)

Our attempted backward scan kept only a compressed state:

latest feasible departure

That compression loses information. It can prove that a route is feasible before the deadline, but it cannot guarantee that the reconstructed route is the best route for that departure time.

Therefore, the issue is not that “backward CSA is impossible.” The issue is:

A simple backward latest-departure scan is not equivalent to the CSA profile algorithm.

To make backward CSA correct, we would need to implement the full profile version with non-dominated (departure, arrival) labels and predecessor tracking. Since we already validated that the simpler backward version can return a worse route even for one candidate, we should not use it as the final candidate generator.
