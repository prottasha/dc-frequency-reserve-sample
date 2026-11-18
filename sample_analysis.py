"""
Sample analysis file for Appendix A.
Part 1: illustrative system-frequency-response (SFR) example with ASSUMED parameters.
Part 2: simulation-based power for the H2 three-outcome decision rule (difference model), safety bound, run counts.
Part 3: planned reporting format on DUMMY paired differences, analysed with the difference model.
Part 4: inter-coder agreement (Cohen's kappa) on DUMMY interview codes.
Part 5: variance pilot on DUMMY data: day variance with interval, interval half-width at its upper end.
Nothing here is a result of the proposed study.
"""
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(2026)
F0 = 50.0


# ---------------------------------------------------------------- Part 1
def sfr(p_dc, flex=0.20, r_dc=0.01, h_v=5.0, delay=0.10, t_act=0.05,
        load=4200.0, s_sync=2800.0, h=4.5, loss=400.0, s_lost=450.0,
        r_gov=0.05, t_g=0.2, t_t=1.0, f_gt=0.65, t_st=10.0, d=1.0, headroom=600.0,
        ufls=(49.3, 49.1, 48.9, 48.7, 48.5), shed=0.05, t_relay=0.2,
        dt=0.001, t_end=30.0):
    """Single-area SFR with under-frequency load shedding (UFLS). Powers in MW, frequency deviation in Hz.
    UFLS: five stages as in Kahramaa's scheme (Talfat et al. 2024); first stage 49.3 Hz and 200 mHz spacing assumed; 5% of load per stage and a
    0.2 s definite-time delay are ASSUMED until Kahramaa's settings are available."""
    n = int(t_end / dt)
    ek = h * (s_sync - s_lost)                  # MW*s left after the unit trips
    s_on = s_sync - s_lost
    df = np.zeros(n)
    valve = turb = steam = pdc = 0.0
    nd = int(delay / dt)
    rocof_f = 0.0                               # filtered df/dt seen by L0
    below = np.zeros(len(ufls))                 # time spent below each stage threshold
    tripped = np.zeros(len(ufls), dtype=bool)
    for k in range(1, n):
        t = k * dt
        dP_loss = loss if t >= 1.0 else 0.0
        # governor and turbine
        ref = np.clip(-(s_on / r_gov) * df[k - 1] / F0, -headroom, headroom)
        valve += dt * (ref - valve) / t_g
        turb += dt * (valve - turb) / t_t          # gas-turbine path (65% of response)
        steam += dt * (ref - steam) / t_st         # slow steam path of combined-cycle units
        pm = f_gt * turb + (1 - f_gt) * steam
        # data-center L0 law on delayed local measurement
        kd = max(k - 1 - nd, 0)
        raw = (df[kd] - df[max(kd - 100, 0)]) / (100 * dt) if kd >= 100 else 0.0
        rocof_f = raw                           # 100 ms moving-window estimate
        cmd = -(p_dc / r_dc) * df[kd] / F0 - 2 * h_v * p_dc * rocof_f / F0
        lim = max(flex * p_dc, 1e-9)
        cmd = lim * np.tanh(cmd / lim)            # smooth saturation at the flexible share
        pdc += dt * (cmd - pdc) / t_act
        for j, th in enumerate(ufls):
            if not tripped[j]:
                below[j] = below[j] + dt if F0 + df[k - 1] < th else 0.0
                tripped[j] = below[j] >= t_relay
        p_shed = shed * load * tripped.sum()
        ddf = F0 / (2 * ek) * (pm + pdc + p_shed - dP_loss - d * load * df[k - 1] / F0)
        df[k] = df[k - 1] + dt * ddf
    tt = np.arange(n) * dt
    i0 = int(1.0 / dt)
    w = int(0.5 / dt)
    rocof = np.min((df[i0 + w:i0 + 5 * w] - df[i0:i0 + 4 * w]) / 0.5)
    return tt, df, df.min(), rocof, int(tripped.sum())


print("PART 1  Illustrative SFR example (assumed parameters, not results)")
print("fleet_MW  flexible_MW  nadir_Hz  RoCoF_Hz_per_s(500ms)  UFLS_stages  nadir_without_UFLS_Hz")
fig, ax = plt.subplots(figsize=(6.3, 3.3), dpi=300)
styles = {0: ("#555555", "-"), 50: ("#7aa6c2", ":"), 500: ("#1b6ca8", "--"), 1500: ("#0b3c5d", "-")}
rows = []
TRACES = {}
for p in (0, 50, 500, 1500):
    tt, df, nadir, rocof, nst = sfr(p)
    TRACES[p] = (tt, F0 + df)
    FIG3_SUMMARY = globals().setdefault('FIG3_SUMMARY', [])
    _, _, nadir0, _, _ = sfr(p, ufls=())
    rows.append((p, 0.2 * p, F0 + nadir, rocof, nst))
    FIG3_SUMMARY.append((p, 0.2 * p, F0 + nadir, rocof, nst))
    print(f"{p:8d}  {0.2*p:11.0f}  {F0+nadir:8.3f}  {rocof:8.3f}  {nst:11d}  {F0+nadir0:8.3f}")
    c, ls = styles[p]
    lab = "No data-center support" if p == 0 else f"{p:,} MW fleet ({0.2*p:.0f} MW flexible)"
    ax.plot(tt, F0 + df, color=c, ls=ls, lw=1.6, label=lab)
ax.axhline(49.3, color="#b03a2e", lw=0.9, ls=(0, (4, 3)))
ax.text(19.8, 49.285, "first load-shedding stage, 49.3 Hz", color="#b03a2e", fontsize=7, ha="right", va="top")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlim(0, 20)
ax.grid(alpha=0.3)
ax.set_ylim(49.05, 50.55)
ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_sfr.png")

# ---------------------------------------------------------------- Part 2
print("\nPART 2  Power for the H2 decision rule (pre-registered secondary test), difference model")
print("        d = mean + day + seed + scenario(day) + residual, where d is the paired nadir difference (mHz)")
print("        between the proposed controller and B5 in the same scenario with the same seed,")
print("        over the stressed stratum: 15 days x 6 scenarios = 90 scenarios")
D, S, K = 15, 10, 6           # operating days, training seeds, scenarios per day in the stressed stratum
                              # (unit trip, interconnection open, fleet 500 or 1,500 MW, three solar levels)
SESOI = 20.0                  # smallest improvement of interest, mHz (provisional until Q1)


def se_mean(sd_day, sd_seed, sd_scn, sd_err):
    """Standard error of the mean paired difference under the difference model."""
    return np.sqrt(sd_day**2 / D + sd_seed**2 / S + sd_scn**2 / (D * K) + sd_err**2 / (D * K * S))


def crossed_fit(y):
    """Mean, standard error and degrees of freedom for a days x seeds table of cell means of paired differences.
    Moment estimator with Satterthwaite degrees of freedom; the study itself fits the same
    model by restricted maximum likelihood with Kenward-Roger degrees of freedom."""
    d, s_ = y.shape
    gm = y.mean()
    ms_a = s_ * y.mean(axis=1).var(ddof=1)
    ms_b = d * y.mean(axis=0).var(ddof=1)
    res = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + gm
    ms_ab = (res**2).sum() / ((d - 1) * (s_ - 1))
    var = max(ms_a + ms_b - ms_ab, 1e-12) / (d * s_)
    num = (ms_a + ms_b - ms_ab) ** 2
    den = ms_a**2 / (d - 1) + ms_b**2 / (s_ - 1) + ms_ab**2 / ((d - 1) * (s_ - 1))
    df = max(num / den, 1.0)
    return gm, np.sqrt(var), df


def crossed_ci(y):
    gm, se, df = crossed_fit(y)
    return gm, stats.t.ppf(0.975, df) * se


def sim_intervals(sd_day, sd_seed, sd_scn, sd_err, reps=4000):
    """Simulate the centred mean and the half-width. The model is a location shift, so the
    interval for a true difference delta is delta + gm0 +/- hw."""
    gm0, hw = np.empty(reps), np.empty(reps)
    for r in range(reps):
        a_ = np.sqrt(sd_day**2 + sd_scn**2 / K) * rng.standard_normal((D, 1))
        c_ = sd_seed * rng.standard_normal((1, S))
        e_ = sd_err / np.sqrt(K) * rng.standard_normal((D, S))
        gm0[r], hw[r] = crossed_ci(a_ + c_ + e_)
    return gm0, hw


def outcome_shares(delta, gm0, hw):
    sup = np.mean(delta + gm0 - hw > SESOI)      # interval wholly above 20 mHz
    notsup = np.mean(delta + gm0 + hw < SESOI)   # interval wholly below 20 mHz
    return sup, 1.0 - sup - notsup, notsup


print("Decision rule: 95% interval wholly above 20 mHz = supported; spanning 20 = inconclusive; wholly below = not supported")
print("DUMMY variance components (mHz): day, seed, scenario, residual")
print("components          SE   mean half-width  true diff for 'supported' at 90%  true diff for 'not supported' at 90%")
keep = {}
for comp in ((4, 5, 15, 10), (8, 5, 15, 10), (16, 5, 15, 10), (8, 10, 15, 10), (16, 10, 15, 10)):
    gm0, hw = sim_intervals(*comp)
    keep[comp] = (gm0, hw)
    up = SESOI + np.quantile(hw - gm0, 0.90)
    down = SESOI - np.quantile(hw + gm0, 0.90)
    print(f"{str(comp):18s} {se_mean(*comp):5.2f}  {hw.mean():8.1f}         {up:8.1f}                         {down:8.1f}")
gm0, hw = keep[(8, 5, 15, 10)]
print("outcome probabilities for components (8, 5, 15, 10): true difference -> supported / inconclusive / not supported")
for delta in (0, 10, 15, 20, 25, 30, 40):
    a_, b_, c_ = outcome_shares(delta, gm0, hw)
    print(f"  {delta:3d} mHz -> {a_:.2f} / {b_:.2f} / {c_:.2f}")
print("A true difference equal to the margin is inconclusive about 95% of the time at ANY sample size (property of the rule).")
print("What sample size sets is the band in which the verdict is probably inconclusive:")
_rng_main = rng
rng = np.random.default_rng(11)
for (d_, s_n) in ((15, 10), (25, 15)):
    D, S = d_, s_n
    for comp in ((8, 5, 15, 10), (16, 10, 15, 10)):
        g_, h_ = sim_intervals(*comp)
        print(f"  {d_} days, {s_n} seeds, components {comp}: band {SESOI - np.quantile(h_ + g_, 0.90):.1f} to "
              f"{SESOI + np.quantile(h_ - g_, 0.90):.1f} mHz; mean half-width {h_.mean():.1f} mHz")
D, S = 15, 10
rng = _rng_main
print(f"rule of three, 1000 Latin-hypercube scenarios: {3/1000:.4f}")

# run ledger. Temperature sweep: 16 temperatures x 20 trace days x 2 battery housings = 640 conditions.
cond_T = 16 * 20 * 2
bisect = 8                                   # bisection steps per value: resolution 1/256 of rated IT load
base = [  # (label, runs, seconds per run)
    ("core, 540 scenarios x 7 controllers x 10 seeds x 2 networks", 540 * 7 * 10 * 2, 90),
    ("core at the strictest accepted thresholds, P and B5, Qatar model (540 x 2 x 10)", 540 * 2 * 10, 90),
    ("temperature sweep, reference sizing (640 conditions x 2 preparations x 8 bisection steps)", cond_T * 2 * bisect, 30),
    ("temperature sweep at the strictest accepted thresholds (640 x 2 x 8)", cond_T * 2 * bisect, 30),
    ("battery size and hold time: 11 further combinations of 10/15/30-min battery and 30-s/5/15/30-min hold "
     "(16 temperatures x 20 days x 11 x 8)", 16 * 20 * 11 * bisect, 30),
    ("chiller-model uncertainty: 10 Latin-hypercube draws of chiller parameters (16 x 20 x 10 x 8)", 16 * 20 * 10 * bisect, 30),
    ("liquid-cooled case: warm-water loop (ASHRAE W32), reference sizing, rule-based preparation (16 x 20 x 8)", 16 * 20 * bisect, 30),
    ("oscillation sweep (32 cases x 6 conditions x 10 seeds x 4 comparators)", 32 * 6 * 10 * 4, 90),
    ("delay and noise (18 scenarios x 8 off-nominal settings x 10 seeds x 4 controllers)", 18 * 8 * 10 * 4, 90),
    ("safety sample (1,000 scenarios x 10 seeds)", 10000, 90),
    ("fault injection (6 x 18 x 10 x 2)", 6 * 18 * 10 * 2, 90),
    ("pilot (15 days x 36 scenarios x 3 controllers x 3 realisations)", 15 * 36 * 3 * 3, 90),
]
optional = [
    ("transfer test: Qatar-trained P, B4, B6 run unchanged on the benchmark (540 x 3 x 10)", 540 * 3 * 10, 90),
    ("learned preparation in the temperature sweep (640 x 10 seeds x 8)", cond_T * 10 * bisect, 30),
    ("curtailed-solar response, P and B5, Qatar model, 6,000 MW level (180 x 2 x 10)", 180 * 2 * 10, 90),
]
conditional = [
    ("GPU-only variant if a participant refuses battery discharge (540 x 2 x 10 x 2)", 540 * 2 * 10 * 2, 90),
    ("second must-run bound on the Qatar model if no list is supplied (540 x 7 x 10)", 540 * 7 * 10, 90),
]
print("RUN LEDGER")
for title, rows in (("base plan", base), ("optional extensions", optional), ("conditional additions", conditional)):
    tot = sum(r[1] for r in rows)
    hrs = sum(r[1] * r[2] for r in rows) / 3600
    print(f" {title}:")
    for lab, n_, sec_ in rows:
        print(f"  {n_:7d}  {lab}")
    print(f"  {tot:7d}  total, {hrs:.0f} core-hours ({hrs/64:.0f} h on 64 cores); x10 margin {hrs*10:.0f} core-hours "
          f"({hrs*10/64/24:.1f} days on 64 cores)")
print(f" training: 60 runs (3 learned variants, P, B4 and B6, x 10 seeds x 2 networks, each trained from scratch) "
      f"at 12 h on 1 GPU + 32 CPU cores: {60*12} GPU-hours, {60*12*32} core-hours; CPU-bound")
test_hrs = sum(r[1] * r[2] for r in base) / 3600
print(f" test runs + training: {test_hrs + 60*12*32:.0f} core-hours ({(test_hrs + 60*12*32)/64/24:.1f} days on 64 cores); "
      f"x10 test runs: {test_hrs*10 + 60*12*32:.0f} core-hours ({(test_hrs*10 + 60*12*32)/64/24:.1f} days)")

# ---------------------------------------------------------------- Part 3
print("\nPART 3  Planned reporting format on DUMMY paired nadir differences (mHz)")
print("        generated from the difference model (15 days x 10 seeds x 6 stressed-stratum scenarios) and analysed with it")


def sim_cells(delta, sd_day, sd_seed, sd_scn, sd_err):
    a_ = np.sqrt(sd_day**2 + sd_scn**2 / K) * rng.standard_normal((D, 1))
    c_ = sd_seed * rng.standard_normal((1, S))
    e_ = sd_err / np.sqrt(K) * rng.standard_normal((D, S))
    return delta + a_ + c_ + e_


def rank_biserial(dv):
    dv = dv[dv != 0]
    r = stats.rankdata(np.abs(dv))
    return (r[dv > 0].sum() - r[dv < 0].sum()) / r.sum()


dummy = {                       # true mean difference, then day, seed, scenario, residual SD (all invented)
    "B5 tuned rule-based":       (9.0, 8, 5, 15, 10),
    "B2 L0 only":                (37.0, 8, 5, 15, 10),
    "B3 L0+L1":                  (15.0, 8, 5, 15, 10),
    "B4 L0+L1+MARL no shield":   (-3.0, 12, 12, 20, 10),
}
out, pvals = [], []
CELLS = {}
for name, (delta, *comp) in dummy.items():
    y = sim_cells(delta, *comp)
    CELLS[name] = y
    gm, se, df = crossed_fit(y)
    hw_ = stats.t.ppf(0.975, df) * se
    p = 2 * stats.t.sf(abs(gm / se), df)
    pvals.append(p)
    out.append((name, gm, gm - hw_, gm + hw_, rank_biserial(y.ravel())))
# H2 (B5) is decided by the interval rule of Table 3 and is not in the Holm family;
# Holm is applied to the secondary comparisons only.
sec = [i for i, o in enumerate(out) if not o[0].startswith("B5")]
holm = {}
run = 0.0
for rank, i in enumerate(sorted(sec, key=lambda k: pvals[k])):
    run = max(run, (len(sec) - rank) * pvals[i])
    holm[i] = min(run, 1.0)
print("comparison (proposed minus)   mean_diff_mHz  95%CI_mHz        r_rb    p_Holm   reading")
for i, (name, md, lo, hi, rrb) in enumerate(out):
    if name.startswith("B5"):
        reading = ("H2 supported" if lo > SESOI else "H2 not supported" if hi < SESOI else "H2 inconclusive")
        pstr = "n/a"
    else:
        reading = "secondary"
        pstr = f"{holm[i]:.3g}"
    print(f"{name:28s} {md:8.1f}      [{lo:6.1f},{hi:6.1f}]  {rrb:6.2f}  {pstr:>7s}   {reading}")

# ---------------------------------------------------------------- Part 4
print("\nPART 4  Cohen's kappa on DUMMY double-coded interview segments (n=60, 5 codes)")
codes = 5
a = rng.integers(0, codes, 60)
b = a.copy()
flip = rng.random(60) < 0.15
b[flip] = rng.integers(0, codes, flip.sum())
po = (a == b).mean()
pe = sum((a == c).mean() * (b == c).mean() for c in range(codes))
print(f"observed agreement={po:.3f} expected={pe:.3f} kappa={(po-pe)/(1-pe):.3f}")

# ---------------------------------------------------------------- Part 5
print("\nPART 5  Variance pilot on DUMMY paired differences of two non-learned controllers")
print("        (15 days x 36 scenarios x 3 noise realisations; true day SD = 8 mHz)")
n_day, n_scn, n_rep = 15, 36, 3
yy = (12.0 + 8.0 * rng.standard_normal((n_day, 1, 1)) + 15.0 * rng.standard_normal((n_day, n_scn, 1))
      + 10.0 * rng.standard_normal((n_day, n_scn, n_rep))).mean(axis=2)          # day x scenario means


def day_sd(y):
    k = y.shape[1]
    msb = k * y.mean(axis=1).var(ddof=1)
    msw = y.var(axis=1, ddof=1).mean()
    return np.sqrt(max((msb - msw) / k, 0.0))


est = day_sd(yy)
boot = [day_sd(yy[rng.integers(0, n_day, n_day)]) for _ in range(5000)]
lo, hi = np.percentile(boot, [5, 95])
print(f"day SD estimate={est:.1f} mHz  90% interval=[{lo:.1f}, {hi:.1f}]")
for lab, sd in (("estimate", est), ("upper end", hi)):
    g_, h_ = sim_intervals(sd, 5, 15, 10, reps=2000)
    print(f"mean half-width of the 95% interval at {lab}: {h_.mean():.1f} mHz; 'supported' at 90% needs a true difference of "
          f"{SESOI + np.quantile(h_ - g_, 0.90):.1f} mHz (seed SD assumed 5 mHz until policies are trained)")
print("pilot rule: enlarge the design (days up to 25, seeds up to 15) if the mean half-width at the upper end exceeds 10 mHz")

# ---------------------------------------------------------------- Export of the dummy data
# Writes every dataset generated above to CSV. No random numbers are drawn here,
# so the printed results are identical with or without this block.
import csv, os
DD = os.path.join(OUT, "dummy_data")
os.makedirs(DD, exist_ok=True)

with open(os.path.join(DD, "fig3_frequency_traces.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s"] + [f"freq_Hz_fleet_{p}MW" for p in TRACES])
    tt0 = TRACES[0][0]
    for k in range(0, len(tt0), 10):          # every 10th step = 10 ms resolution
        w.writerow([f"{tt0[k]:.3f}"] + [f"{TRACES[p][1][k]:.5f}" for p in TRACES])

with open(os.path.join(DD, "fig3_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fleet_MW", "flexible_MW", "nadir_Hz", "RoCoF_Hz_per_s_500ms", "UFLS_stages_operated"])
    for r in FIG3_SUMMARY:
        w.writerow([r[0], f"{r[1]:.0f}", f"{r[2]:.3f}", f"{r[3]:.3f}", r[4]])

with open(os.path.join(DD, "table10_paired_nadir_differences.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["comparison_proposed_minus", "day", "seed", "mean_nadir_difference_mHz_over_6_scenarios"])
    for name, y in CELLS.items():
        for d in range(y.shape[0]):
            for sd in range(y.shape[1]):
                w.writerow([name, d + 1, sd + 1, f"{y[d, sd]:.3f}"])

with open(os.path.join(DD, "table10_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["comparison_proposed_minus", "mean_diff_mHz", "ci95_low", "ci95_high", "rank_biserial", "p_Holm", "reading"])
    for i, (name, md, lo, hi, rrb) in enumerate(out):
        if name.startswith("B5"):
            reading = ("H2 supported" if lo > SESOI else "H2 not supported" if hi < SESOI else "H2 inconclusive")
            pstr = "n/a"
        else:
            reading, pstr = "secondary", f"{holm[i]:.3g}"
        w.writerow([name, f"{md:.1f}", f"{lo:.1f}", f"{hi:.1f}", f"{rrb:.2f}", pstr, reading])

with open(os.path.join(DD, "coding_reliability_segments.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["segment", "coder1_code", "coder2_code"])
    for k in range(len(a)):
        w.writerow([k + 1, int(a[k]), int(b[k])])

with open(os.path.join(DD, "variance_pilot_day_by_scenario.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["day", "scenario", "mean_difference_mHz_over_3_noise_realisations"])
    for d in range(yy.shape[0]):
        for c in range(yy.shape[1]):
            w.writerow([d + 1, c + 1, f"{yy[d, c]:.3f}"])

print(f"\nDummy data written to {DD}/ (6 CSV files). All values are synthetic.")
