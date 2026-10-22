# Experiments and results

## Recorded environment

- Official checkout: `../more-than-a-hack-2027`
- Python seen during inspection: 3.13.5
- Official repository initially clean on `main`
- Gated `data/tactical/` bundle: absent; no ATD XML/text, homography or Smart Tagging files
- Public ground truth and four committed baseline CSVs: present

## Measurements

The unchanged official evaluator was run against the four committed official baseline outputs in
the public five-minute mode. These are measured local public-window scores, not a new model
experiment:

| Official baseline CSV | macroF1 | spellF1 | predicted possessions | mean duration (s) |
|---|---:|---:|---:|---:|
| smooth-125 | 0.5812 | 0.4242 | 16 | 18.5 |
| smooth-50 | 0.6207 | 0.3256 | 26 | 10.8 |
| smooth-25 | 0.6200 | 0.2868 | 36 | 7.6 |
| raw | 0.5972 | 0.0378 | 131 | 2.0 |

Official segment-bootstrap 5th–95th percentile spellF1 intervals from 1,000 draws were:

| Baseline | Public spellF1 interval |
|---|---:|
| smooth-125 | 0.2667–0.5939 |
| smooth-50 | 0.2140–0.4465 |
| smooth-25 | 0.1962–0.3849 |
| raw | 0.0189–0.0595 |

The intervals for the three smoothed candidates overlap substantially; the public sample does
not support choosing between them reliably.

Invocation from the independent project, with the official evaluator subprocess launched while the
working directory was `/tmp`:

```bash
possession evaluate --repo-root /path/to/more-than-a-hack-2027 \
  --results-dir /path/to/more-than-a-hack-2027/evaluation/results \
  --output-dir outputs/adapter-public-evaluation --ground-truth public --bootstrap 1000
```

The official `EVALUATION.md` publishes a separate full-match smooth-50 reference of spellF1
0.2433; that value was not measured here.

The official baseline reproduction command was also attempted with the configured repository data
root. It exited with status 1 and identified missing
`data/tactical/BAR-ATH_tactical_atd.xml` and `data/tactical/BAR-ATH_tactical_atd.txt`. The failure
is expected because the gated bundle is absent. Candidate predictions therefore were not run. No
candidate score, full-match score, runtime, or memory benchmark is claimed.

## Next experiments

1. Rebuild and score the official baseline from local tactical tracking, then compare feature
   extraction and frame alignment against its output byte-for-byte.
2. Evaluate switch persistence, bounded occlusion holds, and explicit dead-ball cues against the
   permitted public labels with segment bootstrap intervals; do not select settings from the
   17-possession sample alone.
3. Add reliability-weighted team geometry/Smart Tagging features and retain only changes that
   improve spell-level results under full-match validation when legitimate labels are available.
