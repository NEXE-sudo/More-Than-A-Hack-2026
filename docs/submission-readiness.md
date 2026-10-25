# Submission-readiness checklist

This checklist distinguishes repository evidence from requirements that cannot be verified in
this checkout. The authoritative local competition references are the sibling official
repository's [`SUBMISSION.md`](../../more-than-a-hack-2027/SUBMISSION.md),
[`DATA.md`](../../more-than-a-hack-2027/DATA.md), [`RULES.md`](../../more-than-a-hack-2027/RULES.md),
and [`EVALUATION.md`](../../more-than-a-hack-2027/EVALUATION.md).

| Check | Status | Evidence and finding |
|---|---|---|
| Required submission artifacts and naming | **Verified** | `SUBMISSION.md` requires a possession CSV, public code repository, and short technical document; it specifies team-based filenames and submission through the competition form. |
| Prediction schema, indexing, and ordering | **Verified** | `SUBMISSION.md` specifies UTF-8 CSV with `frame,label`, 199,319 rows, integer frames `0..199318` in order, labels `H/A/D/X`, and no blank cells. `src/possession/data/validation.py` validates the generated file contract; the official validator is `evaluation/validate_submission.py`. |
| Missing or invalid predictions | **Verified** | `SUBMISSION.md` says write `X` rather than a blank to abstain. `EVALUATION.md` documents that `X` on graded frames is a macro-F1 miss and is filled for spell scoring; `evaluation/run_evaluation.py` skips malformed submissions. |
| Prediction generation and evaluation commands | **Verified** | The repository documents `possession predict` and `possession evaluate` in `README.md`; the official checks are `evaluation/validate_submission.py` and `evaluation/run_evaluation.py --gt public`. |
| Exact regeneration of the final uploaded file | **Blocked** | `RULES.md` requires the repository to regenerate the exact submitted file. No team-named final CSV or technical document was designated in this review for byte-for-byte regeneration and upload-artifact validation. |
| Runtime and resource limits | **Blocked** | No numeric runtime, memory, hardware, or execution-time limits were found in the cited competition documents. Do not infer or claim compliance with unstated limits. |
| Dependencies and isolated environments | **Verified** | Root versions are pinned in `requirements.lock` and `pyproject.toml` supports Python 3.11–3.13. `docs/data_handling.md` documents the distinct Python 3.11 `video-pilot/` environment and lockfile; it is not a production dependency. |
| Permitted model methods | **Verified** | `RULES.md` permits methods and pretrained models when organizers can access and inspect them and the technical document explains them. |
| Specific pretrained-model license | **Not applicable** | No external pretrained model is integrated in the current production pipeline. Match-data terms remain separate. |
| Data access and prohibited information | **Blocked** | `DATA.md` says match assets are gated and the enrollment terms govern their use; those accepted terms are not available in this checkout. `RULES.md` prohibits using private full-match possession truth or corrected truth outside the public five-minute window and prohibits hand-produced submission labels. |
| Data, predictions, and diagnostics kept out of Git | **Verified** | `.gitignore` excludes `Data/`, `data/`, `cache/`, `outputs/`, and `artifacts/`; `.DS_Store` is ignored. At review, `git ls-files Data data outputs cache artifacts` returned no tracked paths, and representative local inputs/reports were ignored. |
| Clean-checkout test reproducibility | **Blocked** | Tests were run in the current modified checkout, not an isolated clean checkout. The project package and test discovery are configured in `pyproject.toml`; scorer-contract tests use the sibling official checkout and skip if it is absent. |

The current evidence verifies the repository's documented file contract and public evaluation
workflow, not the final upload artifact, acceptance of the data terms, full-match possession
accuracy, or compliance with any runtime limit that is not stated in the available documents.
