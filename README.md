# Calibrating Word-Level Confidence in Bangla Speech Recognition Across Regional Dialects

Code and aggregated results for the paper of the same name (preprint: [paper/arxiv/main.pdf](paper/arxiv/main.pdf)).

**Author:** Ismail Hossain Polas, Jahangirnagar University, ismailhossainpolash@gmail.com

We measure word-level confidence of four public Bangla recognisers on 12 regional dialects (RegSpeech12) and on Standard Bangla (Common Voice 13), and study whether confidence calibrators transfer across dialects.

## What is and is not in this repository

| Included | Not included |
|---|---|
| Feature-extraction and analysis scripts, with the Kaggle kernel metadata used to run them | Audio |
| Aggregated result tables (`paper/data/*.csv`) behind every table and figure | Per-word feature files and hidden-state arrays (large; regenerate with the scripts) |
| Paper source and build scripts | Reference transcripts and hypothesis text |

RegSpeech12 is listed on Kaggle with an unknown licence. The authors have not yet given permission to redistribute it or text taken from it, so nothing derived from its transcripts is released here.

## Data and models (all obtained by the user from their original sources)

- RegSpeech12 (Kaggle `mdrezuwanhassan/regspeech12`), test split plus all Comilla and Noakhali clips.
- Common Voice 13 Bengali (Kaggle `umongsain/common-voice-13-bengali-normalized`, CC0), 2,500 test clips drawn with seed 0.
- Models: `bengaliAI/tugstugi_bengaliai-asr_whisper-medium`, `bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium`, `openai/whisper-large-v3`, `arijitx/wav2vec2-xls-r-300m-bengali` (checkpoint revisions and library versions were not pinned; Kaggle's default GPU image of October 2026 was used).

## Pipeline

Everything ran as Kaggle kernels (GPU for extraction, CPU for analysis). Kernel IDs are in each `kernel-metadata.json` and refer to the author's private kernels, so re-create them under your own account and set the `kernel_sources` accordingly.

| Step | Script | Kernel | Produces |
|---|---|---|---|
| Features, dialects (Whisper-medium Std and Reg) | `extract/bn-calib-feat-whisper-medium` | GPU, ~68 min | per-word features |
| Features, dialects (Whisper-large-v3) | `extract/bn-calib-feat-whisper-large` | GPU, ~57 min | per-word features |
| Features, dialects (XLS-R CTC) | `extract/bn-calib-feat-xlsr` | GPU | per-word features |
| Features, Common Voice (all four models) | `extract/bn-calib-feat-cv` | GPU, ~83 min | per-word features |
| Hidden states (Std, Reg; dialects and CV) | `extract/bn-calib-feat-h-dialect`, `-h-cv` | GPU | per-word features and decoder states |
| Leave-one-dialect-out calibration | `analysis/a1/analyze.py` | CPU | `results_lodo.csv` |
| Standard-to-dialect transfer | `analysis/a2/analyze2.py` | CPU | `results_std2dialect.csv` |
| Base-rate control and recalibration | `analysis/a3/analyze3.py` | CPU | `pairs.csv`, `fewshot.csv` |
| Utterance-level recalibration | `analysis/a4/analyze4.py` | CPU | `fewshot_utt.csv` |
| Hidden-state estimator | `analysis/a5/analyze5.py` | CPU | `hidden_baseline.csv` |
| WER, signed gap, Brier, bootstrap, figure | `analysis/a6/analyze6.py` | CPU | `metrics_wer.csv`, `signed_brier.csv`, `bootstrap_ci.csv`, `fig_reliability.png` |

`paper/data/clip_counts.csv` lists the clips per region.

## Rebuilding the paper tables

```bash
cd paper
python3 make_tables.py > tables.md   # every table is generated from paper/data/*.csv
python3 build.py                     # fills draft_template.md -> draft.md
python3 to_tex.py                    # draft.md -> arxiv/main.tex
cd arxiv && tectonic main.tex
```

Only the Python standard library is needed for these four steps. The analysis scripts need pandas, scikit-learn, scipy and (for `analyze6.py`) matplotlib.

## Known limitations

GBM fits are deterministic (early stopping is disabled), so all tables come from one set of fits. Bootstrap intervals resample evaluation clips with the fitted calibrators held fixed and do not include fitting uncertainty. Neither corpus has speaker IDs. Reference labels were not manually audited.

## Generative AI use

An AI assistant (Claude, Anthropic) was used for implementation support, analysis scripting, literature searching, and drafting and editing the manuscript, under the author's direction and review. See the declaration in the paper.

## Licence

Code: MIT (`LICENSE`). The aggregated results contain no audio or transcript text.
