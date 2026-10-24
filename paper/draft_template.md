# Calibrating Word-Level Confidence in Bangla Speech Recognition Across Regional Dialects

Ismail Hossain Polas; Jahangirnagar University; ismailhossainpolash@gmail.com

## Abstract

A transcription pipeline can send its least certain words to a human reviewer, but only if the model's word confidence means what it says. Bangla dialect benchmarks report error rates and nothing about confidence. We measure word-level confidence for four public Bangla recognisers (two Whisper-medium checkpoints, Whisper-large-v3 and an XLS-R CTC model) on 2,668 clips from 12 Bangladeshi regional dialects, and on 2,500 Standard Bangla clips from Common Voice. Raw confidence is overconfident on every model: mean confidence exceeds accuracy by 0.42 to 0.71, and per-region expected calibration error (ECE) is 0.42 to 0.71. A gradient-boosted calibrator over 12 decoding features, trained on the other dialects, brings ECE to between 0.022 and 0.052. A calibrator trained on Standard Bangla ranks dialect errors almost as well (AUROC down 0.017 to 0.020 on three of four models and 0.081 on the fourth) but is miscalibrated on the two Whisper-medium models (ECE 0.33 and 0.21, against 0.052 and 0.049 for a calibrator trained on other dialects). That failure grows with the gap in error rate between training and target speech, and about 100 labelled target words bring the three Whisper models to an ECE between 0.050 and 0.068, whereas estimating the error rate without labels does not help. Even after calibration, at most 0.1% of dialect words reach a confidence of 0.9, so confidence-based review has limited reach on these models.

## 1. Introduction

Word-level confidence is what lets a speech pipeline decide which words a person should check [Oneata2021; Qiu2021]. A call-centre transcription service in Bangladesh would use it that way, and it would meet speech that differs from the Standard Bangla in most training data.

Recent Bangla work documents that difference but not its consequence for confidence. Ben-10 benchmarks Whisper, wav2vec 2.0, a Conformer and Google ASR on ten regional dialects and finds that all of them struggle, zero-shot and fine-tuned [Dipto2025]. RegSpeech12 adds spontaneous speech from twelve regions [Hassan2025], and BanglaTalk evaluates a dialect-aware recogniser on it [Hasan2025]. These papers report word and character error rates. Work on confidence, meanwhile, has measured calibration under additive noise [Huo2025], out-of-domain English text domains [Li2022] and a single English accent [Jia2026], and has related Whisper's confidence to a language's training data [Liang2025]. In the literature we searched (arXiv and the ACL and ISCA archives, through October 2026), we found no study of word-level confidence on dialect speech in any language.

We ask two questions. First, how miscalibrated is the confidence of public Bangla recognisers on regional dialects? Second, does a calibrator fitted on one kind of Bangla speech, in particular Standard Bangla, remain usable on another? We answer both on 12 Bangladeshi dialects with four models, using a 12-feature gradient-boosted calibrator, Platt scaling and the raw model probabilities.

Our findings are these:

1. Raw word confidence is badly overconfident on all four models (mean confidence exceeds accuracy by 0.42 to 0.71; per-region ECE 0.42 to 0.71), and a calibrator trained on the other dialects reduces it to 0.022 to 0.052 on a held-out dialect.
2. A calibrator trained on Standard Bangla keeps most of its ranking ability on dialects but loses calibration on the Whisper-medium models; the loss scales with the error-rate gap between source and target.
3. About 100 labelled target words recalibrate the Whisper models to ECE below 0.07, whereas label-free estimation of the target error rate does not help.
4. Calibrated confidence rarely exceeds 0.9 on dialect speech, which bounds how many words any confidence threshold can accept.

## 2. Related work

<!--TABLE:related-->

Confidence for end-to-end recognisers is usually estimated by a small module over decoder features. Qiu et al. learn word-level confidence for word-piece models [Qiu2021], and Oneata et al. evaluate several estimators on end-to-end systems [Oneata2021]. Laptev and Ginsburg propose entropy-based scores that need no training [Laptev2022]. Naowarat et al. extend the idea to CTC models and add neighbouring words as context [Naowarat2023]. None of these papers evaluates on dialect or accent speech.

Several studies address distribution shift. Li et al. show that a confidence module trained in-domain degrades out of domain and repair it with pseudo-transcriptions and an extra language model [Li2022]. Aggarwal et al. fine-tune Whisper to emit word confidence and report gains on eight out-of-domain datasets over a feature-based module [Aggarwal2025]. Huo et al. find that Whisper's overconfident errors rise at low signal-to-noise ratio and correct them with temperature scaling on selected tokens [Huo2025]. Jia and Van hamme score tokens from beam-search ranks and test on US-accented Common Voice [Jia2026]. Liang et al. probe Whisper's sub-token probabilities across languages [Liang2025], and Ballier et al. relate calibration to training data in three languages [Ballier2024]. The shift in each case is noise, text domain, language or a single accent. Dialect speech, where both the acoustics and the vocabulary change, is not among them.

For Bangla, the corpora and benchmarks above supply error rates only [Dipto2025; Hassan2025; Hasan2025]. Among the Bangla speech papers we reviewed, none mentions confidence or calibration.

## 3. Data and models

**Speech.** The RegSpeech12 test split has 2,132 clips of spontaneous speech labelled with one of 12 regions [Hassan2025]. Its transcripts are written in dialectal spelling, for example "taha" where Standard Bangla writes "taka" (romanised). Comilla and Noakhali do not appear in Ben-10, so the regional checkpoint cannot have trained on them, and the test split holds only 32 and 28 of their clips. We therefore add all their training and validation clips (286 and 250), which gives 2,668 clips (Table 0). Their transcripts show no overlap with Ben-10 in any split (Section below), so this use of training and validation clips adds no leakage for the regional checkpoint. No clip is used both to fit and to evaluate a calibrator: leave-one-dialect-out holds out whole regions, Standard-to-dialect training uses Common Voice only, and recalibration splits by utterance.

<!--TABLE:0-->

For Standard Bangla we draw 2,500 clips at random (seed 0) from the 9,230-clip test split of Common Voice 13 Bengali, which is read speech in short clips (mean 6.4 s, against 16.9 s for the dialect clips).

**Overlap with Ben-10.** The regional Whisper checkpoint comes from the Ben-10 project, so we compared transcripts. In nine regions, 84.7% to 98.8% of RegSpeech12 training clips and 92.7% to 100% of validation clips share a six-word sequence with Ben-10 text. The test split is clean apart from Rangpur, where 93.8% of clips overlap (other regions: 0% to 4.0%). Apart from Comilla and Noakhali, we therefore use only the test split for dialects, and we drop Rangpur for the regional checkpoint (so that checkpoint is evaluated on 11 regions). Neither corpus gives speaker identity, so speaker overlap is unknown.

**Models.** We use two Whisper-medium checkpoints released by BengaliAI, one for Standard Bangla (Std) and one for regional speech (Reg), Whisper-large-v3 used zero-shot with the language set to Bangla, and a public XLS-R-300M CTC checkpoint fine-tuned for Bangla. Decoding is greedy with clips cut at 30 seconds and at most 200 new tokens.

## 4. Method

**Word confidence features.** For the Whisper models we run the generated sequence back through the decoder with the standard prompt and take the probability of each emitted token. Tokens are grouped into words at the leading-space marker. Each word gets 12 features. Five are probability statistics: the mean, minimum, first and last token probability, and the mean log-probability. Three describe the full-vocabulary distribution: mean entropy, maximum entropy and the mean margin between the top two candidates. Four are positional: the token count, the relative position in the utterance, the length of the hypothesis in words and the character length. For the CTC model we use the same statistics over the frame-level maximum probabilities of the emitted, non-blank frames, grouped by the word delimiter.

**Error labels.** We remove angle-bracket tags and punctuation from references and hypotheses, align them at word level by edit distance, and call a hypothesis word an error if it is substituted or inserted. Deleted reference words are not scored, so the error rates we report for calibration are shares of hypothesis words, not word error rates; Table 1 gives conventional WER (substitutions, deletions and insertions over reference words) alongside. The strict label counts any mismatch. The lenient label counts a substitution as correct when its character edit distance is at most 0.34 of the longer word, which forgives many spelling variants.

**Calibrators.** We compare the raw mean token probability and raw minimum token probability, Platt scaling on the logit of the mean probability, and a gradient-boosted classifier (200 iterations, learning rate 0.08, 15 leaves) over the 12 features. Each outputs the probability that the word is correct.

**Protocols.** In leave-one-dialect-out evaluation the calibrator is trained on the words of the other 11 dialects and scored on the held-out one. An in-distribution control trains on a random 80% of utterances from all dialects and scores the rest. For Standard-to-dialect transfer the calibrator is trained only on the Common Voice words. For recalibration we draw whole utterances from the target dialect until they supply n labelled words, fit Platt scaling to the source calibrator's output on them, and score the remaining utterances, repeating 20 times per dialect. We also test two error-rate adjustments: one shifts the log-odds by the difference between the target's and source's error rates, using the true target rate (an upper bound), and one estimates that rate without labels by expectation-maximisation.

**Metrics.** ECE uses 15 equal-width bins. We report it per region and average over regions weighted by word count, because pooling across regions lets opposite errors cancel (pooled ECE is 0.003 to 0.006 for the dialect-trained GBM in both leave-one-dialect-out and the control, which hides regional error). AUROC measures how well confidence separates errors from correct words. Pooled bootstrap intervals resample utterances 100 times.

## 5. Results

### 5.1 Raw confidence is overconfident

<!--TABLE:1-->

Table 1 shows that every model claims far more accuracy than it has. For the Standard Whisper-medium checkpoint, 68% of dialect hypothesis words are wrong (WER 0.80), yet its mean word confidence is 0.91, a signed gap of +0.59 above its accuracy of 0.32. The gap is positive on all four models (+0.42 to +0.71) and in every region (Table 9, Figure 1), so the ECE reflects overconfidence and not merely mismatch. Minimum token probability gives a lower ECE on every model, down to 0.21 on the regional checkpoint, but remains overconfident (gaps +0.21 to +0.48).

![Reliability curves on dialect words: observed accuracy against predicted probability of being correct, for the raw mean token probability, a GBM trained on other dialects and a GBM trained on Standard Bangla (Common Voice). Points under the diagonal are overconfident. Bins with fewer than 30 words are omitted.](fig_reliability.png)

<!--TABLE:9--> AUROC of the raw scores is 0.62 to 0.80, with Whisper-large-v3 weakest. Under the lenient label the pooled ECE of the raw mean probability falls by 0.05 to 0.12 and stays between 0.30 and 0.65.

### 5.2 A calibrator trained on other dialects transfers

<!--TABLE:2-->

Table 2 shows that the 12-feature GBM trained on the other 11 dialects reaches a per-region ECE of 0.022 to 0.052 and an AUROC of 0.767 to 0.851. Platt scaling alone fixes most of the calibration (0.048 to 0.071) but leaves AUROC at the raw level, so the extra ranking ability comes from the other features. The pooled leave-one-dialect-out result is within 0.006 AUROC of the in-distribution control for every model, and the clip-level bootstrap intervals in Table 8 are narrow (for the Standard checkpoint, ECE 0.052 [0.051, 0.058] and AUROC 0.781 [0.777, 0.786]). The calibrated estimators are close to unbiased on average (pooled signed gap within 0.003 of zero on all four models, Table 9), although the gap ranges from -0.14 to +0.07 across regions, and their Brier scores are 0.09 to 0.18 against 0.25 to 0.62 for the raw probabilities. Percentile intervals are one-sided around ECE because ECE is biased upward under resampling. Calibration therefore transfers between these 12 dialects. Table 6 checks whether this depends on the hand-built features.

### 5.3 Standard Bangla to dialects: ranking holds, calibration does not

<!--TABLE:3-->

Table 3 compares calibrators trained on Standard Bangla with the dialect-trained ones. On the two Whisper-medium models the Standard-trained GBM has per-region ECE of 0.334 and 0.210, against 0.052 and 0.049 when trained on other dialects (paired differences 0.282 and 0.161, Table 8), and it stays overconfident (signed gap +0.33 and +0.21, Table 9). Platt scaling is no better (Table 3). Ranking degrades much less: AUROC falls by 0.020 (Std), 0.019 (Reg) and 0.017 (XLS-R), against 0.081 for Whisper-large-v3 (Table 8, intervals within 0.015 of each point). Two GBM fits on the same data in different runs differ slightly, because early stopping uses an internal random split; this is why Table 3 shows ECE of 0.336 and 0.229 for the two Whisper-medium models and Table 8 shows 0.334 and 0.210, and why the large-v3 AUROC drop is 0.072 in Table 3 and 0.081 in Table 8. We quote Table 8 in the text. The reverse direction fares no better (Table 3b).

<!--TABLE:8-->

<!--TABLE:3b-->

XLS-R is the exception to the ECE pattern: its Standard-trained GBM reaches 0.032 with no error-rate correction.

The error rate may account for much of this. Common Voice error shares are 0.09 and 0.13 for the two Whisper-medium models against 0.68 and 0.62 on dialects, whereas Whisper-large-v3 (0.77 to 0.87) and XLS-R (0.44 to 0.83) have smaller gaps.

### 5.4 Transfer error grows with the error-rate gap

<!--TABLE:4-->

Across all source and target pairs, transfer ECE rises with the accuracy gap between them. The Spearman correlations are 0.72, 0.85 and 0.68 for the Std, Reg and large-v3 models, and 0.01 for XLS-R; bootstrap intervals over pairs are [0.61, 0.80], [0.79, 0.90], [0.57, 0.75] and [-0.18, 0.17], and are optimistic because pairs share clips. This is an association, and we have not tested that the gap causes the loss. At matched gaps the source type does not matter: for gaps up to 0.2, Common Voice sources give ECE of 0.062 and 0.085 and dialect sources give 0.060 and 0.088. Above a gap of 0.6 only Common Voice pairs exist (ECE 0.308, seven pairs), so we cannot separate dialect from read speech there. Correcting only the error rate does not remove the effect. With the true target rate, ECE for Std and Reg is still 0.117 and 0.132, and for XLS-R it rises from 0.032 to 0.112 (Table 3).

### 5.5 A little target labelling fixes most of it

<!--TABLE:5-->

Table 5 shows that Platt scaling on 100 labelled words lowers the Standard-trained calibrator's per-region ECE from 0.320 to 0.061 on Std and from 0.213 to 0.068 on Reg, and 400 words reach 0.036 and 0.040. The spread across region-by-repetition runs is large at 100 words (SD 0.033 and 0.035), so the average hides regions where 100 words are not enough. The label-free expectation-maximisation estimate of the target error rate does not help: it moves ECE from 0.213 to 0.336 on Reg and from 0.034 to 0.153 on XLS-R. For XLS-R, Platt scaling on 100 words (0.043) is also worse than leaving the calibrator alone (0.034), so labelled recalibration helps the Whisper models only.

### 5.6 Hidden-state estimator

<!--TABLE:6-->

Table 6 adds the decoder's final-layer hidden state at each word's last token, reduced to 64 principal components fitted on the training words, to the 12 features. This is not the method of Aggarwal et al., who fine-tune the decoder to emit confidence [Aggarwal2025]; it is a cheaper estimator in the same spirit. On the two Whisper-medium models the hidden state raises leave-one-dialect-out AUROC by 0.038 (Std) and 0.032 (Reg) and leaves ECE at 0.049 and 0.042. It does not help transfer from Standard Bangla: AUROC changes by -0.005 and -0.003, and ECE stays high at 0.338 and 0.197. The Standard-to-dialect calibration failure therefore does not depend on the hand-built features. The 12-feature rows reproduce Tables 2 and 3 exactly, since decoding is deterministic.

### 5.7 What review can buy

Confidence is useful only if enough words can be trusted or checked. With a leave-one-dialect-out GBM, reviewing the 20% least confident words lowers the hypothesis-word error share of the Standard checkpoint from 0.680 to only 0.611, against 0.620 for the raw minimum-token probability and 0.680 for random review (Table 7). The GBM beats random review on every model, but its margin over the raw minimum-token probability is 0.003 to 0.014.

<!--TABLE:7--> Accepting words whose calibrated probability of being correct is at least 0.5 covers 24.4% of its dialect words at 63.3% precision (base accuracy 32.0%), and at least 0.7 covers 7.3% at 74.5%. At 0.9, coverage is 0.0% for the Standard checkpoint and 0.1% for the regional one (91.0% precision). For Whisper-large-v3, 0.2% of words reach 0.7 and none reach 0.9; for XLS-R, none reach 0.9.

## 6. Discussion

Our results separate two properties that are usually bundled. The ability to say which of two words is more likely wrong survives a change of speech type, with AUROC losses of 0.017 to 0.020 on three models. The ability to say how likely a word is to be wrong does not, and the loss appears to follow the error rate of the speech the calibrator saw. This fits the matched-gap result in Table 4, where the source's type matters less than how far its accuracy sits from the target's.

The XLS-R model behaves differently. Its transfer ECE is already low without correction, and the error-rate adjustment makes it worse. We do not know why. One possibility is that CTC posteriors, which are sharply peaked, carry error information less tied to the base rate, but we have not tested it.

The finding most likely to be overturned is the Standard-to-dialect gap. Common Voice is short read speech and the Whisper checkpoints may have seen it in training, so some of the 0.336 may reflect exposure rather than dialect. A transcribed spontaneous Standard Bangla set would settle it; we found none available.

The practical message is modest. On these models a deployer should expect raw confidence to be unusable, a dialect-trained calibrator to transfer between dialects, and a Standard-Bangla calibrator to need on the order of a hundred labelled words from the target speech. Beyond that, the low accuracy on dialects caps what any threshold can accept.

## 7. Limitations

The strongest objection concerns the labels. They depend on a reference written in dialectal spelling, so a correct word in standard spelling counts as an error. The lenient label recovers part of this and the conclusions hold under it, but it cannot separate a spelling variant from a different word, and we did not annotate by hand. Because of this, we cannot claim the absolute error shares are right, only that the comparisons between calibrators are stable under both labels.

The Standard Bangla source is read speech from one corpus, possibly seen in training, which confounds speech type, speaking style and exposure (Section 6). We did not detect a source-type effect at matched gaps, but we have no dialect pair with a gap above 0.6, so we cannot rule one out there.

The regional checkpoint's training data is not documented in the repository we used, and RegSpeech12's training and validation text overlaps Ben-10, so we evaluate on the test split and exclude Rangpur for that checkpoint. Speaker overlap between recalibration and evaluation utterances is unknown, which may make the recalibration results optimistic. Our word-level error share ignores deleted reference words, which conventional WER counts. The GBM fits vary slightly between runs (ECE by up to 0.02 on one model, Section 5.3), so differences smaller than that should not be read as findings. Results cover twelve regions of one corpus and four models, and we trained no confidence head on a fine-tuned Whisper, as Aggarwal et al. did; Section 5.6 uses a hidden-state estimator instead, on the two Whisper-medium models only.

## 8. Conclusion

On four Bangla recognisers and twelve regional dialects, raw word confidence is overconfident, a dialect-trained calibrator fixes it for unseen dialects, and a Standard-Bangla calibrator keeps its ranking but loses its calibration on the two Whisper-medium models. The loss follows the gap in error rate and about 100 labelled target words mostly repair it. Calibrated confidence reaches 0.9 for at most 0.1% of dialect words, so on these models review by confidence helps less than the calibration numbers suggest.

## Availability

Common Voice 13 Bengali is released under CC0. RegSpeech12 is listed on Kaggle with an unknown licence, and we have not yet obtained permission from its authors to redistribute it or text taken from it. We therefore release the analysis code, the Kaggle kernel identifiers used for every step, and the aggregated result tables behind every table in this paper, but no audio, per-word feature files, reference transcripts or hypothesis text. Anyone can rebuild the per-word features by running the released scripts on the public corpora. The release is at https://github.com/ihpolash/Calibrating-Word-Level-Confidence-Bangla-ASR (no DOI yet).

## Declaration on the use of generative AI

The author used an AI assistant (Claude, Anthropic) for implementation support, analysis scripting, literature searching, and for drafting and editing this manuscript, working under the author's direction and review. The research question, the choice of corpora and models, and the interpretation of the results are the author's. All numerical results were produced by the released code, and the tables in this paper are generated from the stored result files, not typed by hand. The author takes full responsibility for all content of this paper, irrespective of how it was drafted.

## References

[Aggarwal2025] V. Aggarwal, S. S. Nair, Y. Verma, Y. Jogi. Adopting Whisper for Confidence Estimation. arXiv:2502.13446, 2025.
[Ballier2024] N. Ballier, L. Burin, B. Namdarzadeh, S. Ng, R. Wright, J.-B. Yunès. Probing Whisper Predictions for French, English and Persian Transcriptions. ICNLSP, 2024. (ACL Anthology 2024.icnlsp-1.15)
[Dipto2025] T. T. Dipto et al. Are ASR foundation models generalized enough to capture features of regional dialects for low-resource languages? IJCNLP-AACL 2025 (short), arXiv:2510.23252.
[Hasan2025] J. Hasan, S. R. Dipta. BanglaTalk: Towards Real-Time Speech Assistance for Bengali Regional Dialects. arXiv:2510.06188, 2025.
[Hassan2025] M. R. Hassan et al. RegSpeech12: A Regional Corpus of Bengali Spontaneous Speech Across Dialects. arXiv:2510.24096, 2025.
[Huo2025] M. Huo, Y. Zhang, Y. Tang. Identifying and Calibrating Overconfidence in Noisy Speech Recognition. arXiv:2509.07195, 2025.
[Jia2026] Y. Jia, H. Van hamme. Leveraging Beam Search Information for Confidence Estimation in E2E ASR. arXiv:2607.29299, 2026.
[Laptev2022] A. Laptev, B. Ginsburg. Fast Entropy-Based Methods of Word-Level Confidence Estimation for End-To-End Automatic Speech Recognition. arXiv:2212.08703, 2022.
[Li2022] Q. Li, Y. Zhang, D. Qiu, Y. He, L. Cao, P. C. Woodland. Improving Confidence Estimation on Out-of-Domain Data for End-to-End Speech Recognition. ICASSP 2022, arXiv:2110.03327.
[Liang2025] S. Liang, N. Ballier, G.-A. Levow, R. Wright. Beyond WER: Probing Whisper's Sub-token Decoder Across Diverse Language Resource Levels. arXiv:2509.25516, 2025.
[Naowarat2023] B. Naowarat, T. Kongthaworn, E. Chuangsuwanich. Word-level Confidence Estimation for CTC Models. Interspeech 2023.
[Oneata2021] D. Oneață, A. Caranica, A. Stan, H. Cucu. An evaluation of word-level confidence estimation for end-to-end automatic speech recognition. arXiv:2101.05525, 2021.
[Qiu2021] D. Qiu et al. Learning Word-Level Confidence For Subword End-to-End ASR. arXiv:2103.06716, 2021.
