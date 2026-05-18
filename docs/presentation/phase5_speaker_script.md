# Phase 5 Speaker Script

Working title:

> Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures

This script is synchronized to the **latest exported PDF deck** at [Flow is All You Need for WiFi.pdf](D:/Resfes_2026/Person-in-WiFi-3D/docs/presentation/Flow%20is%20All%20You%20Need%20for%20WiFi.pdf). It assumes:

- the final deck has `30` main slides and `6` appendix slides
- some slide artwork in the PDF still shows `MO`, but the spoken script always says `M0`
- Stage 3 now spans five slides:
  - Slide `18`: draft-to-refine concept
  - Slide `19`: why flow helps
  - Slide `20`: rectified-flow architecture
  - Slide `21`: proposed `M4` pipeline
  - Slide `22`: `M4` training pipeline
- the deck is presented by `3` speakers in about `10 minutes`

Core thesis to repeat:

> Flow is the main accuracy driver, and WiMamba is the component that makes the pipeline practical.

Claim boundaries to preserve while speaking:

- `M0-M4` is the only main ladder discussed in the live talk
- `M3` is the best MPJPE model in the current benchmark snapshot
- `M4` is the best deployment trade-off model in the current benchmark snapshot
- qualitative figures are supportive examples, not stronger evidence than the quantitative table
- appendix is available for deeper architecture, training, and FLOPs questions

---

## Timing Plan

| Section | Slides | Time |
|---|---|---:|
| Opening and Motivation | 1-6 | 1:45 |
| System and Baseline | 7-11 | 1:35 |
| Staged Redesign | 12-22 | 4:05 |
| Experiments and Results | 23-28 | 2:05 |
| Limitations and Close | 29-30 | 0:30 |
| Total target | 1-30 | ~10:00 |

Practical note:

> The deck has many divider slides. The talk only stays on them long enough to signal structure, then moves on quickly.

---

## Recommended 3-Speaker Split

Assumed assignment:

- Nam = `SP1`
- Nghi = `SP2`
- Thu = `SP3`

| Speaker | Slides | Responsibility |
|---|---|---|
| Nam (`SP1`) | 1-8 | opening, motivation, system-level framing |
| Nghi (`SP2`) | 9-22 | baseline, ablation ladder, and technical redesign |
| Thu (`SP3`) | 23-30 | experiments, limitations, and close |

Handoff after Slide 8:

> We have now seen the problem setting and the full system view. Nghi will now define the baseline and explain how we redesign it from `M0` to `M4`.

Handoff after Slide 22:

> With the redesign fully defined, Thu will now show what the experiments tell us.

---

## Speaker Packs

### Nam (`SP1`) - Slides 1 to 8

Mission:

> Make the judges care about the problem before asking them to absorb the model details.

What Nam must deliver clearly:

- this is a privacy-sensitive sensing problem, not a camera-bashing argument
- WiFi is interesting because it is passive, already deployed, and non-visual
- the project already has a concrete end-to-end system

Tone:

- cinematic at Slides 1-6
- cleaner and more technical at Slides 7-8

Danger to avoid:

- speaking too fast in the first minute
- making Slide 2 sound like another agenda slide, because it is now part of the motivation

### Nghi (`SP2`) - Slides 9 to 22

Mission:

> Convince the judges that the contribution is controlled, staged, and technically real.

What Nghi must deliver clearly:

- `M0` is a fair and serious baseline
- the redesign is a clean ablation ladder, not a random stack of modules
- `M1` improves representation
- `M2` improves sequence modeling and efficiency
- `M3` changes the prediction objective through flow
- `M4` integrates the redesign into one practical pipeline

Tone:

- calm, structured, and authoritative

Danger to avoid:

- over-explaining every diagram box
- losing the audience inside Slides 20-22
- saying `M4` is the most accurate model

### Thu (`SP3`) - Slides 23 to 30

Mission:

> Turn the architecture story into evidence, then land the correct final takeaway.

What Thu must deliver clearly:

- the dataset has real acquisition diversity, but still has limits
- MPJPE is the main metric and lower is better
- `M3` is the accuracy winner
- `M4` is the deployment winner
- the final message is balanced: progress, limitations, and future work

Tone:

- concise, confident, and conclusive

Danger to avoid:

- reading the table line by line without interpretation
- overstating the qualitative slide
- making the limitations slide sound apologetic instead of mature

---

## Slide-by-Slide Script

### Slide 1 - Title

Speaker:

> Nam (`SP1`)

Target time:

> 0:15

Script:

> Imagine a bedroom, a hospital room, or the home of an elderly person. In those spaces, we may want to understand human posture, but the moment we place a camera there, we also capture identity, appearance, and private life. Our work begins with one question: can we keep the understanding without keeping the camera?

Transition:

> That question leads us directly to the sensing problem we care about.

---

### Slide 2 - Toward privacy-preserving indoor human sensing

Speaker:

> Nam (`SP1`)

Target time:

> 0:22

Script:

> This slide frames the target application. We care about indoor human sensing in spaces where privacy matters, so the goal is not only accurate perception, but privacy-preserving perception. That is why WiFi is attractive: it is passive, already present in many rooms, and does not directly record visual identity.

Transition:

> With that motivation in place, here is the roadmap of the talk.

---

### Slide 3 - Content

Speaker:

> Nam (`SP1`)

Target time:

> 0:10

Script:

> We will move from the motivation, to the proposed system, to the baseline, then through our staged redesign from `M0` to `M4`, and finally to experiments, limitations, and future work.

Transition:

> We begin with the problem and motivation.

---

### Slide 4 - Problem & Motivation

Speaker:

> Nam (`SP1`)

Target time:

> 0:03

Script:

> First, the motivation.

---

### Slide 5 - Why camera-based sensing is not always acceptable

Speaker:

> Nam (`SP1`)

Target time:

> 0:22

Script:

> Cameras are powerful, but in privacy-sensitive indoor spaces, accuracy alone is not enough. If a system understands posture by continuously watching people, then in many real environments that solution will never feel fully acceptable.

Transition:

> So the natural question is whether we can preserve sensing ability without relying on vision.

---

### Slide 6 - Can WiFi understand human posture without cameras and wearables?

Speaker:

> Nam (`SP1`)

Target time:

> 0:33

Script:

> Wearables avoid cameras, but they still need to be worn, remembered, charged, and accepted by the user. So our question becomes harder and more interesting: can WiFi infer human posture without cameras and without wearables? This slide also previews the task itself: from WiFi CSI on the left, to 3D pose estimation on the right, and later to increasingly stronger models from `M0` to `M3`.

Transition:

> Once the question is clear, we can show the system view that answers it.

---

### Slide 7 - Proposed System

Speaker:

> Nam (`SP1`)

Target time:

> 0:03

Script:

> Now we move from motivation to system design.

---

### Slide 8 - Proposed System

Speaker:

> Nam (`SP1`)

Target time:

> 0:35

Script:

> At a high level, the system starts from raw WiFi CSI, preprocesses it into model-ready tensors, encodes the signal, and finally outputs 3D human poses for one-person, two-person, and three-person scenes. This slide is the global map. In the next block, we zoom in and define the baseline before explaining our redesign.

Transition:

> Nghi will now define the baseline and show how we improve it stage by stage.

---

### Slide 9 - Baseline Method

Speaker:

> Nghi (`SP2`)

Target time:

> 0:03

Script:

> Before claiming anything new, we first need to be fair to compare to the baseline.

---

### Slide 10 - M0 baseline

Speaker:

> Nghi (`SP2`)

Target time:

> 0:27

Script:

> `M0` is our starting point. It linearly projects CSI features, models them with a Transformer-based encoder-decoder pipeline, and predicts the final 3D pose directly. The important point is that `M0` is already a serious pose-estimation baseline, with SOTA Accuracy in the dataset not only a weak strawman.

Transition:

> The next slide clarifies what this baseline is actually optimized to learn.

---

### Slide 11 - What M0 optimizes

Speaker:

> Nghi (`SP2`)

Target time:

> 0:28

Script:

> Conceptually, `M0` first uses Hungarian matching to assign predicted queries to ground-truth persons, then learns person classification, direct keypoint coordinate regression, intermediate decoder supervision, and refine-stage supervision. But the main takeaway is simple: the target is still the final coordinates themselves, not a correction trajectory.

Transition:

> That direct-regression setup is exactly what we start redesigning next.

---

### Slide 12 - What We Propose: A Staged Redesign

Speaker:

> Nghi (`SP2`)

Target time:

> 0:03

Script:

> This section is the core contribution of our work.

---

### Slide 13 - A controlled M0-M4 ablation ladder

Speaker:

> Nghi (`SP2`)

Target time:

> 0:27

Script:

> We organize the family as a controlled ablation ladder. `M1` changes the input representation, `M2` changes the sequence model, `M3` changes the prediction objective through flow, and `M4` combines flow with WiMamba for the final deployment-oriented design. This structure is important because each stage isolates one hypothesis.

Transition:

> Stage 1 begins with the representation itself.

---

### Slide 14 - Stage 1: Residual motion-aware spectral tokenization

Speaker:

> Nghi (`SP2`)

Target time:

> 0:28

Script:

> In `M1`, we keep the original linear path as a stable bypass, but add a residual spectral branch. That branch smooths the temporal signal, moves into the frequency domain, estimates motion-sensitive energy, and uses gating to highlight dynamic cues while suppressing more static noise. So `M1` improves representation without changing the rest of the pipeline.

Transition:

> After improving the tokens, we redesign the encoder itself.

---

### Slide 15 - Stage 2: Replace quadratic attention with factorized WiMamba

Speaker:

> Nghi (`SP2`)

Target time:

> 0:25

Script:

> In `M2`, we replace quadratic attention with factorized WiMamba. The key idea is to respect the structure of CSI tokens by splitting sequence modeling into two passes: a temporal pass over motion along time, and a bidirectional spatial pass over antenna-group interaction. That gives us a more structured and more efficient encoder.

Transition:

> The first internal stage focuses on temporal dynamics.

---

### Slide 16 - Stage 2a: Temporal Mamba

Speaker:

> Nghi (`SP2`)

Target time:

> 0:22

Script:

> In the temporal stage, each spatial group is treated as its own short time sequence. Temporal Mamba scans along that time axis to model motion evolution before the features are reorganized for the spatial stage.

Transition:

> Then we reuse the same features to model spatial interaction.

---

### Slide 17 - Stage 2b: Bidirectional spatial Mamba

Speaker:

> Nghi (`SP2`)

Target time:

> 0:24

Script:

> In the spatial stage, each time step becomes a short sequence over the antenna groups. We run one spatial Mamba forward and one backward, then fuse them. So this is not full-sequence all-pairs modeling. It is a structured spatial scan that improves efficiency while preserving interaction across antenna groups.

Transition:

> Once the encoder is redesigned, we then change the prediction process itself.

---

### Slide 18 - Stage 3: From direct regression to draft-to-refine

Speaker:

> Nghi (`SP2`)

Target time:

> 0:24

Script:

> In Stage 3, we stop forcing the model to predict the final pose in one jump. Instead, the model first predicts a draft pose `X0`, then uses the encoded query feature as a condition to refine that draft toward the target pose `X1`. So this is not just an extra module. It is a different prediction objective.

Transition:

> The next slide explains why that change helps.

---

### Slide 19 - Why flow helps beyond direct regression

Speaker:

> Nghi (`SP2`)

Target time:

> 0:27

Script:

> In `M0`, the model must jump directly to the final pose in one shot. In `M3` and `M4`, the model starts from a draft and learns a guided correction path. In other words, flow changes the learning problem from one-shot prediction into structured correction.

Transition:

> The next slide shows the flow module itself more concretely.

---

### Slide 20 - Rectified-Flow Architecture

Speaker:

> Nghi (`SP2`)

Target time:

> 0:30

Script:

> Here we zoom into the rectified-flow refiner. The inputs are the draft pose, the query-conditioned feature, and a time encoding. A lightweight VelocityMLP predicts a correction velocity over the full 14-joint 3D pose, and one residual update produces the refined pose. The key message is that refinement is lightweight, conditional, and structure-aware.

Transition:

> With that refinement module defined, we can place it back into the full model.

---

### Slide 21 - The proposed M4 pipeline

Speaker:

> Nghi (`SP2`)

Target time:

> 0:30

Script:

> This is the full `M4` pipeline. It starts with spectral tokenization, processes the sequence with repeated factorized WiMamba blocks, generates pose queries through a lightweight decoder, selects top queries, and then applies one-step rectified-flow refinement. So `M4` is the integrated form of all earlier redesign stages.

Transition:

> The final slide in this block shows how `M4` is trained.

---

### Slide 22 - M4 Training Pipeline

Speaker:

> Nghi (`SP2`)

Target time:

> 0:33

Script:

> This slide adds the training view. The top tier shows the forward path from CSI to draft and refined poses. The bottom tier shows Hungarian matching, classification loss, draft pose loss, and rectified-flow training with a velocity objective. So the redesign changes not only the architecture, but also how prediction, matching, and refinement are learned together.

Transition:

> With the redesign fully defined, Thu will now show what the experiments tell us.

---

### Slide 23 - Experiments & Result

Speaker:

> Thu (`SP3`)

Target time:

> 0:03

Script:

> We now move to the experiments and results.

---

### Slide 24 - Dataset Overview

Speaker:

> Thu (`SP3`)

Target time:

> 0:25

Script:

> Our benchmark is grounded in a real acquisition setup. It includes seven volunteers, three indoor locations, and eight daily actions, evaluated across one-person, two-person, and three-person scenes. This matters because ambiguity grows quickly as the scene becomes more crowded, so the benchmark is designed to reflect increasing difficulty.

Transition:

> Before comparing models, we briefly define the evaluation metric.

---

### Slide 25 - Evaluation Metric: MPJPE

Speaker:

> Thu (`SP3`)

Target time:

> 0:18

Script:

> We use MPJPE, or Mean Per Joint Position Error. It measures the average 3D Euclidean distance between each predicted joint and its ground-truth joint, then averages over all joints and all samples. So lower MPJPE means better 3D pose accuracy.

Transition:

> With the metric defined, we can now read the benchmark results directly.

---

### Slide 26 - Flow improves accuracy. Mamba improves deployability.

Speaker:

> Thu (`SP3`)

Target time:

> 0:42

Script:

> This is the main quantitative result. `M3` achieves the best MPJPE at `151.99` millimeters, which tells us that flow is the strongest driver of accuracy. `M4` is slightly higher in MPJPE at `159.00`, but it is the practical deployment winner with `159.85` FPS, only `5.83` million parameters, and `27.02` megabytes of memory. So the correct reading is very important: `M3` is the accuracy winner, and `M4` is the deployment winner.

Transition:

> The next slide shows how this difference appears in difficult qualitative cases.

---

### Slide 27 - More coherent matched poses in challenging scenes

Speaker:

> Thu (`SP3`)

Target time:

> 0:26

Script:

> In these challenging one-person, two-person, and three-person examples, the flow-based variants produce more coherent matched poses than the direct-regression baseline. We use this slide as supportive visual evidence, especially for structural collapse under severe multipath ambiguity, but the quantitative table remains the primary result.

Transition:

> We can now summarize the overall gains from `M0` to `M4`.

---

### Slide 28 - In summary

Speaker:

> Thu (`SP3`)

Target time:

> 0:25

Script:

> From `M0` to `M4`, inference speed rises from `45.69` to `159.85` FPS, parameters drop from `13.13` million to `5.83` million, peak memory falls from `155.60` to `27.02` megabytes, and MPJPE still improves from `169.34` to `159.00` millimeters. In short, the redesign makes the model both lighter and better, while keeping the main accuracy boost tied to flow.

Transition:

> Before closing, we also want to be transparent about the boundaries of the current benchmark.

---

### Slide 29 - Limitations & Future Work

Speaker:

> Thu (`SP3`)

Target time:

> 0:25

Script:

> We close with a transparent view of both limits and next steps. The current benchmark still reflects specific indoor local snapshots, and harder multi-person scenes continue to challenge multipath resolution and signal separation. Looking forward, we want to test more encoder variants such as `Mamba-2` and `Mamba-3`, run more ablation and deployment-oriented testing, explore quantization and router-grade integration, and develop stronger physics-informed correction for severe occlusion cases.

Transition:

> Thank you for listening.

---

### Slide 30 - Thanks For Your Listening

Speaker:

> Thu (`SP3`)

Target time:

> 0:05

Script:

> Thank you. We are happy to take your questions.

---

## Appendix Usage Notes

### Slide 31 - Appendix divider

Use:

- only when entering Q&A backup mode

### Slide 33 - Full architecture / training / loss detail

Use when:

- judges want one dense screen that connects acquisition, architecture, loss, and refinement

Suggested answer:

> This appendix slide compresses the whole pipeline into one diagram, from WiFi acquisition through spectral tokenization, WiMamba encoding, draft prediction, flow refinement, and training loss.

### Slide 35 - VelocityMLP

Use when:

- the question is specifically about the flow refiner internals

Suggested answer:

> This backup slide isolates the VelocityMLP, which is the small network that predicts the correction velocity used by rectified flow.

### Slide 36 - A simple FLOPs intuition

Use when:

- judges ask why WiMamba is more efficient than attention

Suggested answer:

> The intuition is that WiMamba scans the sequence linearly instead of building a full `L by L` attention matrix, which is why it scales more gently for sequence modeling.

---

## Fast Q&A Anchors

### Q1. What is the main takeaway of the ablation?

> The ladder isolates three effects: better representation, better structured sequence modeling, and a better prediction objective. Flow contributes the strongest accuracy gain, while WiMamba contributes the strongest efficiency gain.

### Q2. Why is `M3` more accurate than `M4`?

> `M3` keeps the stronger flow-oriented accuracy setting and achieves the best MPJPE. `M4` gives up a small amount of accuracy to gain much better runtime efficiency and memory behavior.

### Q3. Why choose `M4` if `M3` has lower MPJPE?

> Because `M4` is the better deployment trade-off. It is still clearly better than `M0`, while being much faster, smaller, and lighter.

### Q4. What exactly is the WiMamba contribution?

> Our WiMamba contribution is not only replacing Transformer with Mamba. We factorize the encoder into a Temporal Mamba stage and a Bidirectional Spatial Mamba stage, so the CSI structure is modeled explicitly along time and across antenna groups.

### Q5. What is the real difference between direct regression and flow?

> Direct regression predicts the final pose in one shot. Flow predicts how a draft pose should be corrected toward the target.

### Q6. How should we read the qualitative slide?

> As supportive examples. It visualizes the same trend as the benchmark, but it does not replace quantitative evaluation.

---

## Delivery Checklist

- Move quickly through Slides 3, 4, 7, 9, 12, 23, and 30 because they are structural slides.
- Spend the most attention on Slides 10, 13-22, 26, 28, and 29.
- Say `M3 = accuracy winner` and `M4 = deployment winner` exactly and consistently.
- Do not say `M4` is the best model in every sense.
- Use appendix only for Q&A, not as part of the default live run.
