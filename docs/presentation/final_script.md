---
# Phase 5 Speaker Script
## Speaker Packs
---

### Slide 9 - Baseline Method



Target time:

> 0:03

Script:

> Before claiming anything new, we first need to be fair to compare to the baseline.

---

### Slide 10 - M0 baseline



Target time:

> 0:27

Script:

> `M0` is our starting point. It linearly projects CSI features, models them with a Transformer-based encoder-decoder pipeline, and predicts the final 3D pose directly. The important point is that `M0` is already a serious pose-estimation baseline, with SOTA Accuracy in the dataset not only a weak strawman.

Transition:

>  To clarifies what this baseline is actually optimized to learn, let see. 

---

### Slide 11 - What M0 optimizes



Target time:

> 0:28

Script:

>`M0` first uses Hungarian matching to assign predicted queries to ground-truth persons, then learns person classification, direct keypoint coordinate regression, intermediate decoder supervision, and refine-stage supervision. But the main takeaway is simple: the target is still the final coordinates themselves, not a correction trajectory.


---

### Slide 12 - What We Propose: A Staged Redesign



Target time:

> 0:03

Script:

> This section is the core contribution of our work.

---

### Slide 13 - A controlled M0-M4 ablation ladder



Target time:

> 0:27

Script:

> We organize the family as a controlled ablation ladder. `M1` changes the input representation, `M2` changes the sequence model, `M3` changes the prediction objective through flow, and `M4` combines flow with WiMamba for the final deployment-oriented design. This structure is important because each stage isolates one hypothesis.


---

### Slide 14 - Stage 1: Residual motion-aware spectral tokenization



Target time:

> 0:28

Script:

> In `M1`, we keep the original linear path as a stable bypass, but add a residual spectral branch. That branch smooths the temporal signal, moves into the frequency domain, estimates motion-sensitive energy, and uses gating to highlight dynamic cues while suppressing more static noise. So `M1` improves representation without changing the rest of the pipeline.


---

### Slide 15 - Stage 2: Replace quadratic attention with factorized WiMamba



Target time:

> 0:25

Script:

> In `M2`, we replace quadratic attention with factorized WiMamba. The key idea is to respect the structure of CSI tokens by splitting sequence modeling into two passes: a temporal pass over motion along time, and a bidirectional spatial pass over antenna-group interaction. That gives us a more structured and more efficient encoder.

---

### Slide 16 - Stage 2a: Temporal Mamba



Target time:

> 0:22

Script:

> In the temporal stage, each spatial group is treated as its own short time sequence. Temporal Mamba scans along that time axis to model motion evolution before the features are reorganized for the spatial stage.


---

### Slide 17 - Stage 2b: Bidirectional spatial Mamba



Target time:

> 0:24

Script:

> In the spatial stage, each time step becomes a short sequence over the antenna groups. We run one spatial Mamba forward and one backward, then fuse them. So this is not full-sequence all-pairs modeling. It is a structured spatial scan that improves efficiency while preserving interaction across antenna groups.


---

### Slide 18 - Stage 3: From direct regression to draft-to-refine



Target time:

> 0:24

Script:

> In Stage 3, we stop forcing the model to predict the final pose in one jump. Instead, the model first predicts a draft pose `X0`, then uses the encoded query feature as a condition to refine that draft toward the target pose `X1`. So this is not just an extra module. It is a different prediction objective.

Transition:

> To explain Why flow helps beyond direct regression

---

### Slide 19 - Why flow helps beyond direct regression



Target time:

> 0:27

Script:

> In `M0`, the model must jump directly to the final pose in one shot. In `M3` and `M4`, the model starts from a draft and learns a guided correction path. In other words, flow changes the learning problem from one-shot prediction into structured correction.

---

### Slide 20 - Rectified-Flow Architecture



Target time:

> 0:30

Script:

> Here we zoom into the rectified-flow refiner. The inputs are the draft pose, the query-conditioned feature, and a time encoding. A lightweight VelocityMLP predicts a correction velocity over the full 14-joint 3D pose, and one residual update produces the refined pose. The key message is that refinement is lightweight, conditional, and structure-aware.


---

### Slide 21 - The proposed M4 pipeline



Target time:

> 0:30

Script:

> This is the full `M4` pipeline. It starts with spectral tokenization, processes the sequence with repeated factorized WiMamba blocks, generates pose queries through a lightweight decoder, selects top queries, and then applies one-step rectified-flow refinement. So `M4` is the integrated form of all earlier redesign stages.


---

### Slide 22 - M4 Training Pipeline



Target time:

> 0:33

Script:

> To clarified how M4 is trained. The top tier shows the forward path from CSI to draft and refined poses. The bottom tier shows Hungarian matching, classification loss, draft pose loss, and rectified-flow training with a velocity objective. So the redesign changes not only the architecture, but also how prediction, matching, and refinement are learned together.

Transition:

> With the redesign fully defined, Thu will now show what the experiments tell us.

---

### Slide 23 - Experiments & Result

Speaker:

> Thu (`SP3`)

Target time:

> 0:03

Script:s

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
