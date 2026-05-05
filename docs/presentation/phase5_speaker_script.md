# Phase 5 Speaker Script

Working title:

> Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures

This script is synchronized to the **final exported PDF deck**, not to the older 13-slide technical restructuring plan. It assumes:

- the final deck has `29` main slides and `5` appendix slides
- some slide artwork in the PDF still shows `MO`, but the spoken script uses `M0`
- `Why flow helps beyond direct regression` is Slide `21`, before the experiments block
- `M2` is expanded into three slides: overview, Temporal Mamba, and Bi-Mamba

Core thesis to repeat:

> Flow is the main accuracy driver, and WiMamba is the component that makes the pipeline practical.

Claim boundaries to preserve while speaking:

- `M0-M4` is the only main ladder discussed
- `M3` is the best MPJPE model in the current benchmark snapshot
- `M4` is the best deployment trade-off model in the current benchmark snapshot
- qualitative figures are supportive examples, not absolute proof
- appendix is available for deeper architecture and training questions

---

## Timing Plan

| Section | Slides | Time |
|---|---|---:|
| Opening | 1-2 | 0:20 |
| Problem & Motivation | 3-6 | 1:05 |
| Proposed System | 7-8 | 0:38 |
| Baseline Method | 9-11 | 1:03 |
| Staged Redesign | 12-21 | 3:45 |
| Experiments, Summary & Limits | 22-28 | 2:55 |
| Thanks | 29 | 0:05 |
| Total target | 1-29 | ~9:50 to 10:15 |

Practical note:

> The deck has many divider slides, so the live pace depends on moving through those quickly.

---

## Speaker Split Option

If using 2 speakers:

| Speaker | Slides | Responsibility |
|---|---|---|
| Speaker 1 | 1-11 | motivation, system setup, baseline |
| Speaker 2 | 12-29 | redesign, results, summary |

If using 3 speakers:

| Speaker | Slides | Responsibility |
|---|---|---|
| Nam (`SP1`) | 1-8 | opening, motivation, proposed system |
| Nghi (`SP2`) | 9-21 | baseline and technical redesign |
| Thư (`SP3`) | 22-29 | experiments, summary, close |

Handoff line for 2-speaker mode:

> We have now defined the baseline clearly, so I will hand over to my teammate to explain the staged redesign and the final results.

Handoff line for 3-speaker mode after Slide 8:

> We have seen the global system view. My teammate will now define the baseline and walk through the redesign stage by stage.

Handoff line for 3-speaker mode after Slide 21:

> With the redesign fully defined, my teammate will now show what the experiments tell us.

Speaker 1 style note:

> Speaker 1 should sound more cinematic than technical in the opening block. Start with concrete human spaces, then pivot into the research question, then only after that move into system and baseline detail.

---

## Recommended Named Split

Assumed assignment:

- Nam = `SP1`
- Nghi = `SP2`
- Thư = `SP3`

Recommended division:

| Speaker | Slides | Speaking goal |
|---|---|---|
| Nam (`SP1`) | 1-8 | hook the judges, frame the problem, and bring the room smoothly into the system view |
| Nghi (`SP2`) | 9-21 | prove technical depth, fairness to `M0`, and the logic of the staged redesign |
| Thư (`SP3`) | 22-29 | convert the redesign into evidence, deliver the takeaway, and close confidently |

Why this split works:

- `SP1` handles the most audience-sensitive opening block, where tone matters more than formula density.
- `SP2` handles the heaviest technical block, where continuity from baseline to redesign is crucial.
- `SP3` handles the results block and final summary, which is ideal for strong eye contact and confident closing energy.

---

## Speaker Packs

### Nam (`SP1`) - Slides 1 to 8

Mission:

> Make the judges care before asking them to understand.

What Nam must deliver clearly:

- this is a privacy-sensitive problem, not a camera-bashing problem
- WiFi is interesting because it is passive and non-visual
- the system already has a concrete end-to-end pipeline

Tone:

- cinematic at Slides 1-6
- cleaner and more technical at Slides 7-8

Danger to avoid:

- speaking too fast in the opening
- making the first minute sound like a generic AI introduction

Closing line of Nam:

> Before talking about our contribution, we first define the baseline clearly.

Handoff from Nam to Nghi:

> We have now seen why this problem matters and what the overall system looks like. Nghi will now define the baseline and explain how we redesign it from `M0` to `M4`.

### Nghi (`SP2`) - Slides 9 to 21

Mission:

> Convince the judges that the contribution is technically real, controlled, and not a vague pile of modules.

What Nghi must deliver clearly:

- `M0` is a serious baseline
- the redesign is staged, not arbitrary
- spectral tokenization improves representation
- WiMamba is factorized into temporal and bidirectional spatial stages
- WiMamba improves efficiency
- draft-to-refine changes the prediction process
- `M4` is the integrated final pipeline

Tone:

- calm, structured, and authoritative
- less cinematic than `SP1`, but still not robotic

Danger to avoid:

- over-explaining every box on Slide 19 and Slide 20
- rushing Slide 13, because it is the logic anchor of the redesign block
- making Slides 15-17 sound like three unrelated diagrams instead of one `M2` story

Closing line of Nghi:

> With the redesign fully defined, the next question is whether these changes actually improve the model in practice.

Handoff from Nghi to Thư:

> We have now completed the redesign block, including why flow helps beyond direct regression. Thư will now show the experiments, summarize the gains, and close the talk.

### Thư (`SP3`) - Slides 22 to 29

Mission:

> Turn the technical story into convincing evidence and leave the judges with the correct final interpretation.

What Thư must deliver clearly:

- the dataset includes increasing scene difficulty
- MPJPE is the main benchmark metric
- `M3` is the accuracy winner
- `M4` is the deployment winner
- flow is the source of accuracy gain
- WiMamba is the source of practical efficiency
- the final deck also closes with limitations and future work, not only thank-you energy

Tone:

- confident, concise, and conclusive
- strongest eye contact should happen here

Danger to avoid:

- accidentally saying `M4` is the best model in every sense
- making the qualitative slide sound like proof stronger than the quantitative table
- rushing through Slide 28, because that slide is what makes the ending sound mature

Closing line of Thư:

> Thank you. We are happy to take your questions.

---

## Slide-by-Slide Script

### Slide 1 - Title

Speaker:

> Nam (`SP1`)

Target time:

> 0:18

Script:

> Imagine a bedroom. A hospital room. Or the home of an elderly person. In those spaces, we may want to understand human posture. But the moment we place a camera there, we also capture identity, appearance, and private life. Our work begins with one question: can we keep the understanding, without keeping the camera?

Delivery cue:

> Slow down. Pause after each room example. Make eye contact on the final question.

Transition:

> This is the short roadmap of the talk.

---

### Slide 2 - Content

Speaker:

> Nam (`SP1`)

Target time:

> 0:12

Script:

> In the next few minutes, we will move from that question to a concrete answer. We will first explain the motivation, then the system, then the baseline, then our staged redesign from `M0` to `M4`, and finally the experimental results.

Delivery cue:

> Keep this crisp. This slide is only to reassure the judges that the talk is structured.

Transition:

> Let us start with why this problem matters.

---

### Slide 3 - Problem & Motivation

Speaker:

> Nam (`SP1`)

Target time:

> 0:03

Script:

> First, the motivation.

---

### Slide 4 - Why camera-based sensing is not always acceptable

Speaker:

> Nam (`SP1`)

Target time:

> 0:25

Script:

> Cameras are powerful. That is not the problem. The problem is that in privacy-sensitive indoor spaces, accuracy alone is not enough. If a system understands posture by constantly watching people, then in many real environments, that solution will never truly feel acceptable.

Delivery cue:

> Stress the sentence: "That is not the problem." It makes the motivation sound balanced rather than anti-camera.

Transition:

> So the real question is whether we can keep the sensing ability without keeping the camera.

---

### Slide 5 - Can WiFi understand human posture without cameras and wearables?

Speaker:

> Nam (`SP1`)

Target time:

> 0:22

Script:

> Wearables are another alternative, but they must be worn, remembered, charged, and accepted by the user. So our question became harder, and more interesting: can WiFi infer human posture without cameras, and without wearables? That question defines both the opportunity and the difficulty of this project.

Delivery cue:

> Let the phrase "harder, and more interesting" land. This is the hook line of the motivation block.

Transition:

> The reason WiFi is attractive is not only privacy, but also practicality indoors.

---

### Slide 6 - Toward privacy-preserving indoor human sensing

Speaker:

> Nam (`SP1`)

Target time:

> 0:25

Script:

> This is why WiFi is exciting. It is already deployed in many indoor environments. It is passive. And it does not directly record what a person looks like. In other words: no camera, no wearable, just WiFi signals. That makes it a promising direction for privacy-preserving indoor sensing.

Delivery cue:

> Deliver the line "no camera, no wearable, just WiFi signals" in three clean cuts.

Transition:

> With the motivation clear, we can now show the full system at a high level.

---

### Slide 7 - Proposed System

Speaker:

> Nam (`SP1`)

Target time:

> 0:04

Script:

> So how do we turn invisible radio reflections into visible human pose?

---

### Slide 8 - Proposed System

Speaker:

> Nam (`SP1`)

Target time:

> 0:35

Script:

> At a high level, our system takes raw WiFi CSI, transforms it into model-ready features, processes them through the pose-estimation pipeline, and outputs 3D poses for one-person, two-person, and three-person indoor scenes. This is the global picture. In the next part, we will zoom in and define the baseline before explaining our redesign.

Delivery cue:

> This is the pivot from cinematic opening into technical explanation. Sound more concrete here.

Transition:

> Before talking about our contribution, we first define the baseline clearly.

---

### Slide 9 - Baseline Method

Speaker:

> Nghi (`SP2`)

Target time:

> 0:03

Script:

> Before claiming anything new, we first need to be fair to the baseline.

---

### Slide 10 - `M0` baseline: Linear projection + Transformer + direct regression

Speaker:

> Nghi (`SP2`)

Target time:

> 0:32

Script:

> `M0` is our baseline. It linearly projects CSI features, encodes them with a Transformer-style stack, and uses query-based decoding to predict final 3D poses. The important point is that `M0` is already a serious baseline. But it still predicts the final skeleton directly, in one shot.

Delivery cue:

> Emphasize "already a serious baseline" so the judges feel the later comparison is fair.

Transition:

> The next slide shows what this baseline is optimized to learn.

---

### Slide 11 - What `M0` optimizes

Speaker:

> Nghi (`SP2`)

Target time:

> 0:28

Script:

> After matching predicted instances with ground-truth persons, `M0` is supervised through classification and pose-regression related objectives. But conceptually, one thing matters most: the model is still learning the final coordinates directly. And that direct-regression setup is exactly what we later try to improve.

Delivery cue:

> End this slide with a slight pause after "directly." That pause sets up the redesign section well.

Transition:

> Now we move to our contribution: a staged redesign from `M0` to `M4`.

---

### Slide 12 - What We Propose: A Staged Redesign

Speaker:

> Nghi (`SP2`)

Target time:

> 0:03

Script:

> This section is the core contribution of our work.

---

### Slide 13 - A controlled `M0-M4` ablation ladder

Speaker:

> Nghi (`SP2`)

Target time:

> 0:25

Script:

> We organize the model family as a controlled ablation ladder. `M1` changes representation, `M2` changes the sequence model, `M3` changes the prediction objective through flow, and `M4` combines flow with WiMamba for the final efficient design.

Transition:

> The first stage improves how CSI is represented.

---

### Slide 14 - Stage 1: motion-aware spectral tokenization

Speaker:

> Nghi (`SP2`)

Target time:

> 0:30

Script:

> In `M1`, we introduce motion-aware spectral tokenization. The idea is to use spectral information to highlight dynamic motion cues and suppress static noise before deeper sequence modeling happens.

Transition:

> After improving the representation, we then redesign the sequence encoder itself.

---

### Slide 15 - Stage 2 (`M2`): Replace quadratic attention with factorized WiMamba

Speaker:

> Nghi (`SP2`)

Target time:

> 0:23

Script:

> In `M2`, we replace heavy quadratic attention with factorized WiMamba. The key point is that this is not just a generic Mamba swap. We explicitly split WiFi sequence modeling into two parts: Temporal Mamba for motion over time, and Bidirectional Spatial Mamba for two-way interaction across antenna groups.

Transition:

> The first internal stage focuses on temporal motion dynamics.

---

### Slide 16 - Stage 2a: Temporal Mamba models motion along time

Speaker:

> Nghi (`SP2`)

Target time:

> 0:22

Script:

> In the first WiMamba stage, each spatial group is treated as its own temporal sequence. After reshaping into `B times S` sequences of length `T`, Temporal Mamba scans along time to learn motion dynamics, then reshapes the features back without breaking the WiFi spatial grouping.

Transition:

> After temporal modeling, we reorganize the same features for spatial interaction.

---

### Slide 17 - Stage 2b: Bidirectional Spatial Mamba for two-way antenna interaction

Speaker:

> Nghi (`SP2`)

Target time:

> 0:25

Script:

> This is the most important detail of `M2`. After temporal encoding, each time step becomes a short spatial sequence over the `S = 9` antenna groups. We run one spatial Mamba forward and one backward, fuse both directions, and reshape back. So Bi-Mamba is applied only on the spatial axis, not on the full 180-token sequence.

Transition:

> Once the encoder is redesigned, we then change the prediction process itself.

---

### Slide 18 - Stage 3: From direct regression to draft-to-refine

Speaker:

> Nghi (`SP2`)

Target time:

> 0:28

Script:

> In stage 3, we change the prediction process itself. Instead of forcing the model to guess the final pose in one step, we first predict a draft pose, then learn how that draft should be corrected toward the target. So this is not just an extra block. It is a different prediction objective.

Transition:

> When those ideas are combined, we obtain the final `M4` design.

---

### Slide 19 - The proposed `M4` pipeline

Speaker:

> Nghi (`SP2`)

Target time:

> 0:32

Script:

> This slide shows the final `M4` pipeline. It combines spectral tokenization, WiMamba-based sequence modeling, and flow-driven refinement into one deployment-oriented model. The value of this slide is to show how the stage-wise ideas connect into one coherent system.

Transition:

> The next slide adds the training view of the same model.

---

### Slide 20 - `M4` Training Pipeline

Speaker:

> Nghi (`SP2`)

Target time:

> 0:32

Script:

> Here we show the training path of `M4`. The important idea is that the redesign affects not only the forward architecture, but also how prediction, matching, and refinement are learned together during training.

Transition:

> Before moving to the benchmark, we make the direct-regression versus flow intuition explicit one more time.

---

### Slide 21 - Why flow helps beyond direct regression

Speaker:

> Nghi (`SP2`)

Target time:

> 0:26

Script:

> Before showing the numbers, this slide makes the intuition explicit. In `M0`, the model must jump directly to the final pose in one shot. In `M3` and `M4`, the model starts from a draft pose and learns a guided correction path. That is why flow helps: it changes what the model is asked to learn.

Transition:

> With the redesign block complete, we now move to the experiments and results.

---

### Slide 22 - Experiments & Result

Speaker:

> Thư (`SP3`)

Target time:

> 0:03

Script:

> Next are the experiments and results.

---

### Slide 23 - Dataset Overview

Speaker:

> Thư (`SP3`)

Target time:

> 0:18

Script:

> Our benchmark is grounded in a real indoor acquisition setup. It includes one-person, two-person, and three-person scenes, seven volunteers, eight daily actions, and three collection locations. This matters because scene ambiguity increases significantly as more people appear in the sensing space.

Transition:

> Before comparing models, we briefly define the metric.

---

### Slide 24 - Evaluation Metric: Mean Per Joint Position Error (MPJPE)

Speaker:

> Thư (`SP3`)

Target time:

> 0:16

Script:

> We use MPJPE as the main pose-error metric. It measures the average 3D Euclidean distance between predicted joints and ground-truth joints, so lower MPJPE means better pose accuracy.

Transition:

> Now we can read the benchmark table directly.

---

### Slide 25 - Flow improves accuracy. Mamba improves deployability.

Speaker:

> Thư (`SP3`)

Target time:

> 0:40

Script:

> This is the main quantitative result. `M3` achieves the best MPJPE at `151.99` millimeters, which shows that flow-based refinement is the strongest driver of accuracy. `M4` is slightly behind on MPJPE at `159.00`, but it gives the best runtime trade-off with `159.85` FPS, only `5.83` million parameters, and `27.02` megabytes of peak memory. So the correct reading is: `M3` is the accuracy winner, and `M4` is the deployment winner.

Transition:

> The qualitative slide shows how that pattern looks in difficult scenes.

---

### Slide 26 - More coherent matched poses in challenging scenes

Speaker:

> Thư (`SP3`)

Target time:

> 0:30

Script:

> In these selected challenging cases, the flow-based variants produce more coherent matched poses than the direct-regression baseline. The purpose of this slide is not to replace the quantitative table, but to show visually that refinement helps the model preserve better structure under severe ambiguity.

Transition:

> We can now summarize the practical gains of the final design.

---

### Slide 27 - In summary

Speaker:

> Thư (`SP3`)

Target time:

> 0:25

Script:

> From `M0` to `M4`, inference speed rises from `45.69` to `159.85` FPS, parameters drop from `13.13` million to `5.83` million, peak memory falls from `155.60` to `27.02` megabytes, and MPJPE still improves from `169.34` to `159.00` millimeters. In short, the redesign makes the model both lighter and better.

Transition:

> Before we close, we also want to be transparent about the current boundaries of this work.

---

### Slide 28 - Limitations & Future Work

Speaker:

> Thư (`SP3`)

Target time:

> 0:22

Script:

> To close responsibly, we want to be transparent. The current benchmark is still based on specific indoor local snapshots, and harder multi-person scenes remain challenging. Our next steps are broader validation across more environments, stronger on-device integration, and more robust multi-person correction under severe occlusion.

Transition:

> Thank you for your attention.

---

### Slide 29 - Thanks For Your Listening

Speaker:

> Thư (`SP3`)

Target time:

> 0:05

Script:

> Thank you. We are happy to take your questions.

---

## Appendix Usage Notes

### Slide 30 - Appendix

Use:

- only when entering Q&A backup mode

### Slide 31 - Layered architecture / decoder / matching recap

Use when:

- judges ask where feature encoding, decoding, refinement, and Hungarian matching sit in one conceptual stack

Suggested answer:

> This backup slide shows the layered view from WiFi sensing to CSI feature encoding, pose decoding, refine decoding, and matching against ground truth.

### Slide 32 - Full architecture / training / loss detail

Use when:

- judges want the most complete technical picture on one screen

Suggested answer:

> This is the full technical backup where acquisition setup, architecture blocks, training path, and objectives are shown together.

### Slide 33 - `M4` Backbone and Pose Head

Use when:

- the question is specifically about internals of the final model

Suggested answer:

> This slide zooms into the final `M4` composition, especially the backbone and pose head arrangement.

### Slide 34 - Detailed qualitative figure

Use when:

- judges want more failure-case or qualitative discussion

Suggested answer:

> These extra samples show scene-level variation and help discuss where the model is more or less stable.

---

## Fast Q&A Anchors

### Q1. What is the main takeaway of the ablation?

> The ladder isolates three effects: better representation, better factorized sequence modeling, and better prediction objective. The strongest accuracy gain comes from flow, while the strongest efficiency gain comes from WiMamba.

### Q2. Why is `M3` more accurate than `M4`?

> `M3` keeps the heavier flow-based variant and achieves the best MPJPE. `M4` trades a small amount of accuracy for much stronger runtime efficiency.

### Q3. Why choose `M4` if `M3` has lower MPJPE?

> Because `M4` is the better deployment trade-off. It still improves clearly over `M0` while being much faster, smaller, and lighter.

### Q4. What exactly is the WiMamba contribution?

> Our WiMamba contribution is not only replacing Transformer with Mamba. We factorize the encoder into a Temporal Mamba stage and a Bidirectional Spatial Mamba stage, so the WiFi token structure is modeled explicitly along time and across antenna groups.

### Q5. What is the real difference between direct regression and flow?

> Direct regression predicts the final pose in one shot. Flow predicts how a draft pose should be corrected toward the target.

### Q6. How should we read the qualitative slide?

> As selected supportive cases. It shows the same trend as the benchmark, but it is not meant to replace quantitative evaluation.

---

## Delivery Checklist

- Move quickly through Slides 1-3 and 7-9 and 12 and 22 because they are structural dividers.
- Spend the most attention on Slides 10, 13-21, 25, 27, and 28.
- Say `M3 = accuracy winner` and `M4 = deployment winner` exactly and consistently.
- Treat Slide 21 as the final conceptual reinforcement before the experiments block.
- Use appendix only for Q&A, not as part of the default live run.
