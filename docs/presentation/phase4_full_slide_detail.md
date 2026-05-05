# Phase 4 Full Slide Detail

Working title:

> Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures

This file now mirrors the **final exported PDF deck**. It is synchronized after:

- confirming the exported deck has `34` slides total
- mapping the actual live order used in the final PDF
- preserving the expanded 3-slide explanation of `M2`
- normalizing spoken references to `M0` even though some slide artwork in the PDF still shows `MO`

From this point on, the purpose of this file is simple:

- document the exact slide order currently used
- explain the role of each slide in the talk
- keep main deck and appendix clearly separated

Core message of the final deck:

> Flow improves pose accuracy, while WiMamba makes the pipeline deployment-friendly.

Claim boundaries that this blueprint must preserve:

- `M0-M4` is the main narrative
- `M3` is the accuracy winner
- `M4` is the deployment trade-off winner
- qualitative evidence supports the trend but does not prove everything on its own
- appendix carries deeper architecture and training detail

---

## Deck Summary

| Range | Section | Purpose |
|---|---|---|
| 1-2 | Opening | Title and agenda |
| 3-6 | Problem & Motivation | Why this problem matters |
| 7-8 | Proposed System | Show the end-to-end idea early |
| 9-11 | Baseline Method | Define `M0` clearly |
| 12-21 | What We Propose: A Staged Redesign | Explain `M0 -> M4` |
| 22-28 | Experiments, Summary & Limits | Validate the redesign and close responsibly |
| 29 | Closing | End talk cleanly |
| 30-34 | Appendix | Q&A backup |

Current slide count:

- `29` slides in the main presentation
- `5` appendix slides
- `34` slides total

---

## Main Deck

### Slide 1 - Title

Title:

> Flow is All You Need for WiFi

Role:

> Establish topic, team identity, and the WiFi-pose + flow framing immediately.

Visual note:

- Clean title slide
- Keep the title as the main visual anchor

Speaker note:

> Open briefly and move quickly into the agenda. This slide is not for technical detail.

---

### Slide 2 - Content

Title:

> Content

Role:

> Show the 6-part structure so the audience knows the talk is organized and finite.

On-slide structure:

```text
01 Problem & Motivation
02 Proposed System
03 Baseline Method
04 What We Propose: A Staged Redesign
05 Experiments & Results
06 Appendix
```

Speaker note:

> Mention the roadmap in one breath, then move on. Do not spend time reading every line slowly.

---

### Slide 3 - Section Divider

Title:

> Problem & Motivation

Role:

> Visual reset before the first content block.

Speaker note:

> Use as a transition slide only.

---

### Slide 4 - Why camera-based sensing is not always acceptable

Role:

> Frame the motivation through privacy and practical limitations of vision-based sensing.

Main message:

> The issue is not that cameras are weak; the issue is that they are not always appropriate.

Visual note:

- Keep the comparison readable
- Emphasize privacy-sensitive environments

Speaker note:

> This is the first motivation slide. Keep it human-centered rather than technical.

---

### Slide 5 - Can WiFi understand human posture without cameras and wearables?

Role:

> Convert the motivation into the central research question.

Main message:

> We want pose understanding without appearance capture and without body-worn devices.

Visual note:

- Let the question itself dominate the slide
- This is a rhetorical bridge, not a methods slide

Speaker note:

> Ask the question clearly and let it land before moving into the answer direction.

---

### Slide 6 - Toward privacy-preserving indoor human sensing

Role:

> Reframe WiFi sensing as a practical indoor alternative with clear application relevance.

Main message:

> WiFi-based sensing is attractive because it is passive, already deployed indoors, and less identity-revealing than cameras.

Visual note:

- Keep applications secondary
- Main goal is to close the motivation block strongly

Speaker note:

> End this block by linking privacy motivation to indoor use cases.

---

### Slide 7 - Section Divider

Title:

> Proposed System

Role:

> Transition from problem framing into technical framing.

Speaker note:

> Use as a short pivot only.

---

### Slide 8 - Proposed System

Role:

> Give the audience one global picture before diving into baseline and redesign details.

Main message:

> The system takes WiFi CSI, processes it through the model pipeline, and outputs multi-person 3D pose for indoor applications.

Visual note:

- This slide is the big-picture overview
- It can be slightly denser because later slides unpack it

Speaker note:

> Use this slide to orient the audience, not to explain every box in detail.

---

### Slide 9 - Section Divider

Title:

> Baseline Method

Role:

> Mark the start of the baseline explanation block.

Speaker note:

> Move quickly to the next slide.

---

### Slide 10 - `M0` baseline: Linear projection + Transformer + direct regression

Role:

> Define the baseline architecture clearly and fairly.

Main message:

> `M0` is a serious DETR/PETR-style baseline built around direct regression.

Visual note:

- Keep `M0` naming consistent
- Show the architecture as a clean left-to-right path

Speaker note:

> Emphasize that `M0` is not a strawman. It is the baseline that makes the later comparison meaningful.

---

### Slide 11 - What `M0` optimizes

Role:

> Explain the baseline training objective before showing any redesign.

Main message:

> The baseline ultimately learns final pose coordinates through a matched regression-style objective.

Visual note:

- Formula is the center of attention
- The purpose is conceptual, not implementation-exhaustive

Speaker note:

> Keep the explanation on what the baseline is learning, not on every symbol equally.

---

### Slide 12 - Section Divider

Title:

> What We Propose: A Staged Redesign

Role:

> Announce the contribution block clearly.

Speaker note:

> This divider matters because it signals that everything after this point is the team contribution story.

---

### Slide 13 - A controlled `M0-M4` ablation ladder

Role:

> Show that the redesign is incremental and interpretable.

Main message:

> Each model stage isolates one idea so the comparison remains controlled.

Visual note:

- Make the ladder easy to scan
- Keep `M0 -> M4` progression obvious

Speaker note:

> This slide sets the logic for the next five slides. Do not rush it completely.

---

### Slide 14 - Stage 1: motion-aware spectral tokenization

Role:

> Explain the representation upgrade introduced at `M1`.

Main message:

> The first change is to make CSI tokens more motion-aware before deeper modeling.

Visual note:

- Formula plus intuitive data-flow is enough
- Do not over-expand FFT discussion verbally

Speaker note:

> Frame this as input representation improvement, not as the main accuracy breakthrough.

---

### Slide 15 - Stage 2 (`M2`): Replace quadratic attention with factorized WiMamba

Role:

> Explain why the sequence encoder is redesigned and make the factorized WiMamba contribution visible.

Main message:

> `M2` replaces full pairwise attention with a WiFi-specific factorization: Temporal Mamba plus Bidirectional Spatial Mamba.

Visual note:

- Complexity contrast must stay visible
- This slide should not read as "generic Mamba replacement"
- Make the two internal parts of WiMamba visible even before the next two slides

Speaker note:

> The audience should leave knowing that `M2` changes the encoder, not the head, and that the WiMamba design is factorized on purpose.

---

### Slide 16 - Stage 2a: Temporal Mamba models motion along time

Role:

> Explain the first internal stage inside the WiMamba block.

Main message:

> Temporal Mamba scans along the time axis for each spatial group independently, so motion dynamics are learned before spatial aggregation.

Visual note:

- Input shape, temporal reshape, and output shape must stay explicit
- Make it obvious that the scan is along `T`, not over the full mixed token set

Speaker note:

> This slide proves that the team contribution is not just "we used Mamba", but "we structured Mamba for WiFi time dynamics."

---

### Slide 17 - Stage 2b: Bidirectional Spatial Mamba for two-way antenna interaction

Role:

> Explain the second internal stage inside the WiMamba block and highlight the Bi-Mamba contribution clearly.

Main message:

> After temporal encoding, each time step is reorganized as a short spatial sequence over `S = 9` antenna groups and processed in both forward and backward directions.

Visual note:

- Input tensor, transpose/reshape, forward pass, backward pass, and fused output must all be visible
- The bottom callout should state that Bi-Mamba is applied on the spatial axis only, not on the full 180-token sequence

Speaker note:

> This is the most important technical defense slide of `M2`. Make the two-way spatial interaction story very clear.

---

### Slide 18 - Stage 3: From direct regression to draft-to-refine

Role:

> Introduce the conceptual jump from final-coordinate prediction to refinement-based prediction.

Main message:

> The major change is not only a new module, but a new way of predicting pose.

Visual note:

- Keep the draft pose and refined pose distinction visually obvious
- This slide should feel like the conceptual center of the contribution

Speaker note:

> Make sure the audience hears the phrase draft-to-refine clearly.

---

### Slide 19 - The proposed `M4` pipeline

Role:

> Show how the redesigned components fit together in the final efficient model.

Main message:

> `M4` combines spectral tokenization, WiMamba, and flow-based refinement into one coherent pipeline.

Visual note:

- This is the full-system synthesis slide
- Use it to connect stage-wise ideas back into one architecture

Speaker note:

> Do not read every block. Use it to show assembly, not to repeat all prior details.

---

### Slide 20 - `M4` Training Pipeline

Role:

> Show how the final model is trained and where matching, draft generation, and refinement fit.

Main message:

> The redesign affects both the forward pipeline and the training signal path.

Visual note:

- This is the densest technical slide in the main deck
- It works as a bridge from architecture to experiments

Speaker note:

> Explain the training story at a high level. Do not let this become a wall-of-boxes narration.

---

### Slide 21 - Why flow helps beyond direct regression

Role:

> Place the direct-regression versus flow intuition before the evidence block.

Main message:

> Flow changes the learning problem from one-shot final-pose guessing to guided correction from a draft pose.

Visual note:

- The comparison between `M0` and `M3/M4` should stay visually central
- This slide is still part of the contribution story, not yet the evidence block

Speaker note:

> This is the last conceptual reinforcement before the experiments begin.

---

### Slide 22 - Section Divider

Title:

> Experiments & Result

Role:

> Start the evidence block exactly as it appears in the final PDF.

Speaker note:

> Use only as a transition.

---

### Slide 23 - Dataset Overview

Role:

> Ground the experiments in the actual data setting and task scale.

Main message:

> The benchmark covers indoor WiFi sensing with one-person, two-person, and three-person scenes, plus concrete setup details, sample counts, and action diversity.

Visual note:

- The acquisition photo, signal examples, and sample-count table should all remain readable
- This slide should make the dataset feel real and bounded

Speaker note:

> This slide should be quick. It is context, not the key argument.

---

### Slide 24 - Evaluation Metric: Mean Per Joint Position Error (MPJPE)

Role:

> Define the benchmark metric before quantitative comparison.

Main message:

> MPJPE is the main error metric used to compare pose quality.

Visual note:

- Keep the formula or definition easy to read
- This is a standards slide, not a contribution slide

Speaker note:

> Define the metric briefly so the audience can interpret the result table correctly.

---

### Slide 25 - Flow improves accuracy. Mamba improves deployability.

Role:

> Deliver the main quantitative result of the paper.

Main message:

> `M3` wins on MPJPE, while `M4` wins on overall efficiency trade-off.

Visual note:

- Table and visual comparison should be the main focus
- Make sure `M0` is the baseline reference point

Speaker note:

> This is the slide where the audience should clearly hear `M3 = accuracy winner` and `M4 = deployment winner`.

---

### Slide 26 - More coherent matched poses in challenging scenes

Role:

> Support the benchmark with qualitative examples the audience can inspect visually.

Main message:

> In selected difficult cases, flow-based variants preserve more coherent body structure than the baseline.

Visual note:

- Use GT / `M0` / `M3` / `M4` comparison clearly
- Focus on matched pose quality, not raw query clutter

Speaker note:

> State explicitly that these are selected challenging cases, not universal proof.

---

### Slide 27 - In summary

Role:

> Compress the practical gains of the final design into headline numbers.

Main message:

> The final system improves speed, size, memory, and still beats the baseline on MPJPE while reducing complexity from `O(L^2)` to `O(L)`.

Visual note:

- Let the numeric gains dominate the slide
- This is a factual recap slide, not a new conceptual slide

Speaker note:

> Read this slide as measured deltas from `M0` to `M4`, not as a generic conclusion.

---

### Slide 28 - Limitations & Future Work

Role:

> Close the technical story with a mature view of current boundaries and next steps.

Main message:

> The current benchmark is promising but still local, and the next step is broader validation plus stronger on-device integration.

Visual note:

- Keep the split between current limitations and future work explicit
- This slide should sound honest and forward-looking, not defensive

Speaker note:

> This slide helps the team sound research-mature and trustworthy.

---

### Slide 29 - Thanks For Your Listening

Role:

> End the live presentation and open the floor for questions.

Speaker note:

> Keep it brief and transition naturally into Q&A.

---

## Appendix

### Slide 30 - Section Divider

Title:

> Appendix

Role:

> Separate live talk from backup material cleanly.

---

### Slide 31 - Layered architecture / decoder / matching recap

Role:

> Backup slide for explaining the end-to-end layering from WiFi sensing to decoder outputs and Hungarian matching.

Use when:

- judges ask where decoding, refinement, and matching sit in the overall pipeline
- someone wants a simpler conceptual picture than the dense architecture slides

---

### Slide 32 - Full architecture / training / loss detail

Role:

> Deep technical backup for the acquisition setup, spectral tokenizer, WiMamba encoder, draft-to-refine flow decoder, and combined loss.

Use when:

- judges ask for the one-slide full-system picture
- someone wants to connect architecture blocks to the training losses

---

### Slide 33 - `M4` Backbone and Pose Head

Role:

> Backup slide focused on the internal composition of the final model.

Use when:

- the question is specifically about backbone design or pose head design

---

### Slide 34 - Detailed qualitative figure

Role:

> Backup slide for extended visual comparison and error discussion.

Use when:

- judges want more than one qualitative example
- someone asks about failure modes or scene difficulty

---

## Presentation Notes

What this blueprint now guarantees:

- `M0` naming is consistent with the actual deck
- the final PDF order is reflected exactly, including `Why flow helps...` before the experiments block
- `M2` is now unpacked into overview, Temporal Mamba, and Bi-Mamba slides
- the documents no longer describe the older 13-slide proposal as if it were the active deck
- the main deck now includes both `In summary` and `Limitations & Future Work` before thanks

What this blueprint intentionally does not do:

- it does not propose more content changes
- it does not reopen wording issues already accepted by the team
- it does not move appendix material back into the main deck
