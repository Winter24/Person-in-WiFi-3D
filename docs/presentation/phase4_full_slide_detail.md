# Phase 4 Full Slide Detail

Working title:

> Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures

This file defines the revised **13-slide main deck** for the ResFes presentation. The deck follows the final proposal `docs/paper/NTN_IT_CT.pdf`, but it pulls the most important technical formulas from backup into the main story so the audience can understand the contribution relative to `M0`.

Core message:

> We do not just report that `M3` is more accurate and `M4` is faster. We explain why, stage by stage, by moving from direct regression to draft-to-refine trajectory learning.

Design rules:

- One slide, one message.
- Prefer large figures, short equations, and tight callouts over paragraphs.
- Keep each slide under 40 on-slide words whenever possible.
- Use `M0` as gray/red baseline, `M3` as amber best-accuracy model, and `M4` as blue final efficient model.
- Treat the `M0-M4` deck as a **no-bone-loss runtime snapshot**. The repo exposes an optional bone-loss branch, but it is off in the runs discussed on these slides.
- Do not claim `M4` wins every metric.
- Do not equate false positives in visualization with official matched MPJPE.

---

## Deck Structure

| Slide | Role | Main Asset | Target Time |
|---|---|---|---:|
| 1 | Hook | Hook video or animation | 0:35 |
| 2 | Motivation | Camera vs Wearable vs WiFi visual | 0:35 |
| 3 | Challenge + Gap | CSI ambiguity + one-shot failure | 0:45 |
| 4 | Stage 0 | Baseline `M0` architecture | 0:45 |
| 5 | Stage 0 | Baseline `M0` loss | 0:40 |
| 6 | Stage 1 | Spectral Tokenizer formula | 0:45 |
| 7 | Stage 2 | WiMamba formula and complexity | 0:45 |
| 8 | Stage 3 | Draft-to-Refine flow head | 0:55 |
| 9 | Stage 3 | Flow Matching loss | 0:45 |
| 10 | Stage Summary | `M0-M4` controlled ladder | 0:45 |
| 11 | Results | Figure 4 quantitative | 1:10 |
| 12 | Results | Figure 5 qualitative | 1:00 |
| 13 | Conclusion | Takeaways + impact | 0:35 |
| Total | Main deck | 13 slides | ~10:00 |

---

## Slide 1 - Hook

Title:

> Can WiFi understand human posture without cameras?

Main message:

> Human posture can be inferred from wireless signals, not visual appearance.

On-slide content:

- No camera.
- No wearable.
- Just WiFi signals.

Visual:

- Use the rendered hook video from `render_presentation_video.py` if it is clean enough.
- Fallback: static or animated slide showing indoor motion, WiFi waves, and 3D skeleton fade-in.

Recommended asset:

- Presentation video: `presentation_S23_12_M0_vs_M3_presentation.mp4`
- Do not use audit video with unmatched predictions on this slide.

Speaker note:

> A camera can estimate human pose, but it also captures human identity. Our project asks a different question: can we estimate 3D human pose using only WiFi signals?

Transition:

> To see why this matters, let us compare WiFi sensing with common alternatives.

Design instruction:

- Full-bleed video or large central visual.
- Keep text minimal and cinematic.

---

## Slide 2 - Motivation

Title:

> Why camera-based sensing is not always acceptable

Main message:

> Cameras are powerful but privacy-sensitive; WiFi offers a privacy-preserving sensing path.

On-slide content:

| Camera | Wearable | WiFi |
|---|---|---|
| Captures appearance | Requires device | No visual identity |
| Privacy concern | User compliance | Passive sensing |
| Sensitive spaces issue | Inconvenient | Indoor-ready |

Visual:

- Three-column comparison with icons.
- Camera column slightly red/gray, Wearable amber, WiFi blue.

Speaker note:

> Cameras work well, but they capture appearance and identity. Wearables reduce that problem, but users must wear devices. WiFi is already present in many indoor spaces and does not record visual identity.

Transition:

> But WiFi is not an image, so the technical challenge is very different.

Design instruction:

- Use icons and short phrases only.
- Do not over-explain privacy.

---

## Slide 3 - Challenge + Research Gap

Title:

> CSI is indirect, and one-shot regression is brittle

Main message:

> WiFi CSI is noisy, ambiguous, and hard to map directly to precise 3D coordinates.

On-slide content:

```text
Human motion -> multipath CSI -> ambiguity -> unstable one-shot pose
```

Key labels:

- Noisy signal
- Multipath reflection
- Low spatial resolution
- Multi-person ambiguity

Visual:

- Left: person moving in room.
- Middle: CSI tensor or waveform.
- Right: one highlighted `M0` failure from a crowded scene.

Speaker note:

> CSI does not directly show body parts. The model must infer 3D pose from wireless reflections, and that makes direct one-shot prediction structurally unstable in hard scenes.

Transition:

> So before presenting our upgrades, we first need to define the baseline clearly.

Design instruction:

- This slide replaces the old split between challenge and research gap.
- Keep the visual flow simple and direct.

---

## Slide 4 - Stage 0: Baseline `M0` Architecture

Title:

> `M0` baseline: linear projection + Transformer + direct regression

Main message:

> `M0` is a strong DETR/PETR-style baseline, but it predicts the final skeleton in one shot.

On-slide content:

```text
CSI X -> Linear Adapter -> Transformer Encoder/Decoder -> Pose Y_hat
```

Small callouts:

- 180 CSI tokens
- set prediction
- 100 learnable pose queries

Visual:

- Preferred: simplified redraw of proposal Figure 7.
- Fallback: a clean four-block diagram derived from the implementation.

Speaker note:

> In the baseline, raw CSI is projected linearly, encoded by a Transformer, and decoded through learnable pose queries into final 3D coordinates. The key limitation is that the model must guess the final pose directly.

Transition:

> The next question is what objective this baseline is actually optimizing.

Design instruction:

- This slide must feel concrete and implementation-grounded.
- Use `M0` in gray or red to visually anchor the baseline.

---

## Slide 5 - Stage 0: Baseline `M0` Loss

Title:

> What `M0` optimizes

Main message:

> `M0` learns direct coordinate regression after Hungarian matching.

On-slide content:

```text
L_M0 = L_match
     + lambda_cls L_cls
     + lambda_kpt L_kpt
     + lambda_oks L_oks
     + lambda_ref L_refine
```

Equation reading:

- `L_match`: Hungarian bipartite matching between predicted poses and GT persons
- `L_cls`: query-level person classification loss
- `L_kpt`: direct keypoint coordinate regression loss
- `L_oks`: structure-aware pose consistency term
- `L_refine`: decoder-side coordinate refinement loss

Bottom note:

- Main-slide version only. Auxiliary implementation details stay in backup.

Visual:

- Formula-centered slide with a small side note:
  - Hungarian matching
  - direct coordinate supervision

Speaker note:

> After one-to-one matching, the baseline optimizes classification, keypoint regression, OKS consistency, and refinement losses. But the target is still the final pose coordinates themselves, not a correction trajectory.

Transition:

> Our first upgrade does not change the prediction head yet. It improves the CSI representation.

Design instruction:

- Keep this readable, not dense.
- The audience should leave with one memory: `M0` is still direct regression.

---

## Slide 6 - Stage 1: Spectral Tokenizer (`M1`)

Title:

> Stage 1: motion-aware spectral tokenization

Main message:

> `M1` improves the input representation before changing the sequence model or prediction objective.

On-slide content:

```text
X_lin  = W_in X
X_time = DWConv1D(X_lin)
D      = Mean_c |RFFT(X_lin)|
G      = sigma(MLP(D))
X_spec = LN(X_lin + W_c (X_time odot G))
```

Right-side callout:

- Highlight dynamic motion cues
- Suppress static noise

Visual:

- Clean equation on the left, small pipeline illustration on the right.
- Optional mini-visual: raw CSI -> gated temporal features.

Speaker note:

> `M1` keeps the PETR-style prediction stack, but replaces the plain linear adapter with a motion-aware tokenizer. The key idea is to derive a Doppler-like profile from the frequency domain and use it to gate temporal features.

Transition:

> Once the representation is improved, we ask whether the heavy Transformer is still necessary.

Design instruction:

- Do not overload the slide with FFT theory.
- The story is denoising plus motion emphasis, not signal-processing depth for its own sake.

---

## Slide 7 - Stage 2: WiMamba Encoder (`M2`)

Title:

> Stage 2: replace quadratic attention with factorized WiMamba

Main message:

> `M2` targets efficiency by factorizing temporal and spatial sequence modeling.

On-slide content:

```text
X^(l+1/2) = X^(l) + Mamba_t(LN(X^(l)))
X^(l+1)   = X^(l+1/2) + BiMamba_s(LN(X^(l+1/2)))
```

Complexity box:

```text
Transformer: O(L^2 * C)
WiMamba:     O(L * C)
```

Visual:

- Two-branch diagram:
  - temporal pass
  - bidirectional spatial pass

Speaker note:

> `WiMamba` decouples temporal and spatial modeling. Instead of full quadratic self-attention over the whole token sequence, it uses factorized Mamba blocks, which makes the encoder more deployment-friendly.

Transition:

> But better representation and faster encoding still do not solve the core one-shot prediction problem.

Design instruction:

- This is the efficiency slide, not the final-accuracy slide.
- Keep the complexity comparison visible.

---

## Slide 8 - Stage 3: Draft-to-Refine Flow Head (`M3/M4`)

Title:

> Stage 3: from direct regression to draft-to-refine

Main message:

> The main contribution is changing the prediction process from one-shot coordinates to trajectory-based correction.

On-slide content:

```text
X_t     = t X_1 + (1 - t) X_0
v_hat   = v_theta(X_t, t, c)
X_1_hat = X_0 + v_hat
```

Label mapping:

- `X0`: draft pose
- `c`: query feature from encoded CSI
- `v_theta`: learned correction velocity
- `X1_hat`: refined pose

Equation reading:

- First line: build an interpolation path between draft pose and target pose
- Second line: predict the correction velocity at time `t`
- Third line: apply one-step refinement from draft to final estimate

Visual:

- Use the draft-to-refine conceptual visual from Slide 5, but now with the equation.
- If space allows, add Figure 1 as a faded background or small inset.

Speaker note:

> Instead of predicting the final pose directly, the model first predicts a draft pose `X0`. Then a velocity network learns how that draft should move toward the target. A one-step update produces the refined pose.

Transition:

> To make this work, we need a new objective, not just a new decoder block.

Design instruction:

- This is the most important technical slide in the deck.
- Keep the mapping from symbols to intuition explicit.

---

## Slide 9 - Stage 3: Flow Matching Loss

Title:

> Why flow helps beyond direct regression

Main message:

> `M3/M4` learn a correction field, not only final coordinates.

On-slide content:

```text
L_flow = E || v_theta(X_t, t, c) - (X_1 - X_0) ||_2^2
```

Comparison box:

```text
M0:    predict final pose directly
M3/M4: predict how a draft should move
```

Equation reading:

- target velocity: `X_1 - X_0`
- predicted velocity: `v_theta(X_t, t, c)`
- objective: minimize the squared gap between predicted correction and ideal correction along the path

Visual:

- Formula-centered slide with two-column comparison:
  - direct regression
  - trajectory refinement

Speaker note:

> This loss tells the model to learn the correction direction from the draft pose toward the ground truth. That is the conceptual shift: the target is now a velocity field rather than a one-shot final coordinate guess.

Transition:

> With all three upgrades defined, we can summarize the full `M0-M4` ladder.

Design instruction:

- Do not mention bone loss on this slide.
- Keep the contrast with `M0` explicit and simple.

---

## Slide 10 - Stage Summary: `M0-M4`

Title:

> What changes from `M0` to `M4`?

Main message:

> Each model adds one controlled idea, so the evidence path is interpretable.

On-slide content:

| Model | Input | Sequence Model | Head | Training Signal | Main Purpose |
|---|---|---|---|---|---|
| M0 | Linear | Transformer | Direct regression | matching + regression | baseline |
| M1 | Spectral | Transformer | Direct regression | same as M0 | representation |
| M2 | Spectral | WiMamba | Direct regression | same as M0 | efficiency |
| M3 | Spectral | Transformer | Draft + Flow | + flow matching | accuracy |
| M4 | Spectral | WiMamba | Draft + Flow | + flow matching | trade-off |

Visual:

- Keep the table large and readable.
- Optional top banner: `representation -> efficiency -> refinement`.

Speaker note:

> This is not a loose set of variants. It is a controlled ladder. `M1` tests the tokenizer, `M2` tests the sequence model, `M3` tests the flow objective, and `M4` combines flow with efficient sequence modeling.

Transition:

> With the stage logic in place, we can now read the benchmark results correctly.

Design instruction:

- This slide should make the ablation logic feel rigorous.
- Do not add extra variants here.

---

## Slide 11 - Quantitative Results

Title:

> Flow improves accuracy. WiMamba improves deployability.

Main message:

> `M3` is the accuracy winner, while `M4` is the practical trade-off winner.

On-slide content:

- Use Figure 4 as the main visual.
- Add two callouts:
  - `M3 = best MPJPE`
  - `M4 = best speed-size-memory trade-off`

Optional micro-callout:

```text
vs M0:
M3: -17.35 mm MPJPE
M4: +114.16 FPS
```

Visual:

- Figure 4 from the proposal as the main graphic.
- Optional small reduced table for `M0`, `M3`, `M4`.

Speaker note:

> The benchmark gives two complementary findings. `M3` achieves the lowest MPJPE at `151.99 mm`, confirming the accuracy benefit of flow-based refinement. `M4` is slightly less accurate than `M3`, but it reaches `159.85 FPS` with only `5.83M` parameters and `27.02 MB` memory, which makes it the strongest deployment-oriented trade-off in this frozen local snapshot.

Transition:

> The numbers tell the overall trend, and the qualitative cases show how that trend looks in practice.

Design instruction:

- Use the phrase `preliminary frozen local snapshot` if needed in spoken context.
- Never say `M4` is the most accurate model.

---

## Slide 12 - Qualitative Results

Title:

> More coherent matched poses in difficult scenes

Main message:

> Compared with `M0`, the flow-based variants preserve more coherent body structure in selected crowded cases.

On-slide content:

- Ground Truth
- `M0` baseline
- `M3` best-accuracy model
- `M4` final efficient model

Caption:

> Compared with `M0`, flow-based variants preserve more coherent body structure in difficult scenes.

Visual:

- Use Figure 5 from the proposal.
- Highlight one or two visible `M0` failure patterns with clean callouts.

Speaker note:

> These are selected challenging samples, not universal proof. The right way to read this figure is matched pose quality: the flow-based variants remain more coherent than the direct-regression baseline in difficult multi-person scenes.

Transition:

> Let me close with the three takeaways that matter most.

Design instruction:

- Avoid any annotation that suggests `M4` has bone loss enabled.
- If extra gray predictions appear in video, keep that discussion for backup or Q&A.

---

## Slide 13 - Conclusion + Impact

Title:

> Three takeaways from `M0` to `M4`

Main message:

> The contribution is not one new block. It is a staged redesign from direct regression to efficient trajectory refinement.

On-slide content:

- `M0` is a strong direct-regression baseline.
- Flow is the main accuracy driver.
- WiMamba makes the flow pipeline practical.
- Useful for privacy-sensitive indoor sensing.

Visual:

- Three large takeaway cards plus one small application row:
  - smart home
  - elderly care
  - rehabilitation

Speaker note:

> In summary, we start from a strong `M0` baseline, improve representation with spectral tokenization, improve efficiency with WiMamba, and improve accuracy with draft-to-refine flow learning. The result is a privacy-preserving WiFi pose pipeline that is both interpretable and practical.

Transition:

> Thank you. We are happy to take questions.

Design instruction:

- End with the research takeaway first, the application takeaway second.
- Do not finish on a dense metric recap.

---

## Backup Slide A - Detailed `M4` Architecture

Title:

> Inside `M4`: tokenizer, WiMamba, and flow head

Purpose:

> Use when judges ask how the final efficient model is assembled internally.

Visual:

- Use Figure 3 from the final proposal.

Talking points:

- Spectral tokenizer for motion-aware CSI features
- Factorized WiMamba sequence encoder
- Draft pose plus one-step flow refinement
- The main-deck `M0-M4` runs are treated as bone-loss-off at runtime

---

## Backup Slide B - Full `M0-M4` Variant Summary

Title:

> What changes at each stage?

Purpose:

> Use when judges ask whether the comparison is controlled.

Content:

| Model | Spectral Adapter | WiMamba | Rectified Flow | Role |
|---|---|---|---|---|
| M0 | No | No | No | Baseline |
| M1 | Yes | No | No | Representation test |
| M2 | Yes | Yes | No | Efficiency test |
| M3 | Yes | No | Yes | Accuracy-focused flow variant |
| M4 | Yes | Yes | Yes | Final trade-off model |

---

## Backup Slide C - False Positive / Duplicate Query Explanation

Title:

> Why can some videos show extra gray poses?

Purpose:

> Use only if judges ask about extra predictions in qualitative videos.

Content:

- Query-based decoding can emit low-confidence or duplicate poses.
- Official MPJPE is computed on Hungarian-matched predictions.
- Presentation mode hides unmatched poses; audit mode exposes them.

Safe answer:

> Extra gray poses in visualization are not the same thing as the matched-pose benchmark used in the reported MPJPE table.

---

## Backup Slide D - Runtime Note on Bone Loss

Title:

> Why is bone loss not part of this deck?

Purpose:

> Use only if someone notices that the repo exposes an optional `BoneLengthLoss` branch.

Content:

- The repository can expose an optional bone-loss branch.
- The `M0-M4` runs discussed in this deck treat that branch as disabled at runtime.
- This keeps the main ablation ladder focused on:
  - spectral tokenization
  - WiMamba
  - Rectified Flow

Safe answer:

> Bone loss is not part of the main `M0-M4` evidence story presented here. The runtime snapshot for this deck keeps the ablation focused on representation, sequence modeling, and trajectory refinement.

---

## Backup Slide E - Responsible AI Statement

Title:

> Responsible use of AI-assisted research tools

Purpose:

> Align with ResFes AI responsibility requirements.

Content:

- AI tools were used for drafting, visualization planning, and language refinement.
- Technical claims, code, experiments, and final decisions were reviewed by team members.
- The team remains responsible for correctness, originality, and research integrity.

---

## Asset Checklist Before Building PPT

Mandatory:

- Hook video or hook animation
- Slide 4 baseline architecture visual
- Slide 6 spectral tokenizer visual
- Slide 7 WiMamba visual
- Slide 8 draft-to-refine visual
- Figure 4 quantitative result
- Figure 5 qualitative result

Recommended:

- Figure 1 as support for Slide 8 or backup
- Figure 3 as backup technical architecture
- Reduced `M0/M3/M4` result table
- Application icons for Slide 13

Do not use on main slides:

- Any visual implying that the main-deck `M4` uses bone loss
- Audit video with unmatched poses unless explicitly explaining FP behavior
- Any generated skeleton visual that looks like anatomical bones or a medical body diagram

---

## First PPT Build Order

1. Build Slide 11 and Slide 12 first, because the evidence slides anchor the narrative.
2. Build Slide 4 and Slide 5 to define `M0` clearly.
3. Build Slide 6 to Slide 9 as the technical stage sequence.
4. Build Slide 10 as the summary table.
5. Build Slide 1 to Slide 3 as framing.
6. Build Slide 13 as the final takeaway slide.
7. Add backup slides A to E.
8. Run a 10-minute timing pass.
9. Cut text before adding more graphics.
