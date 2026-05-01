# Phase 4 Full Slide Detail

Working title:

> Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures

This file is the detailed slide blueprint for building the first complete presentation deck. It follows the final proposal `docs/paper/NTN_IT_CT.pdf` and the figure mapping locked in `docs/presentation/plan_presentation.md`.

Core message:

> WiFi can estimate 3D human poses without cameras, and draft-to-refine Rectified Flow makes the prediction more accurate, lightweight, and practical.

Design rule:

- One slide, one message.
- Prefer large figures over paragraphs.
- Keep each slide under 40 on-slide words whenever possible.
- Use `M0` as gray/red baseline, `M3` as amber best-accuracy model, and `M4` as blue final efficient model.
- Do not claim `M4` uses bone loss. Bone loss belongs to `M5`, not the final `M4` proposal model.

---

## Deck Structure

| Slide | Role | Main Asset | Target Time |
|---|---|---|---:|
| 1 | Hook | Hook video or animation | 0:45 |
| 2 | Motivation | Camera vs Wearable vs WiFi visual | 0:45 |
| 3 | Challenge | CSI ambiguity visual | 0:55 |
| 4 | Research Gap | Direct regression failure visual | 0:55 |
| 5 | Key Idea | Draft-to-refine animation/image | 1:05 |
| 6 | Proposed Method | Figure 1 overview | 1:10 |
| 7 | Research Design | M0-M4 ablation ladder | 0:55 |
| 8 | Quantitative Results | Figure 4 quantitative | 1:20 |
| 9 | Qualitative Results | Figure 5 qualitative | 1:10 |
| 10 | Impact | Application icons | 0:50 |
| 11 | Conclusion | Summary takeaways | 0:50 |
| Total | Main deck | 11 slides | ~10:00 |

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
- Fallback: static/animated slide showing indoor motion, WiFi waves, and 3D skeleton fade-in.

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
- Do not over-explain privacy; the comparison should be visually obvious.

---

## Slide 3 - Challenge

Title:

> WiFi CSI is not an image

Main message:

> WiFi signals are indirect, noisy, and ambiguous, especially with multiple people.

On-slide content:

```text
Human motion -> multipath WiFi -> noisy CSI -> ambiguous 3D pose
```

Key labels:

- Noisy signal
- Multipath reflection
- Low spatial resolution
- Multi-person ambiguity

Visual:

- Left: person moving in room.
- Middle: CSI tensor/waveform.
- Right: ambiguous skeleton candidates.

Speaker note:

> Unlike an RGB image, CSI does not directly show body parts. Human motion changes wireless reflections, and the model must recover pose from a noisy, low-resolution signal.

Transition:

> This makes one-shot skeleton prediction unstable.

Design instruction:

- Use one flow diagram, not bullet-heavy text.
- Make the CSI tensor/wave visually distinct from the skeleton output.

---

## Slide 4 - Research Gap

Title:

> Existing models guess the skeleton in one shot

Main message:

> Direct regression can be unstable in crowded or ambiguous scenes.

On-slide content:

```text
CSI -> Final Pose
```

Problem labels:

- distorted skeleton
- duplicate/missing person
- heavy transformer computation

Visual:

- Use a crop from Figure 5 or qualitative asset.
- Highlight a visible `M0` failure with a red circle/callout.

Speaker note:

> The baseline directly predicts the final 3D skeleton. This can work in easy cases, but in crowded scenes the output may become distorted or structurally unstable.

Transition:

> Our key idea is to stop treating pose estimation as a single guess.

Design instruction:

- Do not claim `M3/M4` always have fewer false positives.
- Phrase the claim around matched pose quality and structural stability.

---

## Slide 5 - Key Idea

Title:

> From one-shot guessing to draft-to-refine

Main message:

> The model first predicts a coarse pose, then learns a velocity field to correct it.

On-slide content:

```text
Old:  CSI -> Final Pose
Ours: CSI -> Draft Pose X0 -> Velocity v -> Refined Pose X1
```

Visual:

- Use the approved 3-stage conceptual visual:
  - `Draft Pose X0`
  - `Learned Velocity v`
  - `Refined Pose X1`
- If available, use the Veo 3 animation.
- If Veo output has wrong text or anatomical skeleton artifacts, use the static image or PowerPoint Morph fallback.

Speaker note:

> Instead of directly guessing the final skeleton, we predict a draft pose first. Then Rectified Flow learns a local correction velocity and performs a one-step update toward a refined pose.

Transition:

> This idea is embedded into our full WiFi pose estimation pipeline.

Design instruction:

- Skeleton must look like an abstract keypoint graph, not a medical human skeleton.
- Keep `X0`, `v`, and `X1` readable.

---

## Slide 6 - Proposed Method

Title:

> Proposed M4 pipeline

Main message:

> M4 combines spectral CSI tokenization, efficient WiMamba encoding, and one-step Rectified Flow refinement.

On-slide content:

```text
WiFi CSI -> Spectral Tokenizer -> WiMamba Encoder -> Draft Pose -> Rectified Flow -> 3D Pose
```

Visual:

- Use **Figure 1** from the final proposal as the main method overview.
- Keep the figure large; use callouts only if necessary.

Speaker note:

> The pipeline has three main parts. First, the spectral tokenizer converts raw CSI into cleaner motion-aware tokens. Second, WiMamba models the spatio-temporal sequence efficiently. Third, the decoder predicts a draft pose and refines it with a one-step flow update.

Transition:

> To prove which component matters, we built the model as a controlled ablation ladder.

Design instruction:

- Do not overload this slide with Figure 3 details.
- If asked about internals, move to backup/technical method slide.

---

## Slide 7 - Research Design

Title:

> A controlled M0-M4 ablation ladder

Main message:

> Each model variant tests one research hypothesis.

On-slide content:

| Model | Short Name | Purpose |
|---|---|---|
| M0 | Baseline | Direct regression baseline |
| M1 | Spectral + Transformer + DETR | Test spectral representation |
| M2 | Spectral + Mamba + DETR | Test efficient sequence modeling |
| M3 | Spectral + Transformer + Flow | Test draft-to-refine flow |
| M4 | Spectral + Mamba + Flow | Final efficiency-oriented model |

Visual:

- Ladder diagram from M0 to M4.
- Add one icon per change: spectral, Mamba, flow.

Speaker note:

> The experiments are not just model comparisons. They are a controlled research ladder. M1 tests the spectral input adapter, M2 tests Mamba-based sequence modeling, M3 tests Rectified Flow, and M4 combines efficiency with the flow-based pose head.

Transition:

> The results show two complementary winners: M3 for accuracy and M4 for efficiency.

Design instruction:

- Keep names explicit enough to avoid confusion.
- Do not rename M1-M4 into overly abstract labels.

---

## Slide 8 - Quantitative Results

Title:

> Flow improves accuracy. Mamba improves deployability.

Main message:

> `M3` gives the best MPJPE, while `M4` gives the best overall efficiency trade-off.

On-slide content:

| Model | MPJPE ↓ | FPS ↑ | Params ↓ | Memory ↓ |
|---|---:|---:|---:|---:|
| M0 | 169.34 | 45.69 | 13.13M | 155.60 MB |
| M3 | **151.99** | 138.79 | 7.06M | 38.49 MB |
| M4 | 159.00 | **159.85** | **5.83M** | **27.02 MB** |

Visual:

- Main: **Figure 4** from the final proposal.
- Optional backup/side visual: **Figure 2** teaser bubble chart if there is enough room or for Q&A.

Speaker note:

> The quantitative result has two messages. M3 achieves the lowest MPJPE, so the flow formulation improves accuracy. M4 is slightly less accurate than M3, but it is the best deployment trade-off: fastest, smallest, and lowest memory among the main variants.

Transition:

> Beyond numbers, we also need to check whether the predicted skeletons look physically coherent.

Design instruction:

- Highlight `M3` in amber for best MPJPE.
- Highlight `M4` in blue for best trade-off.
- Avoid saying `M4` is best on every metric.

---

## Slide 9 - Qualitative Results

Title:

> More coherent matched poses in challenging scenes

Main message:

> Flow-based models improve matched skeleton quality in selected challenging one-, two-, and three-person cases.

On-slide content:

- Ground Truth
- M0 baseline
- M3 best accuracy
- M4 final efficient model

Visual:

- Use **Figure 5** from the final proposal.
- Prefer the layout showing one-person, two-person, and three-person rows.

Speaker note:

> This qualitative figure shows selected challenging samples. The important reading is the matched pose quality: M3 and M4 produce more coherent skeletons than the direct-regression baseline in these cases.

Transition:

> These results point to privacy-preserving indoor sensing applications.

Design instruction:

- Use honest captioning: selected challenging samples, not universal visual superiority.
- If footer shows FP, explain that the official benchmark focuses on Hungarian-matched pose error.

---

## Slide 10 - Impact

Title:

> Toward privacy-preserving indoor human sensing

Main message:

> WiFi pose estimation is valuable where cameras are inappropriate or intrusive.

On-slide content:

- Smart home monitoring
- Elderly care
- Rehabilitation
- Fall/activity monitoring
- Privacy-sensitive spaces

Visual:

- Five application icons around a central WiFi-to-pose graphic.

Speaker note:

> This is not about replacing cameras everywhere. It is about enabling pose sensing in places where cameras are not acceptable, such as private homes, care facilities, and rehabilitation scenarios.

Transition:

> To close, we summarize what the project contributes.

Design instruction:

- Keep the impact grounded.
- Avoid medical deployment claims that sound clinically validated.

---

## Slide 11 - Conclusion

Title:

> What we learned

Main message:

> Draft-to-refine flow makes WiFi 3D pose estimation more accurate and more practical.

On-slide content:

```text
Question:
Can WiFi estimate 3D human pose without cameras?

Method:
Spectral Tokenizer + WiMamba + Draft-to-Refine Rectified Flow

Result:
M3 = best accuracy
M4 = best efficiency trade-off
```

Closing line:

> WiFi can sense posture without seeing identity.

Speaker note:

> We asked whether WiFi can estimate 3D human pose without cameras. We proposed a draft-to-refine Rectified Flow framework, combined it with spectral CSI tokenization and WiMamba, and showed that it improves accuracy while moving toward real-time lightweight deployment.

Design instruction:

- Use three large takeaway cards.
- End with the privacy-preserving message, not a dense metric recap.

---

## Backup Slide A - Detailed M4 Architecture

Title:

> Inside M4: Spectral Tokenizer, WiMamba, and Flow Head

Purpose:

> Use when judges ask about the model internals.

Visual:

- Use **Figure 3** from the final proposal.

Talking points:

- Spectral input adapter extracts motion-aware CSI tokens.
- WiMamba performs factorized temporal and spatial sequence modeling.
- The pose head predicts a draft pose and refines it with one-step Rectified Flow.
- `M4` does not use bone loss.

---

## Backup Slide B - Full M0-M4 Variant Summary

Title:

> What changes from M0 to M4?

Purpose:

> Use when judges ask whether the comparison is controlled.

Content:

| Model | Spectral Adapter | Mamba | Rectified Flow | Role |
|---|---|---|---|---|
| M0 | No | No | No | Baseline |
| M1 | Yes | No | No | Spectral test |
| M2 | Yes | Yes | No | Mamba test |
| M3 | Yes | No | Yes | Flow accuracy test |
| M4 | Yes | Yes | Yes | Final trade-off model |

---

## Backup Slide C - Rectified Flow Formula

Title:

> One-step Rectified Flow refinement

Purpose:

> Use when judges ask how the refinement works.

Content:

```text
X1 = X0 + v(X0, condition)
```

Explanation:

- `X0`: draft pose.
- `condition`: query feature conditioned on encoded CSI.
- `v`: learned correction velocity.
- `X1`: refined pose.

Claim boundary:

> This slide explains the conceptual update used by the pose head. It should not be framed as iterative diffusion.

---

## Backup Slide D - False Positive / Duplicate Query Explanation

Title:

> Why can M3 show more gray poses in video?

Purpose:

> Use only if judges ask why visualization sometimes shows extra predictions.

Content:

- The decoder uses multiple learnable pose queries.
- Some low-confidence or duplicate predictions can appear in visualization.
- Official MPJPE is computed on Hungarian-matched predictions.
- Presentation mode hides unmatched poses; audit mode shows them for debugging.

Safe answer:

> M3 improves matched pose accuracy, but confidence calibration and duplicate query suppression are separate visualization issues.

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

- Hook video or hook animation.
- Figure 1 method overview.
- Figure 3 detailed M4 architecture.
- Figure 4 quantitative result.
- Figure 5 qualitative result.
- Draft-to-refine image or animation.

Recommended:

- Bubble chart/Pareto teaser as backup.
- Application icons.
- Problem slide icons.
- Audit video with `--show-unmatched` for Q&A only.

Do not use on main slides:

- Figure or video that implies `M4` uses bone loss.
- Audit video with unmatched/gray predictions unless explicitly explaining FP.
- Any generated visual that turns keypoint skeletons into anatomical human bones.

---

## First PPT Build Order

1. Create Slide 1 with hook video or fallback animation.
2. Create Slide 5 with draft-to-refine visual because it defines the core story.
3. Create Slide 6 with Figure 1 method overview.
4. Create Slide 8 with Figure 4 quantitative result.
5. Create Slide 9 with Figure 5 qualitative result.
6. Fill Slide 2-4 narrative context.
7. Fill Slide 10-11 impact and conclusion.
8. Add backup slides A-E.
9. Run a 10-minute timing pass.
10. Cut text before adding more visuals.

