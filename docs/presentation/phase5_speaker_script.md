# Phase 5 Speaker Script

Working title:

> Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures

Purpose of this file:

- Convert the Phase 4 slide blueprint into a 9:30-10:00 oral script.
- Give each slide a clear opening sentence, 2-3 talking points, and a transition.
- Keep the talk natural enough to present without reading directly from slides.
- Prevent technical over-claims, especially around `M3`, `M4`, false positives, and bone loss.

Core thesis to repeat:

> We turn WiFi-based 3D pose estimation from one-shot guessing into draft-to-refine trajectory refinement.

Important claim boundaries:

- `M3` is the best MPJPE model.
- `M4` is the best overall efficiency trade-off model.
- `M4` does not use bone loss. Bone loss belongs to `M5`, not the final proposal model.
- Visualization false positives are not the same as official matched MPJPE.
- The draft-to-refine animation is conceptual unless explicitly generated from model vectors.

---

## Timing Plan

| Section | Slides | Time |
|---|---|---:|
| Hook + motivation | 1-2 | 1:30 |
| Challenge + gap | 3-4 | 1:50 |
| Key idea + method | 5-6 | 2:15 |
| Research design | 7 | 0:55 |
| Results | 8-9 | 2:20 |
| Impact + conclusion | 10-11 | 1:10 |
| Total target | 11 slides | 10:00 |

Rehearsal target:

> Aim for 9:30 so the real presentation has a 30-second safety margin.

---

## Speaker Split Option

If using 2 speakers:

| Speaker | Slides | Role |
|---|---|---|
| Speaker 1 | 1-5 | Problem, motivation, key idea |
| Speaker 2 | 6-11 | Method, experiments, results, impact |

If using 3 speakers:

| Speaker | Slides | Role |
|---|---|---|
| Speaker 1 | 1-4 | Motivation and research gap |
| Speaker 2 | 5-7 | Key idea, method, ablation design |
| Speaker 3 | 8-11 | Results, impact, conclusion |

Speaker handoff line from Speaker 1 to Speaker 2:

> Now that we understand why WiFi pose estimation is difficult, I will hand over to my teammate to explain our draft-to-refine solution.

Speaker handoff line from Speaker 2 to Speaker 3:

> With this controlled design in place, my teammate will show what the experiments tell us.

---

## Slide 1 - Hook

Title:

> Can WiFi understand human posture without cameras?

Target time:

> 0:45

Opening sentence:

> A camera can estimate human pose, but it also captures human identity.

Full script:

> A camera can estimate human pose, but it also captures human identity. Our project starts from a different question: can we estimate 3D human posture using only WiFi signals? In this work, we explore how indoor WiFi signals can be transformed into multi-person 3D skeletons without cameras or wearable sensors.

Key points to remember:

- Start with privacy, not architecture.
- Let the video or visual carry the first impression.
- Keep the tone curious, not exaggerated.

Transition:

> To see why this matters, let us compare WiFi with the sensing methods that are commonly used today.

Short version if over time:

> Cameras see posture, but they also see identity. Our question is whether WiFi can estimate 3D pose without seeing people visually.

---

## Slide 2 - Motivation

Title:

> Why camera-based sensing is not always acceptable

Target time:

> 0:45

Opening sentence:

> The motivation is not that cameras are weak; it is that cameras are often inappropriate.

Full script:

> The motivation is not that cameras are weak; it is that cameras are often inappropriate. Cameras are accurate, but they capture appearance and raise privacy concerns. Wearables avoid visual identity, but they require the user to wear and maintain a device. WiFi is different: it is already available in many indoor spaces, and it senses motion without recording visual appearance.

Key points to remember:

- Do not attack cameras too strongly.
- Emphasize “where cameras are not acceptable.”
- WiFi is privacy-preserving because it does not capture appearance, not because it has no privacy concerns at all.

Transition:

> However, using WiFi for pose estimation creates a much harder technical problem.

Short version if over time:

> Cameras are accurate but privacy-sensitive. Wearables are inconvenient. WiFi gives us a passive, non-visual sensing signal for indoor spaces.

---

## Slide 3 - Challenge

Title:

> WiFi CSI is not an image

Target time:

> 0:55

Opening sentence:

> The main challenge is that WiFi CSI does not directly show body parts.

Full script:

> The main challenge is that WiFi CSI does not directly show body parts. Human motion changes wireless reflections through multipath propagation. What the model receives is a noisy CSI sequence, not an RGB image. This signal has low spatial resolution, strong ambiguity, and becomes even harder when multiple people move in the same room.

Key points to remember:

- CSI is indirect.
- Multipath is both useful and noisy.
- Multi-person ambiguity is central to the problem.

Transition:

> Because of this ambiguity, directly predicting the final skeleton in one step can become unstable.

Short version if over time:

> CSI is indirect and noisy. The model must infer 3D pose from wireless reflections, not from visible body parts.

---

## Slide 4 - Research Gap

Title:

> Existing models guess the skeleton in one shot

Target time:

> 0:55

Opening sentence:

> Most baselines treat the problem as direct regression from CSI to final pose.

Full script:

> Most baselines treat the problem as direct regression from CSI to final pose. This is simple, but it can be brittle when the input signal is noisy or when multiple people overlap in the wireless reflections. In these cases, the baseline can produce distorted skeletons, missing people, or unstable pose structures.

Key points to remember:

- Say “can produce,” not “always produces.”
- If showing Figure 5 crop, describe it as selected challenging examples.
- Do not claim M3 or M4 always has fewer false positives than M0.

Transition:

> Our key idea is to replace this one-shot guess with a draft-to-refine process.

Short version if over time:

> Direct regression asks the model to guess the final skeleton immediately. We instead want the model to correct a coarse pose step by step in pose space.

---

## Slide 5 - Key Idea

Title:

> From one-shot guessing to draft-to-refine

Target time:

> 1:05

Opening sentence:

> The core idea is simple: first draft, then refine.

Full script:

> The core idea is simple: first draft, then refine. Instead of asking the model to directly output the final skeleton, we first predict a coarse draft pose, denoted as `X0`. Then the model predicts a learned velocity field `v`, which tells each keypoint how it should move. With one Rectified Flow update, the draft pose is corrected into a refined pose `X1`.

Key points to remember:

- `X0` is the coarse draft pose.
- `v` is the local correction velocity.
- `X1` is the refined pose.
- The animation is conceptual if it is not generated from actual model vectors.

Transition:

> We now embed this idea into the full WiFi pose estimation pipeline.

Short version if over time:

> Instead of CSI to final pose, we use CSI to draft pose, then learn a velocity correction from `X0` to `X1`.

---

## Slide 6 - Proposed Method

Title:

> Proposed M4 pipeline

Target time:

> 1:10

Opening sentence:

> Our final M4 pipeline has three main components.

Full script:

> Our final M4 pipeline has three main components. First, the Spectral Tokenizer converts raw CSI into motion-aware tokens by emphasizing useful temporal and spectral patterns. Second, the WiMamba encoder models the spatio-temporal sequence efficiently. Third, the pose decoder predicts a draft pose and applies one-step Rectified Flow refinement to produce the final 3D pose.

Key points to remember:

- Figure 1 is the main overview.
- Do not go too deep into Figure 3 details here.
- M4 means spectral adapter plus WiMamba plus flow head, without bone loss.

Transition:

> To make the contribution measurable, we evaluate the method through a controlled M0-to-M4 ablation ladder.

Short version if over time:

> M4 combines spectral tokenization, WiMamba sequence modeling, and one-step flow refinement.

---

## Slide 7 - Research Design

Title:

> A controlled M0-M4 ablation ladder

Target time:

> 0:55

Opening sentence:

> The ablation is designed so that each model tests one hypothesis.

Full script:

> The ablation is designed so that each model tests one hypothesis. `M0` is the direct-regression baseline. `M1` adds the spectral input adapter. `M2` adds Mamba-based sequence modeling. `M3` tests the draft-to-refine Rectified Flow head with the Transformer backbone. Finally, `M4` combines the spectral adapter, WiMamba, and Rectified Flow as the efficiency-oriented final model.

Key points to remember:

- M3 isolates the flow benefit for accuracy.
- M4 tests whether flow can be made efficient with WiMamba.
- Use “efficiency-oriented final model,” not “best on everything.”

Transition:

> This design gives us two important results: one about accuracy and one about deployability.

Short version if over time:

> M0 is the baseline, M3 tests the flow idea, and M4 combines flow with Mamba for the best deployment trade-off.

---

## Slide 8 - Quantitative Results

Title:

> Flow improves accuracy. Mamba improves deployability.

Target time:

> 1:20

Opening sentence:

> The numbers show two complementary winners.

Full script:

> The numbers show two complementary winners. `M3` achieves the lowest MPJPE, reducing the error from `169.34 mm` in the baseline to `151.99 mm`. This confirms that draft-to-refine Rectified Flow improves pose accuracy. `M4` is slightly less accurate than `M3`, but it achieves the best efficiency trade-off: `159.85 FPS`, only `5.83M` parameters, and `27.02 MB` peak memory.

Metric emphasis:

- `M3`: best MPJPE.
- `M4`: fastest, lightest, lowest memory among main variants.
- `M0`: baseline.

Key points to remember:

- Do not say `M4` is the most accurate.
- Do not say “state-of-the-art” unless the slide/proposal explicitly supports it against external methods.
- Use “preliminary benchmark” if the context is proposal-level evidence.

Transition:

> The benchmark gives the overall trend, but we also want to see how the skeletons behave visually.

Short version if over time:

> M3 proves the accuracy benefit of flow. M4 proves that this idea can be made fast and lightweight.

---

## Slide 9 - Qualitative Results

Title:

> More coherent matched poses in challenging scenes

Target time:

> 1:10

Opening sentence:

> Qualitatively, we focus on selected challenging samples with one, two, and three people.

Full script:

> Qualitatively, we focus on selected challenging samples with one, two, and three people. The important point is matched pose quality. Compared with the direct-regression baseline, the flow-based models produce more coherent skeletons in these selected cases. This supports the quantitative result that refinement helps the model correct pose structure under ambiguity.

If asked about gray or extra poses:

> The gray poses are unmatched or lower-quality predictions shown for visualization. The official metric is based on Hungarian-matched pose error, so false-positive visualization and matched MPJPE should not be interpreted as the same measurement.

Key points to remember:

- Say “selected challenging samples.”
- Say “matched pose quality.”
- Do not hide the fact that M3 may show duplicate/FP predictions in some videos.

Transition:

> These results suggest a practical direction for privacy-preserving indoor sensing.

Short version if over time:

> In selected challenging cases, the flow-based models produce more coherent matched skeletons than direct regression.

---

## Slide 10 - Impact

Title:

> Toward privacy-preserving indoor human sensing

Target time:

> 0:50

Opening sentence:

> The broader goal is not to replace cameras everywhere.

Full script:

> The broader goal is not to replace cameras everywhere. The goal is to enable pose sensing in places where cameras are not acceptable or not desired. Potential applications include smart homes, elderly care, rehabilitation support, fall monitoring, and privacy-sensitive indoor spaces.

Key points to remember:

- Avoid clinical over-claims.
- Say “potential applications.”
- Keep this slide short and grounded.

Transition:

> Let me close with the main takeaways from our work.

Short version if over time:

> WiFi pose estimation is most valuable in private indoor spaces where visual sensing is not acceptable.

---

## Slide 11 - Conclusion

Title:

> What we learned

Target time:

> 0:50

Opening sentence:

> We asked whether WiFi can estimate 3D human pose without cameras.

Full script:

> We asked whether WiFi can estimate 3D human pose without cameras. We proposed a draft-to-refine Rectified Flow framework with spectral CSI tokenization and WiMamba encoding. Our results show that `M3` gives the strongest accuracy, while `M4` gives the best lightweight deployment trade-off. Overall, the project brings WiFi-based 3D pose estimation closer to real-time, privacy-preserving indoor sensing.

Closing line:

> WiFi can sense posture without seeing identity.

Key points to remember:

- End with the privacy-preserving message.
- Do not end with too many numbers.
- Pause after the closing line.

Short version if over time:

> We show that WiFi pose estimation can move from one-shot guessing to efficient draft-to-refine correction.

---

## Backup Answer Bank

### If asked: Why is M3 more accurate than M4?

Answer:

> `M3` keeps the Transformer backbone and adds the Rectified Flow decoder, so it gives the strongest accuracy in this ablation. `M4` replaces the backbone with WiMamba to reduce computation and memory, so it is designed as the best deployment trade-off rather than the absolute best MPJPE model.

### If asked: Why not choose M3 as the final model?

Answer:

> If the only target is MPJPE, `M3` is the best. But our proposal also values speed, memory, and model size. `M4` is much faster and lighter while preserving most of the accuracy improvement, so it is the more practical final model.

### If asked: Does M4 use bone length loss?

Answer:

> No. In our M0-M4 proposal results, `M4` does not use bone length loss. Bone loss belongs to the later `M5` exploration and should not be used to explain the M4 result.

### If asked: Why does M3 sometimes show more gray predictions in video?

Answer:

> The decoder uses multiple learned pose queries, and the visualization can show duplicate or unmatched predictions. The official MPJPE is computed after Hungarian matching, so matched pose accuracy and false-positive visualization are related but not identical.

### If asked: Is the draft-to-refine animation an actual model output?

Answer:

> It is a conceptual visualization of the update `X1 = X0 + v`. The quantitative and qualitative results come from the actual model outputs; the animation is only used to explain the idea intuitively.

### If asked: Is WiFi fully privacy-safe?

Answer:

> We should not claim that any sensing system is privacy-free. The specific advantage is that WiFi CSI does not capture visual appearance or identity like a camera. Responsible deployment would still require data protection and user consent.

### If asked: Why use Rectified Flow instead of diffusion?

Answer:

> Diffusion-style generation often needs multiple denoising steps from noise. Our setting starts from a conditional draft pose, so Rectified Flow can be used as a direct one-step correction, which is more suitable for efficient pose estimation.

---

## Rehearsal Checklist

Before rehearsal:

- Confirm Slide 6 uses Figure 1.
- Confirm technical backup uses Figure 3.
- Confirm Slide 8 uses Figure 4.
- Confirm Slide 9 uses Figure 5.
- Confirm no slide says `M4` uses bone loss.
- Confirm the hook video is presentation mode, not audit mode.
- Confirm any generated skeleton visual uses keypoint graph style, not anatomical bones.

During rehearsal:

- Slide 1-2 should finish by 1:30.
- Slide 5 should not exceed 1:05.
- Slide 8 can take the longest, but must stay under 1:20.
- If total time exceeds 10:00, cut backup-level explanations first.

Do not cut:

- The problem motivation.
- The draft-to-refine idea.
- The M3 vs M4 distinction.
- The final privacy-preserving message.

Cut first if over time:

- Extra details about M1 and M2.
- Long explanation of WiMamba internals.
- Extra application examples.
- Any formula not needed for the main story.

