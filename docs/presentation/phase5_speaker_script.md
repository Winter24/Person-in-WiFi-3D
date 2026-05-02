# Phase 5 Speaker Script

Working title:

> Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures

Purpose of this file:

- Convert the revised 13-slide blueprint into a 9:45-10:00 oral script.
- Keep the talk natural enough to present without reading directly from slides.
- Make the technical contribution relative to `M0` easy to follow.
- Prevent over-claims around `M3`, `M4`, false positives, and bone loss.

Core thesis to repeat:

> We turn WiFi-based 3D pose estimation from one-shot guessing into staged, draft-to-refine trajectory refinement.

Important claim boundaries:

- The main deck is a controlled `M0-M4` ablation story.
- `M3` is the best MPJPE model in the current snapshot.
- `M4` is the best overall efficiency trade-off model in the current snapshot.
- The repository can expose an optional bone-loss branch, but the `M0-M4` runs discussed in this deck treat it as off at runtime.
- Visualization false positives are not the same as official matched MPJPE.
- The draft-to-refine animation is conceptual unless explicitly generated from model vectors.

---

## Timing Plan

| Section | Slides | Time |
|---|---|---:|
| Hook + motivation | 1-2 | 1:10 |
| Problem framing | 3-5 | 1:50 |
| Architectural stages | 6-10 | 4:05 |
| Results | 11-12 | 2:10 |
| Conclusion | 13 | 0:35 |
| Total target | 13 slides | 10:00 |

Rehearsal target:

> Aim for 9:30 to 9:40 so the real presentation keeps a safety margin.

---

## Speaker Split Option

If using 2 speakers:

| Speaker | Slides | Role |
|---|---|---|
| Speaker 1 | 1-5 | Motivation, problem, baseline |
| Speaker 2 | 6-13 | Technical upgrades, results, conclusion |

If using 3 speakers:

| Speaker | Slides | Role |
|---|---|---|
| Speaker 1 | 1-4 | Hook, motivation, challenge, baseline setup |
| Speaker 2 | 5-8 | Baseline loss, spectral, WiMamba, flow head |
| Speaker 3 | 9-13 | Flow loss, ablation summary, results, conclusion |

Speaker handoff line from Speaker 1 to Speaker 2:

> We have now defined the baseline clearly, so I will hand over to my teammate to explain how each upgrade changes the model step by step.

Speaker handoff line from Speaker 2 to Speaker 3:

> With the draft-to-refine mechanism defined, my teammate will show how we train it and what the ablation results tell us.

---

## Slide 1 - Hook

Title:

> Can WiFi understand human posture without cameras?

Target time:

> 0:35

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

> 0:35

Opening sentence:

> The motivation is not that cameras are weak; it is that cameras are often inappropriate.

Full script:

> The motivation is not that cameras are weak; it is that cameras are often inappropriate. Cameras are accurate, but they capture appearance and raise privacy concerns. Wearables avoid visual identity, but they require the user to wear and maintain a device. WiFi is different: it is already available in many indoor spaces, and it senses motion without recording visual appearance.

Key points to remember:

- Do not attack cameras too strongly.
- Emphasize where cameras are not acceptable.
- WiFi is privacy-preserving because it does not capture appearance.

Transition:

> However, using WiFi for pose estimation creates a much harder technical problem.

Short version if over time:

> Cameras are accurate but privacy-sensitive. Wearables are inconvenient. WiFi gives us a passive, non-visual sensing signal for indoor spaces.

---

## Slide 3 - Challenge + Research Gap

Title:

> CSI is indirect, and one-shot regression is brittle

Target time:

> 0:45

Opening sentence:

> The main challenge is that WiFi CSI does not directly show body parts.

Full script:

> The main challenge is that WiFi CSI does not directly show body parts. Human motion changes wireless reflections through multipath propagation. What the model receives is a noisy CSI sequence, not an RGB image. This signal has low spatial resolution, strong ambiguity, and becomes even harder when multiple people move in the same room. Because of that, one-shot regression to the final 3D pose is structurally brittle in difficult scenes.

Key points to remember:

- CSI is indirect.
- Multipath is both useful and noisy.
- Multi-person ambiguity is central to the problem.

Transition:

> Before showing our upgrades, we first need to define the baseline that we are trying to improve.

Short version if over time:

> CSI is indirect and noisy, so directly guessing the final skeleton in one step can become unstable.

---

## Slide 4 - Stage 0: Baseline `M0` Architecture

Title:

> `M0` baseline: linear projection + Transformer + direct regression

Target time:

> 0:45

Opening sentence:

> `M0` is not a weak strawman; it is a serious Transformer-based baseline.

Full script:

> `M0` is not a weak strawman; it is a serious Transformer-based baseline derived from the Person-in-WiFi 3D pipeline. Raw CSI is projected linearly, encoded through a Transformer, and decoded by learnable pose queries into final 3D coordinates. The important point is that the model still predicts the final skeleton directly in one shot.

Key points to remember:

- Emphasize fairness to the baseline.
- Mention linear projection, Transformer encoding, and query-based decoding.
- End with the phrase one-shot final pose prediction.

Transition:

> Once the architecture is clear, the next question is what this baseline is actually optimizing during training.

Short version if over time:

> `M0` uses linear CSI projection, a Transformer pose stack, and direct one-shot regression to the final pose.

---

## Slide 5 - Stage 0: Baseline `M0` Loss

Title:

> What `M0` optimizes

Target time:

> 0:40

Opening sentence:

> The baseline objective is still a direct coordinate-learning objective.

Full script:

> The baseline objective is still a direct coordinate-learning objective. After Hungarian matching aligns predicted pose instances with ground-truth persons, the model is supervised through classification, keypoint regression, OKS-style consistency, and refinement losses. In the formula, `L_cls` says whether a query corresponds to a person, `L_kpt` regresses the 3D joint coordinates directly, `L_oks` encourages pose consistency, and `L_refine` improves the final coordinates again in the decoder. The key limitation is that the training target remains the final coordinates themselves rather than a correction trajectory.

Key points to remember:

- Say Hungarian matching clearly.
- Do not over-enumerate every implementation detail.
- The memory sentence is direct coordinates, not correction trajectory.

Transition:

> Our first upgrade keeps the prediction head unchanged, but improves how the CSI signal is represented.

Short version if over time:

> `M0` learns final coordinates directly after matching. It does not learn how a draft pose should be corrected.

---

## Slide 6 - Stage 1: Spectral Tokenizer (`M1`)

Title:

> Stage 1: motion-aware spectral tokenization

Target time:

> 0:45

Opening sentence:

> `M1` asks a simple question: can we feed the model a cleaner motion-aware CSI representation?

Full script:

> `M1` asks a simple question: can we feed the model a cleaner motion-aware CSI representation? We first project CSI linearly, then apply temporal filtering, extract a Doppler-like profile through the frequency domain, and use that profile to gate temporal features. In short, `M1` tries to suppress static noise and emphasize dynamic motion cues before the sequence model even begins.

Key points to remember:

- Say input representation, not architecture overhaul.
- Mention gating from the spectral profile.
- Make it clear that the PETR-style prediction head is still unchanged.

Transition:

> After improving the representation, we ask whether the heavy Transformer is still necessary.

Short version if over time:

> `M1` improves the CSI representation by using spectral gating to emphasize motion and suppress static noise.

---

## Slide 7 - Stage 2: WiMamba Encoder (`M2`)

Title:

> Stage 2: replace quadratic attention with factorized WiMamba

Target time:

> 0:45

Opening sentence:

> `M2` is the efficiency stage of the ablation.

Full script:

> `M2` is the efficiency stage of the ablation. Instead of full quadratic self-attention over the entire token sequence, WiMamba factorizes the modeling into a temporal pass and a bidirectional spatial pass. This reduces the sequence-modeling burden from quadratic attention toward linear-time processing, so `M2` mainly tests whether we can improve deployability without yet changing the prediction objective.

Key points to remember:

- Emphasize temporal pass plus bidirectional spatial pass.
- Say efficiency stage.
- Do not oversell `M2` as the main accuracy result.

Transition:

> But better representation and a lighter encoder still do not solve the core one-shot prediction problem.

Short version if over time:

> `M2` replaces heavy self-attention with factorized WiMamba to make the encoder more efficient.

---

## Slide 8 - Stage 3: Draft-to-Refine Flow Head (`M3/M4`)

Title:

> Stage 3: from direct regression to draft-to-refine

Target time:

> 0:55

Opening sentence:

> This is the main conceptual change of our work.

Full script:

> This is the main conceptual change of our work. Instead of directly predicting the final pose, the model first predicts a coarse draft pose `X0`. We then construct an interpolated pose `X_t`, condition a velocity network on the encoded CSI feature `c`, and learn a correction field that moves the draft toward the refined pose. In the equation, the first line defines the path between draft and target, the second line predicts the correction velocity along that path, and the third line applies a one-step update to produce the final pose estimate.

Key points to remember:

- `X0` is the draft pose.
- `c` is the encoded WiFi-conditioned feature.
- The model learns how the draft should move, not only what the final coordinates should be.

Transition:

> To train that refinement stage properly, we also need a new objective.

Short version if over time:

> We replace one-shot coordinate prediction with draft pose generation plus learned velocity-based correction.

---

## Slide 9 - Stage 3: Flow Matching Loss

Title:

> Why flow helps beyond direct regression

Target time:

> 0:45

Opening sentence:

> The key difference is the training target.

Full script:

> The key difference is the training target. In `M0`, the model is asked to predict the final coordinates directly. In `M3` and `M4`, the model is instead trained to predict the correction velocity from the draft toward the ground truth. In the loss, the target correction is `X_1 - X_0`, while the network predicts `v_theta(X_t, t, c)`. The squared error between these two terms is the flow-matching objective. This is why the flow head changes more than the architecture. It changes what the model learns to do.

Key points to remember:

- Repeat direct coordinates versus correction trajectory.
- This slide explains the learning objective, not diffusion sampling.
- Do not mention bone loss here.

Transition:

> With that objective in place, we can now summarize the full logic of the `M0-M4` ladder.

Short version if over time:

> `M0` predicts final coordinates directly. `M3/M4` predict how a draft pose should move toward the target.

---

## Slide 10 - Stage Summary: `M0-M4`

Title:

> What changes from `M0` to `M4`?

Target time:

> 0:45

Opening sentence:

> The ablation is designed as a controlled evidence ladder.

Full script:

> The ablation is designed as a controlled evidence ladder. `M1` changes the input representation, `M2` changes the sequence model, `M3` changes the prediction objective through flow refinement, and `M4` combines efficient sequence modeling with the flow head. This lets us attribute gains more carefully instead of claiming that one giant black-box model is simply better.

Key points to remember:

- Use controlled ladder language.
- Emphasize interpretability of the ablation.
- Mention that `M4` is the combination model.

Transition:

> With the design logic established, the benchmark results become much easier to interpret.

Short version if over time:

> `M1` tests representation, `M2` tests efficiency, `M3` tests flow accuracy, and `M4` combines flow with WiMamba for the final trade-off.

---

## Slide 11 - Quantitative Results

Title:

> Flow improves accuracy. WiMamba improves deployability.

Target time:

> 1:10

Opening sentence:

> The benchmark shows two complementary winners.

Full script:

> The benchmark shows two complementary winners. `M3` achieves the lowest MPJPE, reducing the error from `169.34 mm` in the baseline to `151.99 mm`. That confirms the accuracy benefit of the flow-based refinement idea. `M4` is slightly less accurate than `M3`, but it achieves the best efficiency trade-off: `159.85 FPS`, only `5.83M` parameters, and `27.02 MB` peak memory. So `M3` is the accuracy winner, while `M4` is the practical deployment winner in this frozen local snapshot.

Metric emphasis:

- `M3`: best MPJPE
- `M4`: fastest, smallest, lowest memory among the main variants
- `M0`: baseline reference

Key points to remember:

- Do not say `M4` is the most accurate.
- Do not say state of the art unless externally validated.
- Use frozen local snapshot language if needed.

Transition:

> The table gives the global pattern, but the qualitative samples show what that pattern looks like in hard scenes.

Short version if over time:

> `M3` proves the accuracy gain of flow. `M4` proves that the same idea can be made fast and lightweight.

---

## Slide 12 - Qualitative Results

Title:

> More coherent matched poses in difficult scenes

Target time:

> 1:00

Opening sentence:

> Qualitatively, we focus on selected challenging cases rather than universal proof.

Full script:

> Qualitatively, we focus on selected challenging cases rather than universal proof. The important reading is matched pose quality. Compared with the direct-regression baseline, the flow-based variants produce more coherent skeletons in these selected one-person, two-person, and three-person cases. This is visually consistent with the benchmark trend that refinement helps the model correct pose structure under ambiguity.

If asked about gray or extra poses:

> The gray poses are unmatched or lower-quality predictions shown for visualization. The official metric is based on Hungarian-matched pose error, so false-positive visualization and matched MPJPE should not be interpreted as the same measurement.

Key points to remember:

- Say selected challenging cases.
- Say matched pose quality.
- Do not hide the fact that query-based decoders can show duplicates in some visualizations.

Transition:

> Let me close with the three takeaways that matter most.

Short version if over time:

> In selected difficult cases, the flow-based variants produce more coherent matched skeletons than direct regression.

---

## Slide 13 - Conclusion + Impact

Title:

> Three takeaways from `M0` to `M4`

Target time:

> 0:35

Opening sentence:

> The contribution is a staged redesign, not just one extra module.

Full script:

> The contribution is a staged redesign, not just one extra module. First, `M0` is a strong direct-regression baseline that defines the problem clearly. Second, Rectified Flow is the main accuracy driver because it changes the learning target from final coordinates to trajectory correction. Third, WiMamba makes that refinement pipeline practical for privacy-sensitive indoor deployment. That is why we view `M4` as the final efficiency-oriented model.

Key points to remember:

- End with the research takeaway first.
- Mention deployment only after the technical point is clear.
- Keep the finish crisp.

Transition:

> Thank you. We are happy to take questions.

Short version if over time:

> `M0` defines the baseline, flow drives accuracy, and WiMamba makes the flow pipeline practical.

---

## Fast Q&A Anchors

### Q1. Why did you put formulas in the main deck?

> Because the contribution is not only empirical. We want the audience to see how the model changes from `M0` to `M4` at the level of representation, sequence modeling, and training objective.

### Q2. Why is `M3` more accurate than `M4`?

> `M3` keeps the heavier Transformer-based flow variant, so it preserves stronger representation capacity and achieves the lowest MPJPE. `M4` trades a small amount of accuracy for much better speed, parameter count, and memory.

### Q3. If `M3` is more accurate, why choose `M4`?

> `M3` is the accuracy upper bound in this snapshot. `M4` is the deployment-oriented model because it gives the strongest overall accuracy-efficiency trade-off while still improving over `M0`.

### Q4. What about bone loss in the repository?

> The repository can expose an optional bone-loss branch, but the `M0-M4` runs discussed in this presentation treat that branch as off at runtime. We intentionally keep this deck focused on representation, sequence modeling, and flow refinement.

### Q5. Do extra gray poses in video contradict the benchmark?

> No. Extra gray poses reflect unmatched or low-confidence query outputs in visualization. The reported benchmark uses Hungarian-matched pose error, so the two should not be interpreted as the same quantity.

---

## Delivery Checklist

- Slide 4 and Slide 5 must clearly define `M0`.
- Slide 8 and Slide 9 must clearly separate flow architecture from flow loss.
- Slide 11 must say `M3` is the accuracy winner and `M4` is the trade-off winner.
- No main slide should imply that bone loss is part of the reported `M0-M4` results.
- Keep the pace tight through the technical middle of the talk.
