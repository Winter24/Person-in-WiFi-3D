# Phase 1 Presentation Strategy — ResFes 2026

This document finalizes the Phase 1 outputs for the ResFes presentation based on the final proposal [NTN_IT_CT.pdf](D:/Resfes_2026/Person-in-WiFi-3D/docs/paper/NTN_IT_CT.pdf).

## 1. Official presentation title

**Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures**

Recommended subtitle:

**From One-Shot Regression to Draft-to-Refine Rectified Flow**

Use the full title on the opening slide and the subtitle on the same slide or on Slide 5, where the key idea is introduced.

## 2. Three presentation messages

### Message 1 — Problem

Camera-based pose estimation is accurate, but it is not acceptable in many privacy-sensitive indoor spaces. WiFi sensing is attractive because it does not capture visual identity, but WiFi CSI is noisy, indirect, and ambiguous.

### Message 2 — Innovation

Instead of forcing the model to predict the final 3D pose in one shot, the proposed method first generates a coarse draft pose and then refines it through Rectified Flow. This changes WiFi pose estimation from direct guessing into trajectory-based correction.

### Message 3 — Evidence

The ablation results tell a clear story: `M3` is the accuracy winner, while `M4` is the deployment-oriented winner with the strongest overall trade-off in speed, model size, and memory.

## 3. Speaker assignment

This strategy assumes **3 speakers**, because the final proposal lists 3 student authors.

| Speaker | Responsibility | Main goal |
|---|---|---|
| Speaker 1 | Hook, motivation, problem framing | Make the audience care immediately |
| Speaker 2 | Method, Figures 1 and 3, ablation ladder | Show technical depth clearly |
| Speaker 3 | Figures 4 and 5, impact, conclusion, main Q&A close | Convert evidence into value |

Recommended owner mapping:

- **Speaker 1:** Le Ngoc Anh Thu
- **Speaker 2:** Nguyen Vinh Nghi
- **Speaker 3:** Nguyen Ngo Nhat Nam

If the team later decides on 2 speakers, merge Speaker 1 and Speaker 2 into one continuous method-driven opening.

## 4. Language decision

The actual presentation should be **fully in English**, because the competition language requirement is English.

Recommended practice mode:

- Slide text: English only
- Speaker script: English only
- Internal rehearsal discussion: Vietnamese allowed
- Q&A rehearsal: English first, Vietnamese explanation only when clarifying within the team

This keeps delivery competition-ready while still making rehearsals efficient.

## 5. Official 11-slide outline

### Slide 1 — Hook

**Title:** Can WiFi understand human posture without cameras?

Purpose:

- Open with the privacy-sensitive sensing problem
- Use short video or animation hook
- Show project title and team name

### Slide 2 — Motivation

**Title:** Why camera-based sensing is not always acceptable

Purpose:

- Compare camera, wearable, and WiFi sensing
- Establish privacy as the main motivation

### Slide 3 — Challenge

**Title:** WiFi CSI is not an image

Purpose:

- Explain noise, multipath, ambiguity, and multi-person difficulty

### Slide 4 — Research Gap

**Title:** Existing models still predict pose in one shot

Purpose:

- Explain why direct regression is unstable
- Explain why heavy Transformer pipelines hurt deployability

### Slide 5 — Key Idea

**Title:** From one-shot guessing to draft-to-refine

Purpose:

- Introduce the central method idea
- This is the conceptual heart of the talk

### Slide 6 — Proposed Method

**Title:** Proposed M4 Pipeline

Purpose:

- Use **Figure 1** from the final proposal as the main overview
- Explain the five-block story:
  - WiFi CSI
  - Spectral Tokenizer
  - WiMamba Encoder
  - Draft Pose
  - Rectified Flow Refinement

### Slide 7 — Research Design

**Title:** A controlled M0–M4 ablation ladder

Purpose:

- Explain the role of `M0`, `M1`, `M2`, `M3`, and `M4`
- If needed, use **Figure 3** as backup or technical support for the M4 detail

### Slide 8 — Quantitative Results

**Title:** Flow improves accuracy. Mamba improves deployability.

Purpose:

- Use **Figure 4** as the main visual
- Present the reduced M0/M3/M4 table
- Say clearly:
  - `M3` = best MPJPE
  - `M4` = best trade-off

### Slide 9 — Qualitative Results

**Title:** More stable skeletons in crowded scenes

Purpose:

- Use **Figure 5** as the main visual
- Show `Ground Truth / M0 / M3 / M4`
- Emphasize that flow-based variants remain structurally more coherent in difficult scenes

### Slide 10 — Impact

**Title:** Toward privacy-preserving indoor human sensing

Purpose:

- Show applications:
  - smart home
  - elderly care
  - rehabilitation
  - privacy-sensitive indoor monitoring

### Slide 11 — Conclusion

**Title:** What we learned

Purpose:

- Restate the research question
- Restate the proposed idea
- Restate the final evidence:
  - M3 improves accuracy
  - M4 improves deployability

## 6. Figure-to-slide mapping

To keep the talk aligned with the final proposal, lock the mapping below:

- **Figure 1** → Slide 6 main overview visual
- **Figure 3** → Slide 7 technical detail visual or backup method slide
- **Figure 4** → Slide 8 main quantitative visual
- **Figure 5** → Slide 9 main qualitative visual

Figure 2 should be treated as an optional supporting visual for the accuracy-efficiency frontier, not as the main quantitative slide if Figure 4 is already used.

## 7. Phase 1 final outputs

At the end of Phase 1, the team should consider these items fixed:

- Presentation title and subtitle
- Three core messages
- Three-speaker division
- English-only delivery policy
- Official 11-slide outline
- Figure-to-slide mapping for Figures 1, 3, 4, and 5

## 8. Practical note

The presentation should not try to prove that `M4` is universally the best model. It should instead make the more defensible claim already supported by the final proposal: `M3` is the strongest accuracy-focused variant, while `M4` is the final deployment-oriented model because it gives the strongest accuracy-efficiency trade-off.
