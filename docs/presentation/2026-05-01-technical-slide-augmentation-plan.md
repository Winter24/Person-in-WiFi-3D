# Technical Slide Augmentation Plan

Muc tieu cua ban bo sung nay la dua cong thuc, kien truc, va loss vao **main deck** de giai thich ro contribution cua chung ta so voi baseline `M0`, thay vi day phan nay xuong backup nhu plan cu.

## 1. Ket luan nhanh sau khi doi chieu code + proposal

### 1.1. Baseline `M0` trong code

- `M0` runtime hien tai nam o [configs/wifi/petr_wifi.py](D:/Resfes_2026/Person-in-WiFi-3D/configs/wifi/petr_wifi.py), voi:
  - `WifiInputAdapter(mode='linear')`
  - `PETR` detector + `PETRHead`
  - Transformer encoder/decoder theo kieu set prediction
  - loss chinh: `FocalLoss + MSE keypoint + OKS + refine losses`
- Nhanh gon de noi tren slide:
  - `M0 = Linear CSI projection + Transformer + direct pose regression`

### 1.2. Nhanh kiem tra cac contribution moi trong code

- `Spectral Tokenizer` nam o [opera/models/utils/spectral_tokenizer.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/utils/spectral_tokenizer.py):
  - linear projection
  - temporal depthwise conv
  - Doppler profile tu `torch.fft.rfft`
  - frequency gate de loc dynamic motion cues
- `WiMamba` nam o [opera/models/backbones/wimamba.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py):
  - factorized temporal Mamba
  - bidirectional spatial Mamba
  - muc tieu la giam `O(L^2)` cua Transformer xuong linear-time sequence modeling
- `Draft-to-Refine Rectified Flow` nam o [opera/models/dense_heads/wi_tidar_head.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/dense_heads/wi_tidar_head.py) va [opera/models/utils/rectified_flow.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/models/utils/rectified_flow.py):
  - draft pose `X0`
  - velocity network `v_theta`
  - one-step Euler refinement
  - Flow Matching loss

### 1.3. Metrics va evidence ma slide nen dung

- [opera/datasets/wifi_pose.py](D:/Resfes_2026/Person-in-WiFi-3D/opera/datasets/wifi_pose.py) da support:
  - `mpjpe`
  - `mpjpe_1p`, `mpjpe_2p`, `mpjpe_3p`
  - `per_joint_mpjpe`
  - `bone_length_error`
- [tools/analysis/benchmark.py](D:/Resfes_2026/Person-in-WiFi-3D/tools/analysis/benchmark.py) da xuat:
  - `FPS`
  - `params`
  - `peak memory`

### 1.4. Diem can khoa de tranh noi sai

- Proposal va presentation docs deu khoa narrative `M0-M4` la snapshot **khong dung bone loss**.
- Tuy nhien [configs/wifi/wi_tidir_wifi.py](D:/Resfes_2026/Person-in-WiFi-3D/configs/wifi/wi_tidir_wifi.py) hien dang bat `BoneLengthLoss`.
- Vi vay:
  - tren **main deck**: khong goi bien the co bone loss la `M4`
  - neu muon noi bone loss: dua thanh `M5` hoac appendix/backup co ghi ro

## 2. Van de cua plan trinh bay hien tai

### 2.1. Diem manh

- Story line rat ro: problem -> key idea -> method -> ablation -> evidence
- Figure mapping tu proposal da dung
- Claim boundary cho `M3` va `M4` da rat on

### 2.2. Diem yeu can sua

- Cong thuc hien dang bi day xuong backup:
  - baseline architecture
  - Rectified Flow formula
  - Spectral Tokenizer formula
- Slide 6 hien tai dung Figure 1 overview, nhung chua du de tra loi cau hoi:
  - `M0` cu the khac gi?
  - contribution nao den tu representation?
  - contribution nao den tu sequence modeling?
  - contribution nao den tu objective/loss?
- Chua co slide nao tach rieng:
  - `M0` objective
  - `Flow Matching loss`
  - su khac nhau giua `direct regression` va `draft-to-refine`

## 3. Khuyen nghi lon: doi tu 11 slides sang 13 slides

De giai thich contribution ro hon so voi `M0`, nen chuyen tu `11-slide public story` sang `13-slide technical competition story`.

### 3.1. Logic tong quat

- Giu phan hook va motivation ngan gon hon.
- Merge mot phan `challenge` va `research gap`.
- Ke contribution theo dung ladder `M0 -> M1 -> M2 -> M3/M4`.
- Moi stage co **it nhat 1 slide ky thuat**.
- Loss chi co slide rieng o nhung stage co doi objective that su.

## 4. De xuat deck moi

### Slide 1 - Hook

- Gi y nguyen.

### Slide 2 - Motivation

- Gi y nguyen nhung cat gon.

### Slide 3 - Challenge + Research Gap

- Gop Slide 3 va Slide 4 cu.
- Chot 1 thong diep:
  - `CSI is indirect, and one-shot regression is brittle.`

### Slide 4 - Stage 0: Baseline `M0` Architecture

- Tieu de:
  - `M0 baseline: Linear projection + Transformer + direct regression`
- Muc tieu:
  - cho hoi dong thay ro chung ta **khong bat dau tu so 0**
  - xac dinh chinh xac baseline de moi contribution ve sau co diem tua
- Nen dung:
  - Figure 7 appendix proposal hoac ban redraw don gian
- Cong thuc nen dat tren slide:

```text
CSI X -> Linear Adapter -> Transformer Encoder/Decoder -> Pose Y_hat
```

- Cau noi chot:
  - `M0 asks the model to predict the final 3D skeleton in one shot.`

### Slide 5 - Stage 0: Baseline `M0` Loss

- Tieu de:
  - `What M0 optimizes`
- Muc tieu:
  - lam ro baseline la set prediction + regression, khong phai flow
- Cong thuc khuyen nghi:

```text
L_M0 = L_match + lambda_cls L_cls + lambda_kpt L_kpt
     + lambda_oks L_oks + lambda_ref L_refine
```

- Ghi chu nho ben duoi:
  - auxiliary losses ton tai trong implementation, nhung tren slide chi giu phien ban toi thieu de de hieu
- Cau noi chot:
  - `So the baseline learns to regress coordinates directly after Hungarian matching.`

### Slide 6 - Stage 1: Spectral Tokenizer (`M1`)

- Tieu de:
  - `Stage 1: Motion-aware spectral tokenization`
- Muc tieu:
  - giai thich contribution thu nhat so voi `M0`
- Cong thuc nen dua:

```text
X_lin = W_in X
X_time = DWConv1D(X_lin)
D = Mean_c |RFFT(X_lin)|
G = sigma(MLP(D))
X_spec = LN(X_lin + W_c (X_time odot G))
```

- Cau hoi slide nay phai tra loi:
  - tai sao chung ta khong chi dung linear projection nhu `M0`?
- Speaker note:
  - `M1 keeps the PETR-style prediction head, but improves the input representation before sequence modeling.`

### Slide 7 - Stage 2: WiMamba Encoder (`M2`)

- Tieu de:
  - `Stage 2: Replace quadratic attention with factorized WiMamba`
- Muc tieu:
  - lam ro contribution thu hai so voi `M0/M1`
- Cong thuc nen dua:

```text
X^(l+1/2) = X^(l) + Mamba_t(LN(X^(l)))
X^(l+1)   = X^(l+1/2) + BiMamba_s(LN(X^(l+1/2)))
```

- O goc slide:

```text
Transformer: O(L^2 * C)
WiMamba:     O(L * C)
```

- Cau noi chot:
  - `M2 mainly targets deployability, not the full accuracy gain yet.`

### Slide 8 - Stage 3: Draft-to-Refine Flow Head (`M3/M4`)

- Tieu de:
  - `Stage 3: From direct regression to draft-to-refine`
- Muc tieu:
  - day la slide contribution quan trong nhat
- Cong thuc nen dua:

```text
X_t = t X_1 + (1 - t) X_0
v_hat = v_theta(X_t, t, c)
X_1_hat = X_0 + v_hat
```

- Mapping truc giac:
  - `X0`: draft pose
  - `c`: query feature from encoded CSI
  - `v_theta`: learned correction velocity
  - `X1_hat`: refined pose

### Slide 9 - Stage 3: Flow Matching Loss

- Tieu de:
  - `Why flow helps beyond direct regression`
- Cong thuc nen dua:

```text
L_flow = E || v_theta(X_t, t, c) - (X_1 - X_0) ||_2^2
```

- Neu can doi chieu voi `M0`, dat 1 box nho ben phai:

```text
M0: predict final coordinates directly
M3/M4: predict correction trajectory from draft to target
```

- Them 1 dong nho:
  - `For proposal-aligned M4, do not include bone loss on this slide.`

### Slide 10 - Stage Summary: `M0 -> M4`

- Tieu de:
  - `What changes at each stage`
- Dung bang 5 cot:

| Model | Input | Sequence Model | Head | Training Signal | Main Purpose |
|---|---|---|---|---|---|
| M0 | Linear | Transformer | Direct regression | matching + regression | baseline |
| M1 | Spectral | Transformer | Direct regression | same as M0 | representation |
| M2 | Spectral | WiMamba | Direct regression | same as M0 | efficiency |
| M3 | Spectral | Transformer | Draft + Flow | + flow matching | accuracy |
| M4 | Spectral | WiMamba | Draft + Flow | + flow matching | trade-off |

- Day la slide phai lam ro contribution **theo giai doan**, khong phai chi liet ke model.

### Slide 11 - Quantitative Results

- Giu Figure 4.
- Them 2 callout:
  - `M3 = best accuracy`
  - `M4 = best trade-off`
- Neu con cho:
  - them 1 dong `vs M0: -17.35 mm MPJPE for M3, +114.16 FPS for M4`

### Slide 12 - Qualitative Results

- Giu Figure 5.
- Caption nen doi thanh:
  - `Compared with M0, flow-based variants preserve more coherent body structure in difficult scenes.`

### Slide 13 - Conclusion

- Gop impact vao conclusion de tiet kiem thoi gian.
- Chot 3 cau:
  - `M0 is a strong direct-regression baseline.`
  - `Flow is the main accuracy driver.`
  - `WiMamba makes the flow pipeline practical for deployment.`

## 5. Neu bat buoc giu 11 slides

Neu BTC bat buoc deck ngan hon, uu tien cat nhu sau:

- Gop Hook + Motivation thanh 1 slide mo dau
- Gop Impact + Conclusion thanh 1 slide ket
- Giu nguyen 4 slide ky thuat o giua:
  - `M0 architecture`
  - `Spectral`
  - `WiMamba`
  - `Flow + flow loss`

Khong nen cat:

- Slide baseline `M0`
- Slide flow formula
- Slide flow loss

Vi day la 3 diem giup hoi dong thay ro contribution cua chung ta la gi.

## 6. Slide nao nen dua `bone loss`

- Khong dua vao main deck neu main deck dang theo proposal `M0-M4`.
- Dua thanh appendix/backup:
  - `Optional Stage 4+: Bone-length regularization (M5)`
- Cong thuc backup:

```text
L_M5 = L_M4 + lambda_bone L_bone
```

- Speaker wording:
  - `Bone loss is a later structural regularizer, not part of the proposal-aligned M4 benchmark snapshot.`

## 7. Thu tu lam viec de cap nhat slide

1. Keo `M0 architecture` tu appendix/proposal len main deck.
2. Tao moi `M0 loss` slide.
3. Keo `Rectified Flow formula` tu backup len main deck.
4. Tao moi `Flow Matching loss` slide.
5. Chuyen `Figure 3` tu backup thanh support cho Slide 7 hoac Slide 8.
6. Sua Slide 7 cu thanh `stage summary` thay vi chi la bang liet ke model.
7. Chot lai script de moi slide ky thuat chi noi 35-45 giay.

## 8. Mot cau chot cho team

Neu muc tieu la **giai thich contribution de thuyet phuc hoi dong**, deck moi khong nen chi noi `M3 tot hon` va `M4 nhanh hon`. Deck can cho thay:

- `M0` dang hoc cai gi
- `M1/M2` thay doi bieu dien va sequence modeling nhu the nao
- `M3/M4` thay doi objective tu direct regression sang trajectory refinement ra sao

Do la ly do nen dua cong thuc va loss vao **main story**, khong chi de o backup.
