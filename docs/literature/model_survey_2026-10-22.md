# Reference-model survey (Cecilia's request, 2026-10-19 call and follow-up email)

Purpose: pick ONE detector as the fixed reference on which every data strategy (rebalancing, synthetic data) is run.
Cecilia's criteria: peer-reviewed, state of the art, usable code, affordable on our GPUs, and a clear answer to how it
takes single-channel 16-bit radiographs instead of 8-bit RGB photos.

Compute we actually have: the data box, 4 x T4 (16 GB each, sm_75). YOLOv8m on the 143k-tile set takes about
15 min/epoch on 4 GPUs (25 h for 100 epochs, task `4bca601e…`). Every "GPU" figure below is what the authors used,
not what we need; fine-tuning on SWRD is far cheaper than their COCO/Objects365 schedules.

Verification key: **V** = venue confirmed on the proceedings page; **P** = arXiv/preprint only, venue not confirmed;
**?** = claimed by a third party, not checked.

## 1. Candidate detectors

| # | Model | Paper / venue / year | Trained on (by authors) | COCO AP (50-95) | Code quality | Authors' GPUs, schedule | Fits our pipeline? |
|---|---|---|---|---|---|---|---|
| 1 | **YOLOv8m** (current baseline A) | No paper. Ultralytics GitHub release, Jan 2023. Not peer-reviewed. | COCO | 50.2 (m) | Ultralytics `8.4.171`, already pinned in `swrd_paper_baseline/pyproject.toml`. Maintained, huge user base. | Ours: 4 x T4, 100 ep = 25 h, 50 ep = 12 h | Yes, running. Keep as baseline row in every table. |
| 2 | **YOLO11m / YOLO26m** (Ultralytics) | YOLO26: arXiv 2509.25164 (**P**). YOLO11: no paper. Ultralytics itself advises against YOLO12/13 (attention instability, slow CPU). | COCO | YOLO11m 51.5, YOLO26m 53.1 (NMS-free head 52.5) | Same package as row 1, one-line model swap. | Same as row 1 | Yes. Cheapest "newer YOLO" control, but NOT a transformer and not peer-reviewed; weak as the thesis reference. |
| 3 | **RT-DETR** (Baidu) | "DETRs Beat YOLOs on Real-time Object Detection", **CVPR 2024 (V)**, arXiv 2304.08069. RT-DETRv2 arXiv 2407.17140 (**P**). | COCO; O365 variants exist | R50 53.1; Ultralytics `rtdetr-l` 53.0 (HGNetv2-L), `rtdetr-x` 54.8 | Official repo `lyuwenyu/RT-DETR` (PaddlePaddle + PyTorch, 4 GPUs in commands). **Ultralytics ships it**: `RTDETR("rtdetr-l.pt").train(...)` with train/val/export supported. | Official: 4 GPUs. Ultralytics: our 4 x T4; expect slower per epoch than YOLOv8m at 640 (deformable attention, 300 queries), measure one epoch first. `deterministic=False` required. | **Yes, cheapest transformer to test.** Same data yaml, same tiles, same eval script. Peer-reviewed (CVPR). First candidate. |
| 4 | **D-FINE** | "D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement", **ICLR 2025 (V)**, arXiv 2410.13842. | COCO; O365 -> COCO | L 54.0, X 55.8; with O365 pretrain L 57.1, X 59.3 | `Peterande/D-FINE`, Apache-2.0, HGNetv2 backbone, COCO-format custom-dataset guide, O365 checkpoints for custom data. | 4 GPUs in all commands (`--nproc_per_node=4`). COCO epochs not stated on the README. | Needs COCO-json export of our YOLO labels (one script) and its own train loop. ClearML wrapper to write. Moderate. |
| 5 | **DEIM-D-FINE** | "DEIM: DETR with Improved Matching for Fast Convergence", **CVPR 2025 (V)**, arXiv 2412.04234. A training recipe (dense O2O matching + matchability-aware loss) on top of D-FINE / RT-DETRv2. | COCO; O365 -> COCO | DEIM-D-FINE M 52.7 (19 M params), L 54.7 (31 M), X 56.5 (62 M); O365-pretrained X 59.5 after 24 COCO ep | `Intellindust-AI-Lab/DEIM`, Apache-2.0, same codebase style as D-FINE, custom-dataset guide (`remap_mscoco_category: False`), fine-tune with `-t model.pth`. | 4 GPUs in all commands. Converges in fewer epochs than D-FINE (that is the paper's point). | Same cost as row 4; strictly better numbers and faster convergence. **Best peer-reviewed state-of-the-art candidate** if we accept a second codebase. |
| 6 | **DEIMv2** | "Real-Time Object Detection Meets DINOv3", arXiv 2509.20787, Sept 2025 (**P**, no venue found). | COCO | S 50.9 (<10 M params), up to X ~57 | `Intellindust-AI-Lab/DEIMv2`, same lab and layout as DEIM. Backbone = DINOv3 features (distilled for small sizes). | Not checked | Not peer-reviewed yet. Backbone is a ViT pretrained on RGB web images; grayscale needs the patch-embed trick (section 2). Keep as "watch", not reference. |
| 7 | **RF-DETR** (Roboflow + CMU) | "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers", arXiv 2511.09554; project page says ICLR 2026 (**?**, verify on the ICLR site before citing). | COCO, RF100-VL | N..L Apache-2.0: L 56.5; XL/2XL (60.1) under a restrictive licence | `pip install rfdetr`; the product is custom-data fine-tuning (COCO json), single-GPU friendly, DINOv2 ViT backbone + LW-DETR-style decoder. | Fine-tune on 1 GPU is the advertised use. Pretraining not reproducible by us. | Easy to run, strong numbers, but venue unconfirmed and the backbone is a frozen-ish DINOv2 trained on RGB. Optional third arm. |
| 8 | **LW-DETR** (Baidu) | "LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection", arXiv 2406.03459, June 2024 (**P**). | O365 pretrain (CAE v2 MIM + 30 ep supervised) -> COCO 60 ep | tiny 42.6, S 48.0, M 52.5, L 56.1, XL 58.3 | `Atten4Vis/LW-DETR`, Apache-2.0, but validated only on PyTorch 1.13 / CUDA 11.6, no custom-dataset guide. | GPU count not stated. | Superseded by RF-DETR, which reuses its design. Skip. |
| 9 | **DINO** (IDEA) | "DINO: DETR with Improved DeNoising Anchor Boxes", **ICLR 2023 (V)**, arXiv 2203.03605. | COCO; O365 for Swin-L | R50 12 ep 49.0, 36 ep 50.9; Swin-L 36 ep 58.5 | `IDEA-Research/DINO`, Apache-2.0, short custom-dataset note, last news 2023, 150 open issues. | 8 GPUs (batch 16), results batch-size sensitive. | Stale toolchain, heavy, no real-time claim. Skip; cite as the ancestor of rows 4-7. |
| 10 | **Co-DETR / Co-DINO** (SenseTime) | "DETRs with Collaborative Hybrid Assignments Training", **ICCV 2023 (V)**, arXiv 2211.12860. | COCO, O365 | R50 12 ep 52.1; Swin-L 12 ep 59.3; ViT-L 66.0 test-dev (COCO leader 2023) | `Sense-X/Co-DETR`, MIT, bundled MMDetection 2.25 / MMCV 1.5, PyTorch 1.11, CUDA 11.3. | 8 GPUs. ViT-L is 304 M params. | Best absolute AP but an old MMDet stack and 8-GPU recipes. Not for a 4 x T4 thesis. Skip. |

Not listed on purpose: Faster/Mask R-CNN (torchvision) because the weld-radiograph papers already show them below YOLO-class
models, and YOLO-World / Grounding DINO because open-vocabulary text prompts add nothing for six fixed defect classes.

### Shortlist and order of testing

1. **RT-DETR-L via Ultralytics** (row 3): same yaml, same tiles, same `06_eval.py`. If it beats YOLOv8m on the
   film-split val, it is already a peer-reviewed transformer reference and the thesis needs no second codebase.
2. **DEIM-D-FINE-L** (row 5): the current peer-reviewed state of the art at our model size. Costs a COCO-json export
   and a ClearML wrapper. Run it only if (1) is close to YOLO or Cecilia wants a "newest" reference.
3. **RF-DETR-M/L** (row 7): optional, only after the ICLR 2026 claim is verified.

Decision rule (Cecilia, item 11 of the call notes): compare the candidates on (nearly) all training data for a short,
equal schedule, e.g. 20 epochs, and compare the validation curves; a 10-film run is a sanity check only.

## 2. The 16-bit, single-channel question

What is actually different about our data, and what each stack does with it.

**Fact 1: no detector "reads" 16 bits.** Every stack converts the image to float and normalises it before the first
convolution. Bit depth only decides how much dynamic range survives the step before that. Our pipeline already makes
that choice: `03_render.py` does a percentile stretch (p_lo..99.5) per tile, 8-bit, then CLAHE, and writes PNG. The
paper did the same. So the live question is **the window** (which 256 grey levels we keep), not the container.

**Fact 2: one channel vs three is a non-problem for the first layer.** A convolution is linear, so feeding the same
grey image into all three channels gives exactly the same output as a one-channel convolution whose kernel is the sum
of the three RGB kernels. That summed-kernel trick is what `timm` does for `in_chans=1` (`adapt_input_conv`: sums over
the channel axis, no rescaling). Replicating grey to RGB therefore loses nothing; it only costs 3x input memory, which
is negligible at 640 px. The same holds for ViT patch embeddings (a strided conv). What does NOT transfer
automatically is the pretrained feature statistics (ImageNet mean/std, natural-image texture priors). Fine-tuning
handles that; it is the normal domain shift, and the weld-radiograph literature (MADet 2025, EMA-DETR 2026 on
RT-DETR) fine-tunes RGB-pretrained detectors on 8-bit radiographs without special input layers.

**Fact 3: what each codebase does with our files**

| Stack | Loader | 1-channel | 16-bit |
|---|---|---|---|
| Ultralytics (YOLO, RT-DETR) | `cv2.imread` with the colour flag -> uint8 x 3, so a 16-bit PNG is silently truncated to 8 bit | `channels: 1` in the model yaml (added in 8.3.131-8.3.146, `yolo11n-grayscale.pt` exists). Check how the first conv is initialised when 3-channel weights are loaded into a 1-channel model before relying on it. Does not apply to RT-DETR without a test. | Not supported (forum: uint8 only, patch the loader yourself). Keep rendering to 8-bit PNG and treat the window as a hyperparameter. |
| D-FINE / DEIM / DEIMv2 | PIL `Image.open(...).convert("RGB")` in the COCO dataset class, torchvision transforms | Replicate grey -> RGB (Fact 2), zero code | Subclass the dataset to read 16-bit TIFF/PNG, scale to [0,1] float, apply our window there. ~50 lines. |
| RF-DETR | PIL RGB, COCO json | Replicate grey -> RGB | Same subclassing; the DINOv2 backbone expects ImageNet normalisation, so window first, then normalise. |
| DINO / Co-DETR | MMDet / DETR-style COCO loaders, PIL RGB | Replicate | Same, but in an old MMDet stack. |

**What to do.** Keep the 8-bit rendered tiles at **352 px** as the common input for all candidates (352 px adopted
2026-10-22, see `docs/decisions.md`: at least as good as 640 px on test at a third of the cost, porosity unaffected), so
the model comparison changes exactly one thing (the model). At 352 px RT-DETR sees about 2,500 encoder tokens
(44² + 22² + 11²) instead of 8,400, which also shrinks its cost gap to YOLO. Its Ultralytics weights were pretrained at
640 px, so give the RT-DETR run a few warm-up epochs rather than copying the YOLO schedule. Separately, the **window
check** remains open: a 16-bit-aware loader is only worth building if a window sweep shows that the 8-bit stretch is
throwing away defect contrast. That check can be done on YOLOv8m alone.

**For the thesis text.** Cite: `timm` `adapt_input_conv` (channel summation), Ultralytics `channels` docs, and one
remote-sensing or medical example of the same practice (satellite imagery is the usual 12/16-bit precedent).
Phrase it as: "models are not applied out of the box; the adaptation is a windowing step and a trivially equivalent
input layer, and the real transfer cost is the domain gap of the pretrained features."

## 3. Cecilia's three references, skimmed

None of the three is radiographic. Cecilia expected that ("pixel space"), and it is right.

| Reference | What it is | Useful for us |
|---|---|---|
| **Kaputt** (Höfer et al., **ICCV 2025**, arXiv 2510.05903; HF `amazon/kaputt`, CC BY-NC-ND + extra terms, research only) | 238k RGB top-down photos of retail items in logistics trays, 2048 px, 29k defective, 7 defect types, severity labels, item masks. Task = binary "any defect" classification, AP_any. Supervised ViT-S 90.7 AP; anomaly-detection and zero-shot methods <=57 AUROC; adding reference images *hurt*. | (a) Supports supervised detection over anomaly detection when defective training data exists, which is our case. (b) Split by item id = our film-level split; cite it as precedent. (c) Long-tail classes (spillage, missing unit) perform worst, same shape as our undercut problem. (d) High resolution needed for small defects, an argument against the 338 -> 640 upscaling and for a native-resolution check. (e) Majority-vote labels and "ambiguous" cases: a label-noise protocol we can borrow when describing SWRD pseudo-defects. |
| **Severstal steel defects** (Kaggle 2019, datasetninja mirror) | 12,568 train / 5,506 test JPEG strips of 256 x 1600 (8-bit, grey-looking photos stored as JPEG), 4 unexplained defect classes, run-length masks. Class 3: 5,150 images; class 1: 897; class 4: 801; class 2: 247. 63 % of the mirror's images are unlabeled; test labels never published. | Same long-strip geometry and the same two-orders-of-magnitude imbalance as SWRD. Candidate as a second, public, **8-bit** dataset to show that a rebalancing or synthetic-data method is not SWRD-specific (WP5, if time). Licence is Kaggle competition terms, check before redistribution. Not usable as a 16-bit test. |
| **Carvalho, Durupt, Grandvalet, "A Review of Benchmarks for Visual Defect Detection in the Manufacturing Industry"**, arXiv 2305.13261 (2023), HAL hal-03895824 | Reviews DAGM (synthetic grey), NEU (grey, classification), Severstal, KolektorSDD/SDD2, MVTec AD. No X-ray set (no GDXray). Metrics: use precision/recall/F, FPR and **FNR**, not accuracy; AP over AUROC when positives are rare; IoU/Dice for masks, never pixel accuracy. Procedure: train/val/test, test used once, fixed published split. Recommends datasets with a test split, precise typed labels, real production conditions; says none of the reviewed sets meets all three. | Direct justification for three decisions: the ~100-film held-out test set (test used once), reporting per-class recall / false-negative rate next to mAP, and the binary defect-vs-clean evaluation Cecilia asked for (item 9). Also a sentence for the thesis: SWRD *does* meet the review's three criteria (real films, typed polygon labels, and now a fixed film-level split), which is a reason the thesis stays on SWRD. |

## 4. Open items from this survey

- Verify RF-DETR's ICLR 2026 claim on the ICLR site before it enters any table.
- Time one RT-DETR-L epoch at 352 px on the 4 x T4 box at the bm1 batch size; decide the equal-epoch comparison budget.
- Check how Ultralytics initialises the first conv for `channels: 1` with 3-channel pretrained weights (relevant only if
  we drop the grey -> RGB replication).
- Add rows 3, 5, 7 and the three references to `literature_review.xlsx` once read in full (this note is a skim).
- Related work found on the way, for the weld-radiograph related-work section: MADet (*Materials*, Aug 2025, weld
  radiograph detector), EMA-DETR (RT-DETR for girth-weld X-ray, 2026, Chinese petroleum journal), WRT-SAM
  (arXiv 2502.11338, SAM for weld RT segmentation).

## Sources

RT-DETR: https://github.com/lyuwenyu/RT-DETR , https://arxiv.org/abs/2304.08069 , https://ieeexplore.ieee.org/document/10657220 ,
https://arxiv.org/abs/2407.17140 , https://docs.ultralytics.com/models/rtdetr/
D-FINE: https://github.com/Peterande/D-FINE , https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html
DEIM: https://github.com/Intellindust-AI-Lab/DEIM , https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html
DEIMv2: https://arxiv.org/abs/2509.20787
RF-DETR: https://arxiv.org/abs/2511.09554 , https://rfdetr.roboflow.com/latest/ , https://blog.roboflow.com/rf-detr/
LW-DETR: https://arxiv.org/abs/2406.03459 , https://github.com/Atten4Vis/LW-DETR
DINO: https://github.com/IDEA-Research/DINO
Co-DETR: https://github.com/Sense-X/Co-DETR
YOLO26 / YOLO11: https://arxiv.org/abs/2509.25164 , https://docs.ultralytics.com/models/yolo26/ , https://docs.ultralytics.com/compare
Ultralytics grayscale / 8-bit: https://docs.ultralytics.com/datasets/detect/coco8-grayscale , https://community.ultralytics.com/t/new-release-ultralytics-v8-3-146/1107 ,
https://community.ultralytics.com/t/can-yolo-be-trained-using-raw-images/903
timm adapt_input_conv: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_manipulate.py
Kaputt: https://openaccess.thecvf.com/content/ICCV2025/html/Hofer_Kaputt_A_Large-Scale_Dataset_for_Visual_Defect_Detection_ICCV_2025_paper.html ,
https://arxiv.org/abs/2510.05903 , https://huggingface.co/datasets/amazon/kaputt
Severstal: https://datasetninja.com/severstal
Carvalho review: https://arxiv.org/abs/2305.13261 , https://hal-univ-paris8.archives-ouvertes.fr/ROBERVAL-SIPP/hal-03895824v1
Weld RT related work: https://pmc.ncbi.nlm.nih.gov/articles/PMC12348205 , https://yqcy.pipechina.com.cn/en/article/doi/10.6047/j.issn.1000-8241.2026.05.007 , https://arxiv.org/abs/2502.11338
