# Model Runs

This document records completed FastFace model runs and deployment measurements.

## Current Completed Runs

| Run | Backbone | Input | Training Data | Best Epoch | Gender Balanced Acc | Gender Acc | Age MAE | FP32 ONNX | Static INT8 ONNX |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface` | EfficientNetV2-S | 128 | FairFace + UTKFace + IMDB-clean, source-limited train/val, gender-priority loss | 38 | 0.98605 | 0.98605 | 4.84 | 79.5 MB | 22.5 MB |
| `mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | MobileNetV3-Large | 128 | FairFace + UTKFace + IMDB-clean, source-limited train/val, gender-priority loss, IMDB V2-S gender teacher | 24 | 0.97929 | 0.97955 | 5.71 | 13.4 MB | 3.8 MB |
| `mobilenetv3_small112_imdb_facecrop_real_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + IMDB-clean, natural source mix | 24 | 0.96994 | 0.97006 | 5.95 | 4.9 MB | 1.5 MB |
| `mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + IMDB-clean, source-limited train/val, gender-priority loss, IMDB V2-S gender teacher | 35 | 0.96800 | 0.96855 | 6.25 | 4.9 MB | 1.5 MB |
| `mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface` | MobileNetV3-Small | 112 | IMDB-expanded Small112 initialization, fine-tuned on FairFace + UTKFace | 8 | 0.96784 | 0.96756 | 6.21 | 4.9 MB | 1.5 MB |
| `mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + IMDB-clean, source-limited checkpoint selection | 25 | 0.96553 | 0.96633 | 6.28 | 4.9 MB | 1.5 MB |
| `efficientnet_v2_s_128_gender_priority_real_fairface_utkface` | EfficientNetV2-S | 128 | FairFace + UTKFace, gender-priority loss | 34 | 0.94703 | 0.94678 | 5.28 | 79.5 MB | 22.5 MB |
| `efficientnet_v2_s_128_real_fairface_utkface` | EfficientNetV2-S | 128 | FairFace + UTKFace | 22 | 0.94594 | 0.94574 | 4.87 | 79.5 MB | 22.5 MB |
| `efficientnet_b0_128_gender_priority_real_fairface_utkface` | EfficientNet-B0 | 128 | FairFace + UTKFace, gender-priority loss | 37 | 0.93910 | 0.93916 | 5.42 | 18.0 MB | 5.4 MB |
| `efficientnet_b0_128_real_fairface_utkface` | EfficientNet-B0 | 128 | FairFace + UTKFace | 34 | 0.93850 | 0.93834 | 5.26 | 18 MB | 5.4 MB |
| `resnet18_128_real_fairface_utkface` | ResNet18 | 128 | FairFace + UTKFace | 39 | 0.93467 | 0.93490 | 5.18 | 43.8 MB | 11.1 MB |
| `convnext_tiny_128_real_fairface_utkface` | ConvNeXt-Tiny | 128 | FairFace + UTKFace | 32 | 0.93443 | 0.93408 | 5.13 | 107.9 MB | 27.8 MB |
| `mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface` | MobileNetV3-Large | 128 | FairFace + UTKFace + EfficientNetV2-S gender teacher, gender-priority loss | 28 | 0.93376 | 0.93408 | 5.61 | 13.4 MB | 3.8 MB |
| `mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface` | MobileNetV3-Large | 128 | FairFace + UTKFace + EfficientNetV2-S gender teacher | 34 | 0.93230 | 0.93266 | 5.72 | 13.4 MB | 3.8 MB |
| `mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface` | MobileNetV3-Large | 128 | FairFace + UTKFace + EfficientNetV2-S teacher | 24 | 0.93092 | 0.93131 | 5.39 | 13.4 MB | 3.8 MB |
| `mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface` | MobileNetV3-Large | 128 | FairFace + UTKFace + light EfficientNetV2-S teacher | 33 | 0.92994 | 0.92996 | 5.41 | 13.4 MB | 3.8 MB |
| `mobilenetv3_real_fairface_utkface` | MobileNetV3-Large | 128 | FairFace + UTKFace | 34 | 0.92965 | 0.92984 | 5.53 | 14 MB | 3.9 MB |
| `mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface` | MobileNetV3-Large | 112 | FairFace + UTKFace + EfficientNetV2-S gender teacher | 30 | 0.92895 | 0.92892 | 5.72 | 13.4 MB | 3.8 MB |
| `mobilenetv3_large112_real_fairface_utkface` | MobileNetV3-Large | 112 | FairFace + UTKFace | 31 | 0.92828 | 0.92835 | 5.68 | 14 MB | 3.9 MB |
| `mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface` | MobileNetV3-Large | 112 | FairFace + UTKFace + EfficientNetV2-S teacher | 23 | 0.92687 | 0.92660 | 5.46 | 13.4 MB | 3.8 MB |
| `mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + EfficientNet-B0 teacher | 25 | 0.91051 | 0.91061 | 5.90 | 4.9 MB | 1.5 MB |
| `mobilenetv3_small128_real_fairface_utkface` | MobileNetV3-Small | 128 | FairFace + UTKFace | 27 | 0.91028 | 0.91068 | 5.95 | 4.9 MB | 1.5 MB |
| `mobilenetv3_small112_real_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace | 39 | 0.90989 | 0.91049 | 6.03 | 4.9 MB | 1.6 MB |
| `mobilenetv3_small112_distill_large112_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + Large112 teacher | 31 | 0.90745 | 0.90765 | 5.97 | 4.9 MB | 1.6 MB |
| `mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + EfficientNetV2-S teacher | 27 | 0.90711 | 0.90694 | 5.89 | 4.9 MB | 1.5 MB |
| `mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + EfficientNetV2-S gender teacher | 32 | 0.90677 | 0.90679 | 6.04 | 4.9 MB | 1.5 MB |
| `mobilenetv3_small112_distill_light_large112_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + light Large112 teacher | 28 | 0.90622 | 0.90646 | 6.02 | 4.9 MB | 1.6 MB |
| `mobilenetv3_small112_lagenda_real_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + Lagenda-HF, no bbox crop | 31 | 0.87875 | 0.87821 | 6.92 | 4.9 MB | 1.6 MB |
| `mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface` | MobileNetV3-Small | 112 | FairFace + UTKFace + Lagenda-HF, bbox crop margin 0.2 | 34 | 0.90670 | 0.90691 | 6.12 | 4.9 MB | 1.6 MB |
| `swin_t_128_real_fairface_utkface` | Swin-T | 128 | FairFace + UTKFace | 36 | 0.92206 | 0.92204 | 5.53 | 109.3 MB | 29.9 MB |

EfficientNetV2-S gender-priority remains the current public FairFace + UTKFace teacher candidate. The IMDB-expanded EfficientNetV2-S source-balanced run has the highest mixed aggregate score and is a useful IMDB-inclusive teacher/challenger, but it is not the CPU deployment default and does not replace the public-data teacher for FairFace robustness because its FairFace source slice is slightly lower. The IMDB-expanded MobileNetV3-Large distillation run is the strongest completed MobileNetV3 accuracy/CPU tradeoff candidate. The IMDB-expanded MobileNetV3-Small runs keep stronger CPU throughput, but none is promoted as the deployment default because the source-sliced FairFace metric remains modest. MobileNetV3-Small 112 FP32 remains the current high-throughput CPU family.

## Current Active Runs

| Run | Backbone | Input | Training Data | Status |
| --- | --- | ---: | --- | --- |
| None | - | - | - | No active training run is currently recorded after the MobileNetV3-Large IMDB source-balanced gender-distillation run finalized on 2026-07-30. |

The EfficientNetV2-S source-balanced IMDB run is the strongest completed mixed-manifest teacher/challenger. It reaches full-manifest mixed evaluation gender balanced accuracy `0.98605`, driven by `imdb-clean` at `0.99138`, while FairFace lands at `0.94386`. That FairFace slice is slightly lower than the earlier public-data EfficientNetV2-S gender-priority run's FairFace slice `0.94586`, so this checkpoint is useful for IMDB-inclusive pseudo-labeling and age/gender coverage, not as a clean replacement for the public FairFace robustness teacher.

The MobileNetV3-Large source-balanced IMDB gender-distillation run is the strongest completed MobileNetV3 accuracy/CPU tradeoff candidate so far. It reaches mixed evaluation gender balanced accuracy `0.97929`, with FairFace `0.92877`, IMDB-clean `0.98548`, and UTKFace `0.95017`. It is much stronger than the Small112 distillation candidate on accuracy, but tuned FP32 batch-128 throughput is `4,477.7` img/s, roughly 41% of the Small112 distillation run's `10,837.8` img/s and 31% of the original Small112 throughput candidate's `14,483.4` img/s. Use it when gender accuracy matters more than maximum CPU throughput.

The MobileNetV3-Small source-balanced IMDB gender-distillation run is the strongest completed Small-family IMDB-inclusive gender candidate so far. It uses the completed source-balanced IMDB EfficientNetV2-S checkpoint as a low-weight gender-only teacher, reaches mixed evaluation gender balanced accuracy `0.96800`, and improves source-sliced FairFace to `0.90562`, above the earlier Small112 FairFace + UTKFace slice `0.90319` and the source-balanced non-distilled IMDB slice `0.89950`. It is still not the pure throughput winner: tuned FP32 batch-128 throughput is `10,837.8` img/s versus `14,483.4` img/s for the original Small112 FP32 throughput candidate.

The natural IMDB-expanded MobileNetV3-Small run is a useful completed data-expansion result, but not the current deployment default. It reaches mixed evaluation gender balanced accuracy `0.96994`, mostly driven by `imdb-clean` validation at `0.97854`; FairFace lands at only `0.90004`, below the FairFace + UTKFace small baseline. The source-balanced rerun caps IMDB-clean during training and checkpoint-selection validation, but final full-manifest evaluation still shows FairFace at only `0.89950` while IMDB-clean is `0.97347`. Source caps alone therefore do not fix the public FairFace regression. The IMDB-pretrained then FairFace + UTKFace fine-tuned run improves FairFace only to `0.90024`, still below the original non-IMDB Small112 FairFace slice at `0.90319`, so it is also not promoted.

The tested EfficientNetV2-S gender-priority challenger is the best completed public-data teacher candidate so far. It raises evaluation gender balanced accuracy from regular V2-S `0.94594` to `0.94703`, while age MAE worsens from `4.87` to `5.28`. It is not a CPU deployment default: default FP32 batch-128 throughput is only `885.1` img/s, and default static INT8 batch-128 throughput is `841.0` img/s.

The tested EfficientNet-B0 gender-priority challenger is the best completed B0 run for gender. It raises evaluation gender balanced accuracy from `0.93850` to `0.93910`, but age MAE worsens from `5.26` to `5.42`, and tuned FP32 batch-128 throughput stays around `2,513.0` img/s. It is a useful middle accuracy candidate, not the CPU throughput default and not a replacement for EfficientNetV2-S gender-priority as teacher.

The tested MobileNetV3-Small distillation and 128-input runs are not default candidates. The EfficientNet-B0 teacher run slightly improves MobileNetV3-Small gender balanced accuracy from `0.90979` evaluation aggregate to `0.91051`, and improves age MAE from `6.03` to `5.90`, but tuned FP32 batch-128 throughput drops from `14,483.4` to `11,386.4` img/s. The stronger EfficientNetV2-S teacher reduces the student to `0.90711`, likely because its logits over-constrain the small model on this limited data mix. Disabling age distillation for the EfficientNetV2-S Small112 student does not help; the gender-only run lands at `0.90677`. The MobileNetV3-Small 128 run reaches `0.91028`, but tuned FP32 batch-128 throughput is only `10,320.6` img/s. MobileNetV3-Large EfficientNetV2-S teacher runs can improve gender accuracy over some non-distilled baselines; the Large128 gender-priority variant is the strongest completed MobileNetV3-Large gender run, while the Large112 variants do not become defaults. The Large112 standard V2-S distillation run improves age MAE to `5.46` but drops gender balanced accuracy to `0.92687`.

The tested Lagenda-HF expansion is also not a default candidate yet. Directly resizing Lagenda raw images failed because the images are full scenes rather than aligned face crops. Applying manifest `bbox_face` cropping fixes Lagenda validation dramatically, but the combined gender balanced accuracy still lands at `0.90670`, below the FairFace + UTKFace small baseline.

The tested Swin-T 128 transformer challenger is also not a default candidate. It reaches only `0.92206` gender balanced accuracy, below ConvNeXt-Tiny, ResNet18, EfficientNet-B0, EfficientNetV2-S, and the best MobileNetV3-Large variants. Its default FP32 CPU batch-128 throughput is only `105.9` img/s, and static INT8 drops to `29.6` img/s. The full thread sweep was intentionally stopped after `model_fp32_threads1` had run for more than 8 minutes without completing; the default CPU benchmark is already enough to rule it out for CPU deployment.

Age MAE is measured on the mixed public validation split. Treat it as directional only because FairFace contributes age-range labels, while UTKFace contributes exact ages.

## Source-Sliced Validation

| Run | Source | Count | Gender Balanced Acc | Gender Acc | Age MAE | Age CS@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface` | FairFace | 10,954 | 0.94386 | 0.94376 | 5.32 | 0.57395 |
| `efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface` | IMDB-clean | 102,059 | 0.99138 | 0.99135 | 4.79 | 0.63523 |
| `efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface` | UTKFace | 2,425 | 0.95424 | 0.95423 | 4.57 | 0.66969 |
| `mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | FairFace | 10,954 | 0.92877 | 0.92879 | 5.94 | 0.54619 |
| `mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | IMDB-clean | 102,059 | 0.98548 | 0.98568 | 5.70 | 0.55329 |
| `mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | UTKFace | 2,425 | 0.95017 | 0.95052 | 4.96 | 0.63423 |
| `mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | FairFace | 10,954 | 0.90562 | 0.90588 | 6.48 | 0.50858 |
| `mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | IMDB-clean | 102,059 | 0.97542 | 0.97594 | 6.24 | 0.50751 |
| `mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | UTKFace | 2,425 | 0.94101 | 0.94103 | 5.37 | 0.61443 |
| `mobilenetv3_small112_imdb_facecrop_real_fairface_utkface` | FairFace | 10,954 | 0.90004 | 0.90031 | 6.22 | 0.52894 |
| `mobilenetv3_small112_imdb_facecrop_real_fairface_utkface` | IMDB-clean | 102,059 | 0.97854 | 0.97852 | 5.94 | 0.54274 |
| `mobilenetv3_small112_imdb_facecrop_real_fairface_utkface` | UTKFace | 2,425 | 0.92936 | 0.92907 | 5.16 | 0.62887 |
| `mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface` | FairFace | 10,954 | 0.90024 | 0.90004 | 6.15 | 0.53396 |
| `mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface` | IMDB-clean | 102,059 | 0.97605 | 0.97565 | 6.24 | 0.52547 |
| `mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface` | UTKFace | 2,425 | 0.93190 | 0.93196 | 5.11 | 0.64000 |
| `mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface` | FairFace | 10,954 | 0.89950 | 0.90004 | 6.25 | 0.52145 |
| `mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface` | IMDB-clean | 102,059 | 0.97347 | 0.97414 | 6.31 | 0.51614 |
| `mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface` | UTKFace | 2,425 | 0.93607 | 0.93691 | 5.21 | 0.63381 |
| `efficientnet_v2_s_128_gender_priority_real_fairface_utkface` | FairFace | 10,954 | 0.94586 | 0.94559 | 5.42 | 0.57066 |
| `efficientnet_v2_s_128_gender_priority_real_fairface_utkface` | UTKFace | 2,425 | 0.95229 | 0.95216 | 4.63 | 0.66474 |
| `efficientnet_v2_s_128_real_fairface_utkface` | FairFace | 10,954 | 0.94427 | 0.94404 | 4.97 | 0.57413 |
| `efficientnet_v2_s_128_real_fairface_utkface` | UTKFace | 2,425 | 0.95351 | 0.95340 | 4.42 | 0.68330 |
| `efficientnet_b0_128_gender_priority_real_fairface_utkface` | FairFace | 10,954 | 0.93648 | 0.93655 | 5.55 | 0.56637 |
| `efficientnet_b0_128_gender_priority_real_fairface_utkface` | UTKFace | 2,425 | 0.95093 | 0.95093 | 4.82 | 0.64701 |
| `efficientnet_b0_128_real_fairface_utkface` | FairFace | 10,954 | 0.93578 | 0.93555 | 5.39 | 0.55779 |
| `efficientnet_b0_128_real_fairface_utkface` | UTKFace | 2,425 | 0.95079 | 0.95093 | 4.69 | 0.66186 |
| `resnet18_128_real_fairface_utkface` | FairFace | 10,954 | 0.93229 | 0.93254 | 5.32 | 0.55770 |
| `resnet18_128_real_fairface_utkface` | UTKFace | 2,425 | 0.94544 | 0.94557 | 4.58 | 0.66639 |
| `convnext_tiny_128_real_fairface_utkface` | FairFace | 10,954 | 0.93166 | 0.93135 | 5.25 | 0.56436 |
| `convnext_tiny_128_real_fairface_utkface` | UTKFace | 2,425 | 0.94692 | 0.94639 | 4.56 | 0.66722 |
| `swin_t_128_real_fairface_utkface` | FairFace | 10,954 | 0.91832 | 0.91829 | 5.72 | 0.55624 |
| `swin_t_128_real_fairface_utkface` | UTKFace | 2,425 | 0.93891 | 0.93897 | 4.64 | 0.66309 |
| `mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface` | FairFace | 10,954 | 0.93070 | 0.93108 | 5.79 | 0.55039 |
| `mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface` | UTKFace | 2,425 | 0.94753 | 0.94763 | 4.82 | 0.65237 |
| `mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface` | FairFace | 10,954 | 0.92959 | 0.92998 | 5.90 | 0.52474 |
| `mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface` | UTKFace | 2,425 | 0.94452 | 0.94474 | 4.91 | 0.64948 |
| `mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface` | FairFace | 10,954 | 0.92753 | 0.92797 | 5.53 | 0.53898 |
| `mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface` | UTKFace | 2,425 | 0.94622 | 0.94639 | 4.79 | 0.65031 |
| `mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface` | FairFace | 10,954 | 0.92638 | 0.92642 | 5.57 | 0.53779 |
| `mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface` | UTKFace | 2,425 | 0.94602 | 0.94598 | 4.69 | 0.65773 |
| `mobilenetv3_real_fairface_utkface` | FairFace | 10,954 | 0.92724 | 0.92742 | 5.72 | 0.53396 |
| `mobilenetv3_real_fairface_utkface` | UTKFace | 2,425 | 0.94077 | 0.94103 | 4.71 | 0.66722 |
| `mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface` | FairFace | 10,954 | 0.92619 | 0.92615 | 5.91 | 0.52629 |
| `mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface` | UTKFace | 2,425 | 0.94144 | 0.94144 | 4.86 | 0.63835 |
| `mobilenetv3_large112_real_fairface_utkface` | FairFace | 10,954 | 0.92465 | 0.92478 | 5.88 | 0.52501 |
| `mobilenetv3_large112_real_fairface_utkface` | UTKFace | 2,425 | 0.94450 | 0.94433 | 4.75 | 0.64742 |
| `mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface` | FairFace | 10,954 | 0.92410 | 0.92386 | 5.63 | 0.54619 |
| `mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface` | UTKFace | 2,425 | 0.93938 | 0.93897 | 4.70 | 0.66969 |
| `mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface` | FairFace | 10,954 | 0.90365 | 0.90387 | 6.08 | 0.53852 |
| `mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface` | UTKFace | 2,425 | 0.94147 | 0.94103 | 5.07 | 0.63876 |
| `mobilenetv3_small128_real_fairface_utkface` | FairFace | 10,954 | 0.90442 | 0.90497 | 6.12 | 0.52921 |
| `mobilenetv3_small128_real_fairface_utkface` | UTKFace | 2,425 | 0.93671 | 0.93649 | 5.20 | 0.62474 |
| `mobilenetv3_small112_real_fairface_utkface` | FairFace | 10,954 | 0.90319 | 0.90387 | 6.23 | 0.52657 |
| `mobilenetv3_small112_real_fairface_utkface` | UTKFace | 2,425 | 0.93960 | 0.93979 | 5.12 | 0.63134 |
| `mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface` | FairFace | 10,954 | 0.90058 | 0.90049 | 6.08 | 0.53661 |
| `mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface` | UTKFace | 2,425 | 0.93656 | 0.93608 | 5.04 | 0.63052 |
| `mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface` | FairFace | 10,954 | 0.90002 | 0.90013 | 6.23 | 0.52885 |
| `mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface` | UTKFace | 2,425 | 0.93729 | 0.93691 | 5.19 | 0.62474 |
| `mobilenetv3_small112_distill_large112_fairface_utkface` | FairFace | 10,954 | 0.90257 | 0.90278 | 6.14 | 0.53232 |
| `mobilenetv3_small112_distill_large112_fairface_utkface` | UTKFace | 2,425 | 0.92928 | 0.92948 | 5.19 | 0.63876 |
| `mobilenetv3_small112_distill_light_large112_fairface_utkface` | FairFace | 10,954 | 0.90042 | 0.90068 | 6.21 | 0.52912 |
| `mobilenetv3_small112_distill_light_large112_fairface_utkface` | UTKFace | 2,425 | 0.93308 | 0.93320 | 5.13 | 0.62309 |
| `mobilenetv3_small112_lagenda_real_fairface_utkface` | FairFace | 10,954 | 0.89729 | 0.89702 | 6.32 | 0.52109 |
| `mobilenetv3_small112_lagenda_real_fairface_utkface` | Lagenda | 1,282 | 0.62406 | 0.62324 | 15.28 | 0.25429 |
| `mobilenetv3_small112_lagenda_real_fairface_utkface` | UTKFace | 2,425 | 0.92867 | 0.92825 | 5.20 | 0.62845 |
| `mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface` | FairFace | 10,954 | 0.90197 | 0.90232 | 6.19 | 0.53496 |
| `mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface` | Lagenda | 1,282 | 0.88850 | 0.88846 | 7.54 | 0.49454 |
| `mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface` | UTKFace | 2,425 | 0.93735 | 0.93732 | 5.08 | 0.64000 |

## CPU Benchmark

ONNX Runtime `CPUExecutionProvider` on `<remote-gpu-host>`.

Default-thread results:

| Run | Model | Batch 1 | Batch 8 | Batch 32 | Batch 128 |
| --- | --- | ---: | ---: | ---: | ---: |
| `efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface` | FP32 | 44.4 img/s | 282.3 img/s | 638.8 img/s | 1,044.1 img/s |
| `efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface` | Static INT8 | 102.6 img/s | 387.5 img/s | 634.7 img/s | 990.5 img/s |
| `mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | FP32 | 164.0 img/s | 938.1 img/s | 2,705.2 img/s | 4,401.1 img/s |
| `mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | Static INT8 | 385.1 img/s | 693.3 img/s | 963.1 img/s | 1,081.7 img/s |
| `mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | FP32 | 215.4 img/s | 1,042.9 img/s | 4,332.8 img/s | 9,540.9 img/s |
| `mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | Static INT8 | 768.7 img/s | 1,298.1 img/s | 1,882.6 img/s | 2,352.7 img/s |
| `mobilenetv3_small112_imdb_facecrop_real_fairface_utkface` | FP32 | 215.3 img/s | 1,057.9 img/s | 4,357.1 img/s | 9,523.0 img/s |
| `mobilenetv3_small112_imdb_facecrop_real_fairface_utkface` | Static INT8 | 797.7 img/s | 1,302.4 img/s | 1,912.4 img/s | 2,252.2 img/s |
| `mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface` | FP32 | 212.5 img/s | 1,060.8 img/s | 4,417.5 img/s | 9,440.9 img/s |
| `mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface` | Static INT8 | 767.1 img/s | 1,301.6 img/s | 1,892.4 img/s | 2,228.4 img/s |
| `mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface` | FP32 | 215.6 img/s | 1,060.9 img/s | 4,417.4 img/s | 9,505.4 img/s |
| `mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface` | Static INT8 | 772.8 img/s | 1,295.9 img/s | 1,881.2 img/s | 2,249.1 img/s |
| `efficientnet_v2_s_128_gender_priority_real_fairface_utkface` | FP32 | 45.4 img/s | 240.3 img/s | 536.1 img/s | 885.1 img/s |
| `efficientnet_v2_s_128_gender_priority_real_fairface_utkface` | Static INT8 | 101.2 img/s | 381.7 img/s | 637.9 img/s | 841.0 img/s |
| `efficientnet_v2_s_128_real_fairface_utkface` | FP32 | 42.5 img/s | 269.7 img/s | 556.6 img/s | 1,062.4 img/s |
| `efficientnet_v2_s_128_real_fairface_utkface` | Static INT8 | 103.3 img/s | 386.4 img/s | 637.1 img/s | 962.8 img/s |
| `efficientnet_b0_128_gender_priority_real_fairface_utkface` | FP32 | 115.9 img/s | 650.5 img/s | 1,494.2 img/s | 2,402.9 img/s |
| `efficientnet_b0_128_gender_priority_real_fairface_utkface` | Static INT8 | 205.0 img/s | 557.3 img/s | 890.7 img/s | 1,140.0 img/s |
| `efficientnet_b0_128_real_fairface_utkface` | FP32 | 115.8 img/s | 667.3 img/s | 1,487.4 img/s | 2,424.5 img/s |
| `efficientnet_b0_128_real_fairface_utkface` | Static INT8 | 205.9 img/s | 556.0 img/s | 896.2 img/s | 1,049.5 img/s |
| `resnet18_128_real_fairface_utkface` | FP32 | 557.8 img/s | 1,957.0 img/s | 3,037.8 img/s | 4,082.1 img/s |
| `resnet18_128_real_fairface_utkface` | Static INT8 | 607.0 img/s | 1,882.2 img/s | 2,406.6 img/s | 2,679.1 img/s |
| `convnext_tiny_128_real_fairface_utkface` | FP32 | 78.6 img/s | 210.8 img/s | 264.1 img/s | 307.9 img/s |
| `convnext_tiny_128_real_fairface_utkface` | Static INT8 | 74.4 img/s | 151.2 img/s | 175.2 img/s | 219.7 img/s |
| `swin_t_128_real_fairface_utkface` | FP32 | 37.5 img/s | 81.2 img/s | 106.8 img/s | 105.9 img/s |
| `swin_t_128_real_fairface_utkface` | Static INT8 | 21.8 img/s | 29.2 img/s | 30.1 img/s | 29.6 img/s |
| `mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface` | FP32 | 162.3 img/s | 927.9 img/s | 2,681.4 img/s | 4,416.7 img/s |
| `mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface` | Static INT8 | 448.4 img/s | 935.6 img/s | 1,092.4 img/s | 1,341.2 img/s |
| `mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface` | FP32 | 164.5 img/s | 937.1 img/s | 2,952.4 img/s | 4,275.3 img/s |
| `mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface` | Static INT8 | 431.0 img/s | 806.3 img/s | 1,200.4 img/s | 1,362.1 img/s |
| `mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface` | FP32 | 163.8 img/s | 911.4 img/s | 2,631.8 img/s | 4,214.4 img/s |
| `mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface` | Static INT8 | 381.8 img/s | 691.3 img/s | 969.6 img/s | 1,102.6 img/s |
| `mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface` | FP32 | 165.1 img/s | 934.4 img/s | 2,695.8 img/s | 4,286.1 img/s |
| `mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface` | Static INT8 | 383.0 img/s | 691.6 img/s | 963.9 img/s | 1,164.2 img/s |
| `mobilenetv3_real_fairface_utkface` | FP32 | 196.4 img/s | 1,131.2 img/s | 2,986.3 img/s | 4,633.1 img/s |
| `mobilenetv3_real_fairface_utkface` | Static INT8 | 438.5 img/s | 952.5 img/s | 1,315.3 img/s | 1,350.2 img/s |
| `mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface` | FP32 | 207.7 img/s | 1,222.6 img/s | 3,313.0 img/s | 5,251.1 img/s |
| `mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface` | Static INT8 | 437.2 img/s | 783.8 img/s | 1,461.9 img/s | 1,287.4 img/s |
| `mobilenetv3_large112_real_fairface_utkface` | FP32 | 217.6 img/s | 1,179.6 img/s | 3,106.5 img/s | 5,139.9 img/s |
| `mobilenetv3_large112_real_fairface_utkface` | Static INT8 | 509.9 img/s | 966.7 img/s | 1,351.5 img/s | 1,629.2 img/s |
| `mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface` | FP32 | 168.8 img/s | 946.4 img/s | 3,285.2 img/s | 4,980.5 img/s |
| `mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface` | Static INT8 | 433.8 img/s | 777.1 img/s | 1,111.8 img/s | 1,292.4 img/s |
| `mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface` | FP32 | 212.3 img/s | 1,039.3 img/s | 4,342.1 img/s | 9,479.8 img/s |
| `mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface` | Static INT8 | 764.6 img/s | 1,294.7 img/s | 2,017.5 img/s | 2,440.6 img/s |
| `mobilenetv3_small128_real_fairface_utkface` | FP32 | 207.7 img/s | 1,082.5 img/s | 4,676.0 img/s | 8,432.3 img/s |
| `mobilenetv3_small128_real_fairface_utkface` | Static INT8 | 673.1 img/s | 1,168.2 img/s | 1,636.6 img/s | 1,944.7 img/s |
| `mobilenetv3_small112_real_fairface_utkface` | FP32 | 255.5 img/s | 1,369.4 img/s | 5,539.2 img/s | 10,286.6 img/s |
| `mobilenetv3_small112_real_fairface_utkface` | Static INT8 | 897.1 img/s | 1,677.0 img/s | 2,778.8 img/s | 3,097.7 img/s |
| `mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface` | FP32 | 214.6 img/s | 1,057.4 img/s | 4,363.6 img/s | 9,521.9 img/s |
| `mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface` | Static INT8 | 912.9 img/s | 1,570.8 img/s | 2,402.8 img/s | 2,983.9 img/s |
| `mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface` | FP32 | 214.7 img/s | 1,058.1 img/s | 4,363.8 img/s | 9,425.5 img/s |
| `mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface` | Static INT8 | 766.6 img/s | 1,302.3 img/s | 1,878.3 img/s | 2,244.0 img/s |
| `mobilenetv3_small112_distill_large112_fairface_utkface` | FP32 | 267.7 img/s | 1,338.8 img/s | 5,144.0 img/s | 10,003.3 img/s |
| `mobilenetv3_small112_distill_large112_fairface_utkface` | Static INT8 | 1,200.1 img/s | 1,955.7 img/s | 2,261.2 img/s | 2,273.2 img/s |
| `mobilenetv3_small112_distill_light_large112_fairface_utkface` | FP32 | 268.6 img/s | 1,471.4 img/s | 5,407.8 img/s | 10,254.8 img/s |
| `mobilenetv3_small112_distill_light_large112_fairface_utkface` | Static INT8 | 920.3 img/s | 1,595.9 img/s | 2,472.9 img/s | 3,177.1 img/s |
| `mobilenetv3_small112_lagenda_real_fairface_utkface` | FP32 | 268.6 img/s | 1,470.3 img/s | 5,480.0 img/s | 9,530.8 img/s |
| `mobilenetv3_small112_lagenda_real_fairface_utkface` | Static INT8 | 915.7 img/s | 1,593.8 img/s | 2,490.9 img/s | 2,499.2 img/s |
| `mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface` | FP32 | 217.2 img/s | 1,063.2 img/s | 4,436.6 img/s | 9,496.8 img/s |
| `mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface` | Static INT8 | 893.4 img/s | 1,554.8 img/s | 2,422.3 img/s | 3,052.0 img/s |

Best results from intra-op thread sweep with `inter_op_num_threads=1` and sequential execution:

| Run | Model | Batch 1 | Batch 8 | Batch 32 | Batch 128 |
| --- | --- | ---: | ---: | ---: | ---: |
| `mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | FP32 | 527.0 img/s @ 8 threads | 1,859.5 img/s @ 16 threads | 3,452.2 img/s @ 28 threads | 4,477.7 img/s @ 28 threads |
| `mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | Static INT8 | 742.5 img/s @ 8 threads | 1,345.4 img/s @ 28 threads | 1,527.4 img/s @ 28 threads | 1,517.0 img/s @ 16 threads |
| `mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | FP32 | 1,353.7 img/s @ 2 threads | 2,873.5 img/s @ 8 threads | 8,063.9 img/s @ 8 threads | 10,837.8 img/s @ 28 threads |
| `mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface` | Static INT8 | 1,436.0 img/s @ 8 threads | 2,777.1 img/s @ 8 threads | 3,446.4 img/s @ 28 threads | 3,598.3 img/s @ 28 threads |
| `mobilenetv3_small112_imdb_facecrop_real_fairface_utkface` | FP32 | 1,413.5 img/s @ 4 threads | 3,360.9 img/s @ 4 threads | 7,904.4 img/s @ 28 threads | 12,748.3 img/s @ 28 threads |
| `mobilenetv3_small112_imdb_facecrop_real_fairface_utkface` | Static INT8 | 1,382.9 img/s @ 4 threads | 2,546.3 img/s @ 4 threads | 3,152.7 img/s @ 8 threads | 3,332.3 img/s @ 8 threads |
| `mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface` | FP32 | 1,303.7 img/s @ 1 thread | 2,824.2 img/s @ 28 threads | 9,196.2 img/s @ 28 threads | 13,050.6 img/s @ 28 threads |
| `mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface` | Static INT8 | 1,418.7 img/s @ 4 threads | 2,525.5 img/s @ 4 threads | 3,100.2 img/s @ 28 threads | 3,278.9 img/s @ 16 threads |
| `mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface` | FP32 | 1,314.7 img/s @ 1 thread | 3,116.5 img/s @ 16 threads | 8,626.6 img/s @ 16 threads | 12,579.9 img/s @ 28 threads |
| `mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface` | Static INT8 | 1,352.0 img/s @ 8 threads | 2,480.1 img/s @ 8 threads | 3,100.8 img/s @ 8 threads | 3,251.8 img/s @ 16 threads |
| `efficientnet_v2_s_128_real_fairface_utkface` | FP32 | 125.1 img/s @ 28 threads | 427.0 img/s @ 28 threads | 760.1 img/s @ 28 threads | 1,000.6 img/s @ 28 threads |
| `efficientnet_v2_s_128_real_fairface_utkface` | Static INT8 | 181.8 img/s @ 16 threads | 565.5 img/s @ 16 threads | 799.5 img/s @ 28 threads | 1,051.8 img/s @ 56 threads |
| `efficientnet_b0_128_gender_priority_real_fairface_utkface` | FP32 | 500.8 img/s @ 8 threads | 1,374.3 img/s @ 28 threads | 2,013.7 img/s @ 28 threads | 2,513.0 img/s @ 28 threads |
| `efficientnet_b0_128_gender_priority_real_fairface_utkface` | Static INT8 | 373.7 img/s @ 8 threads | 985.7 img/s @ 28 threads | 1,295.1 img/s @ 28 threads | 1,509.3 img/s @ 28 threads |
| `efficientnet_b0_128_real_fairface_utkface` | FP32 | 515.7 img/s @ 8 threads | 1,433.8 img/s @ 16 threads | 1,973.2 img/s @ 28 threads | 2,549.7 img/s @ 56 threads |
| `efficientnet_b0_128_real_fairface_utkface` | Static INT8 | 389.3 img/s @ 8 threads | 832.4 img/s @ 16 threads | 1,326.9 img/s @ 28 threads | 1,533.6 img/s @ 28 threads |
| `resnet18_128_real_fairface_utkface` | FP32 | 936.7 img/s @ 16 threads | 1,760.1 img/s @ 28 threads | 2,284.2 img/s @ 112 threads | 3,129.4 img/s @ 56 threads |
| `resnet18_128_real_fairface_utkface` | Static INT8 | 1,156.1 img/s @ 28 threads | 3,323.3 img/s @ 28 threads | 4,084.8 img/s @ 28 threads | 4,163.2 img/s @ 28 threads |
| `convnext_tiny_128_real_fairface_utkface` | FP32 | 152.7 img/s @ 16 threads | 237.3 img/s @ 28 threads | 277.8 img/s @ 28 threads | 298.4 img/s @ 28 threads |
| `convnext_tiny_128_real_fairface_utkface` | Static INT8 | 128.7 img/s @ 16 threads | 195.2 img/s @ 28 threads | 231.3 img/s @ 28 threads | 228.3 img/s @ 28 threads |
| `mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface` | FP32 | 747.8 img/s @ 8 threads | 2,218.3 img/s @ 8 threads | 3,787.6 img/s @ 28 threads | 4,376.9 img/s @ 28 threads |
| `mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface` | Static INT8 | 578.4 img/s @ 8 threads | 1,102.1 img/s @ 16 threads | 1,366.0 img/s @ 16 threads | 1,505.9 img/s @ 8 threads |
| `mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface` | FP32 | 697.5 img/s @ 4 threads | 1,878.6 img/s @ 8 threads | 3,964.9 img/s @ 28 threads | 4,365.8 img/s @ 28 threads |
| `mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface` | Static INT8 | 611.5 img/s @ 28 threads | 1,444.1 img/s @ 8 threads | 1,353.3 img/s @ 8 threads | 1,489.7 img/s @ 16 threads |
| `mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface` | FP32 | 737.6 img/s @ 4 threads | 1,941.7 img/s @ 16 threads | 3,359.0 img/s @ 16 threads | 4,555.4 img/s @ 28 threads |
| `mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface` | Static INT8 | 610.6 img/s @ 2 threads | 1,038.1 img/s @ 8 threads | 1,364.7 img/s @ 16 threads | 1,515.1 img/s @ 16 threads |
| `mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface` | FP32 | 770.3 img/s @ 8 threads | 2,517.4 img/s @ 16 threads | 3,955.7 img/s @ 16 threads | 4,500.8 img/s @ 28 threads |
| `mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface` | Static INT8 | 705.1 img/s @ 4 threads | 1,380.8 img/s @ 16 threads | 1,624.0 img/s @ 28 threads | 1,468.2 img/s @ 16 threads |
| `mobilenetv3_real_fairface_utkface` | FP32 | 620.2 img/s @ 16 threads | 2,138.1 img/s @ 28 threads | 4,600.4 img/s @ 28 threads | 4,570.7 img/s @ 28 threads |
| `mobilenetv3_real_fairface_utkface` | Static INT8 | 577.7 img/s @ 8 threads | 1,086.9 img/s @ 8 threads | 1,358.4 img/s @ 8 threads | 1,463.2 img/s @ 16 threads |
| `mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface` | FP32 | 856.2 img/s @ 8 threads | 2,502.6 img/s @ 16 threads | 4,585.6 img/s @ 16 threads | 5,345.9 img/s @ 28 threads |
| `mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface` | Static INT8 | 766.7 img/s @ 28 threads | 1,572.5 img/s @ 28 threads | 1,919.5 img/s @ 28 threads | 1,839.8 img/s @ 28 threads |
| `mobilenetv3_large112_real_fairface_utkface` | FP32 | 598.2 img/s @ 8 threads | 2,050.4 img/s @ 8 threads | 3,989.5 img/s @ 16 threads | 5,418.1 img/s @ 56 threads |
| `mobilenetv3_large112_real_fairface_utkface` | Static INT8 | 844.3 img/s @ 8 threads | 1,715.2 img/s @ 8 threads | 2,172.9 img/s @ 8 threads | 1,883.1 img/s @ 8 threads |
| `mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface` | FP32 | 823.5 img/s @ 4 threads | 2,672.7 img/s @ 16 threads | 4,487.6 img/s @ 16 threads | 5,342.3 img/s @ 28 threads |
| `mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface` | Static INT8 | 794.7 img/s @ 16 threads | 1,314.1 img/s @ 16 threads | 1,608.7 img/s @ 8 threads | 1,778.7 img/s @ 16 threads |
| `mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface` | FP32 | 1,460.2 img/s @ 4 threads | 3,326.9 img/s @ 4 threads | 9,130.1 img/s @ 16 threads | 11,386.4 img/s @ 28 threads |
| `mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface` | Static INT8 | 1,408.0 img/s @ 4 threads | 2,811.0 img/s @ 16 threads | 3,615.8 img/s @ 16 threads | 3,483.2 img/s @ 16 threads |
| `mobilenetv3_small128_real_fairface_utkface` | FP32 | 1,258.2 img/s @ 8 threads | 3,554.0 img/s @ 8 threads | 8,392.6 img/s @ 28 threads | 10,320.6 img/s @ 28 threads |
| `mobilenetv3_small128_real_fairface_utkface` | Static INT8 | 1,274.2 img/s @ 8 threads | 2,442.0 img/s @ 8 threads | 2,453.3 img/s @ 28 threads | 2,767.3 img/s @ 28 threads |
| `mobilenetv3_small112_real_fairface_utkface` | FP32 | 1,468.3 img/s @ 4 threads | 3,461.0 img/s @ 4 threads | 11,273.4 img/s @ 16 threads | 14,483.4 img/s @ 16 threads |
| `mobilenetv3_small112_real_fairface_utkface` | Static INT8 | 1,323.6 img/s @ 28 threads | 2,347.9 img/s @ 28 threads | 3,431.4 img/s @ 28 threads | 3,118.0 img/s @ 8 threads |
| `mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface` | FP32 | 1,367.3 img/s @ 8 threads | 3,607.2 img/s @ 8 threads | 11,321.9 img/s @ 16 threads | 12,153.4 img/s @ 28 threads |
| `mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface` | Static INT8 | 1,310.1 img/s @ 2 threads | 2,408.9 img/s @ 28 threads | 2,900.3 img/s @ 16 threads | 3,325.8 img/s @ 16 threads |
| `mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface` | FP32 | 1,351.2 img/s @ 2 threads | 2,587.2 img/s @ 8 threads | 6,961.0 img/s @ 16 threads | 10,611.2 img/s @ 28 threads |
| `mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface` | Static INT8 | 1,327.2 img/s @ 28 threads | 2,364.5 img/s @ 28 threads | 3,223.9 img/s @ 28 threads | 3,239.5 img/s @ 16 threads |
| `mobilenetv3_small112_distill_large112_fairface_utkface` | FP32 | 1,407.2 img/s @ 4 threads | 3,321.3 img/s @ 4 threads | 8,434.1 img/s @ 16 threads | 12,005.3 img/s @ 28 threads |
| `mobilenetv3_small112_distill_large112_fairface_utkface` | Static INT8 | 1,399.7 img/s @ 8 threads | 2,719.1 img/s @ 8 threads | 3,934.3 img/s @ 8 threads | 3,040.7 img/s @ 16 threads |
| `mobilenetv3_small112_distill_light_large112_fairface_utkface` | FP32 | 1,297.2 img/s @ 1 thread | 2,668.3 img/s @ 8 threads | 7,212.9 img/s @ 16 threads | 11,308.5 img/s @ 28 threads |
| `mobilenetv3_small112_distill_light_large112_fairface_utkface` | Static INT8 | 1,368.6 img/s @ 16 threads | 2,803.4 img/s @ 16 threads | 4,097.9 img/s @ 16 threads | 3,145.1 img/s @ 8 threads |
| `mobilenetv3_small112_lagenda_real_fairface_utkface` | FP32 | 1,331.7 img/s @ 1 thread | 3,413.7 img/s @ 16 threads | 9,625.2 img/s @ 16 threads | 12,300.6 img/s @ 28 threads |
| `mobilenetv3_small112_lagenda_real_fairface_utkface` | Static INT8 | 1,315.2 img/s @ 28 threads | 2,539.5 img/s @ 4 threads | 2,816.6 img/s @ 8 threads | 3,271.9 img/s @ 8 threads |
| `mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface` | FP32 | 1,290.2 img/s @ 1 thread | 2,898.1 img/s @ 8 threads | 7,727.2 img/s @ 28 threads | 12,675.4 img/s @ 28 threads |
| `mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface` | Static INT8 | 1,379.9 img/s @ 16 threads | 2,690.8 img/s @ 16 threads | 3,032.9 img/s @ 28 threads | 3,191.3 img/s @ 16 threads |

Thread tuning changes the deployment recommendation:

- `MobileNetV3-Small 112 FP32` is the current CPU throughput winner for all measured batch sizes.
- The source-balanced IMDB gender-distilled `MobileNetV3-Large 128 FP32` run is the strongest completed MobileNetV3 accuracy candidate: FairFace `0.92877`, IMDB-clean `0.98548`, UTKFace `0.95017`, and tuned FP32 batch-128 throughput `4,477.7` img/s.
- The natural IMDB-expanded `MobileNetV3-Small 112 FP32` run keeps the same high-throughput deployment class and reaches `12,748.3` img/s at tuned batch 128, but it is not promoted because FairFace source-sliced validation regresses.
- The source-balanced IMDB-expanded `MobileNetV3-Small 112 FP32` run also stays in the high-throughput class and reaches `12,579.9` img/s at tuned batch 128, but it does not improve FairFace source robustness; FairFace gender balanced accuracy is `0.89950`.
- The IMDB-pretrained FairFace + UTKFace fine-tune reaches `13,050.6` img/s at tuned FP32 batch 128, but still does not restore FairFace source robustness; FairFace gender balanced accuracy is `0.90024`.
- The IMDB V2-S gender-distilled, source-balanced `MobileNetV3-Small 112 FP32` run is the strongest completed Small-family IMDB-inclusive gender candidate so far. It reaches FairFace `0.90562`, IMDB-clean `0.97542`, UTKFace `0.94101`, and tuned FP32 batch-128 throughput `10,837.8` img/s. It improves source robustness versus prior Small IMDB runs but does not beat the original Small112 throughput result.
- The IMDB-expanded `EfficientNetV2-S 128` source-balanced gender-priority run is the strongest mixed-manifest teacher/challenger so far at aggregate `0.98605`, but default FP32 batch-128 throughput is only `1,044.1` img/s and the FairFace slice `0.94386` does not beat the earlier public-data V2-S gender-priority FairFace slice `0.94586`.
- `EfficientNetV2-S 128 gender-priority` is the current public-validation accuracy winner and teacher candidate, but it is much slower than MobileNetV3-Small and MobileNetV3-Large on CPU.
- The full thread sweeps for `EfficientNetV2-S 128 gender-priority` and the IMDB-expanded `EfficientNetV2-S 128` source-balanced gender-priority run were intentionally skipped after default CPU benchmarks and model-card generation, because the regular V2-S run already has a same-architecture full sweep and these slow challengers are not CPU deployment defaults.
- `EfficientNet-B0 128 FP32` remains a useful smaller teacher/accuracy baseline, but it is no longer the top public-validation accuracy run. The gender-priority B0 variant is the best B0 gender result so far (`0.93910`), at the cost of age MAE and with essentially the same CPU throughput class.
- `ResNet18 128` lands between MobileNetV3-Large and EfficientNet-B0 on gender accuracy. Its INT8 throughput is better than EfficientNet-B0 but still far behind MobileNetV3-Small FP32 at high batch.
- `ConvNeXt-Tiny 128` is not a useful deployment or teacher default on the current data mix. It does not beat EfficientNetV2-S, EfficientNet-B0, or ResNet18 on gender accuracy, and its CPU throughput is far below every MobileNet/ResNet deployment candidate.
- `Swin-T 128` is not a useful deployment or teacher default on the current data mix. It lands at only `0.92206` gender balanced accuracy and is much slower than ConvNeXt-Tiny on CPU.
- `MobileNetV3-Large 128` with gender-priority EfficientNetV2-S distillation is the best completed MobileNetV3-Large gender-accuracy variant so far (`0.93376`), while age MAE lands at `5.61` and FP32 throughput remains Large-class.
- The earlier Large128 gender-only EfficientNetV2-S distillation run remains useful but is superseded for the gender-first target: it reaches `0.93230` gender balanced accuracy and `5.72` age MAE.
- The standard EfficientNetV2-S distillation run is the better balanced Large 128 distillation run so far: it reaches `0.93092` gender balanced accuracy and `5.39` age MAE, while the lower-weight variant lands at `0.92994`.
- The EfficientNetV2-S gender-priority teacher challenger is the best public-data gender run so far: it reaches `0.94703` evaluation gender balanced accuracy, improving over regular V2-S `0.94594`, while age MAE regresses from `4.87` to `5.28`.
- The natural IMDB-expanded MobileNetV3-Small run is not a clean default despite its mixed aggregate `0.96994`. It overfits toward the IMDB-clean celebrity domain: FairFace gender balanced accuracy is only `0.90004`, while IMDB-clean is `0.97854`.
- The source-balanced IMDB-expanded MobileNetV3-Small rerun is also not a clean default. Its checkpoint-selection validation used capped IMDB-clean rows, but final full-manifest evaluation still lands at FairFace `0.89950`, IMDB-clean `0.97347`, and UTKFace `0.93607`.
- The IMDB-pretrained FairFace + UTKFace fine-tune is also not a default. It tests whether IMDB initialization plus public-domain fine-tuning can recover the FairFace slice, but lands at FairFace `0.90024`, still below the original non-IMDB Small112 FairFace slice `0.90319`.
- `MobileNetV3-Large 128 FP32` remains the better balanced accuracy/CPU-throughput family than EfficientNet-B0.
- `MobileNetV3-Large 112` with EfficientNetV2-S gender-only distillation slightly improves Large112 gender balanced accuracy (`0.92895` versus `0.92828`) and default FP32 batch-128 throughput (`5,251.1` versus `5,139.9` img/s), but tuned FP32 batch-128 throughput remains effectively similar and it does not beat Large128 gender-priority accuracy.
- `MobileNetV3-Large 112` with standard EfficientNetV2-S distillation is not useful for the gender-first target. It improves age MAE to `5.46`, but drops gender balanced accuracy to `0.92687`.
- `MobileNetV3-Large 112 FP32` is not a clear default. It keeps most of the large accuracy and improves batch-128 throughput, but does not beat large-128 at batch 1, 8, or 32 in the tuned sweep.
- The EfficientNet-B0 teacher distillation run is the best MobileNetV3-Small accuracy variant so far, but the improvement is tiny and FP32 throughput trails the non-distilled small model after tuning.
- The EfficientNetV2-S teacher distillation run is not a default. Despite the stronger teacher, the MobileNetV3-Small student drops to `0.90711` gender balanced accuracy.
- The EfficientNetV2-S gender-only distillation run is also not a MobileNetV3-Small default. It lands at `0.90677`, below both the regular V2-S distillation run and the non-distilled Small112 baseline.
- The lower-weight EfficientNetV2-S distillation run for MobileNetV3-Large 128 is not useful; it lands at `0.92994`, below the standard-weight V2-S distillation run.
- `MobileNetV3-Small 128 FP32` does not justify replacing `MobileNetV3-Small 112 FP32`: it gains only about `0.00039` gender balanced accuracy and loses high-batch tuned throughput.
- The earlier MobileNetV3-Large teacher distillation runs are not defaults. They improve some INT8 measurements, but gender accuracy drops and FP32 throughput trails the non-distilled small model after tuning.
- Lagenda-HF requires bbox cropping. Without it, Lagenda validation collapses; with it, Lagenda becomes usable but still does not improve the primary mixed validation metric.
- Static INT8 is not the default path. It improves some low-latency cases but loses to FP32 for sustained throughput after thread tuning and additional variant tuning.

## Gender Disagreement Review

Use `scripts/compare-gender-models.sh` for a fixed comparison between the current converged candidates, public FairFace-ONNX, and MiVOLO. The default review set is FairFace validation, UTKFace validation, and a seed-stable IMDB-clean validation sample capped to the FairFace validation size.

The original 2026-07-31 public-FairFace review output is `outputs/analysis/gender-comparison-current`. The MiVOLO-inclusive review output is `outputs/analysis/gender-comparison-mivolo-current`. It compares:

- `our_large128_imdb_distill`
- `our_small112_imdb_distill`
- `teacher_v2s_imdb`
- `public_fairface_onnx`
- `mivolo_imdb_face`

On 24,333 selected validation rows, aggregate gender balanced accuracy is `0.96638` for `teacher_v2s_imdb`, `0.95618` for `our_large128_imdb_distill`, `0.94658` for `public_fairface_onnx`, `0.94461` for `mivolo_imdb_face`, and `0.94059` for `our_small112_imdb_distill`. The MiVOLO baseline is the official face-only IMDB-clean age+gender checkpoint, staged at `third_party/mivolo/weights/model_imdb_face_4.22_99.38.pth.tar`.

MiVOLO source-sliced gender balanced accuracy:

| Dataset | Gender Balanced Acc | Gender Acc |
| --- | ---: | ---: |
| FairFace | 0.89753 | 0.89858 |
| IMDB-clean | 0.99614 | 0.99607 |
| UTKFace | 0.92825 | 0.92701 |

Pairwise against the main CPU candidate, `our_large128_imdb_distill` and MiVOLO disagree on 1,532 rows. FastFace Large is correct by public labels on 901 of those rows, and MiVOLO is correct on 631. MiVOLO and public FairFace-ONNX disagree on 1,753 rows.

Important review files:

```text
outputs/analysis/gender-comparison-mivolo-current/summary.json
outputs/analysis/gender-comparison-mivolo-current/predictions.jsonl
outputs/analysis/gender-comparison-mivolo-current/gender_disagreements.csv
outputs/analysis/gender-comparison-mivolo-current/gender_disagreements_top.jpg
outputs/analysis/gender-comparison-mivolo-current/focused/public_vs_our_large.csv
outputs/analysis/gender-comparison-mivolo-current/focused/teacher_vs_our_large.csv
outputs/analysis/gender-comparison-mivolo-current/focused/mivolo_vs_our_large.csv
outputs/analysis/gender-comparison-mivolo-current/focused/mivolo_vs_public_fairface.csv
```

For manual labeling, run `scripts/build-manual-gender-review.sh` after the comparison. The default manual review now uses only `focused/public_vs_our_large.csv`, because rows where every model agrees are not useful for deciding whether FastFace or the public baseline is wrong. The current output is `outputs/analysis/manual-public-gender-review-current`, with `manual_gender_review.xlsx`, `manual_gender_review.csv`, `images/`, and `manual_gender_review_package.zip`. The workbook embeds one face-crop thumbnail per row, replaces `image_path` with `image`, adds explicit `our_large_gender` and `public_fairface_gender` columns, and leaves `manual_gender` blank for human labels. The current public-vs-our review has 1,301 rows.

## INT8 Variant Tuning

Additional static INT8 tuning was run on the current throughput candidate, `mobilenetv3_small112_real_fairface_utkface`, with 1,024 calibration samples and the same thread sweep as the main benchmark.

Best result per model and batch:

| Variant | Batch 1 | Batch 8 | Batch 32 | Batch 128 |
| --- | ---: | ---: | ---: | ---: |
| FP32 tuned baseline | 1,468.3 img/s @ 4 threads | 3,461.0 img/s @ 4 threads | 11,273.4 img/s @ 16 threads | 14,483.4 img/s @ 16 threads |
| Existing static INT8 QDQ U8/S8 per-channel | 1,323.6 img/s @ 28 threads | 2,347.9 img/s @ 28 threads | 3,431.4 img/s @ 28 threads | 3,118.0 img/s @ 8 threads |
| QDQ U8/S8 per-channel | 1,403.5 img/s @ 4 threads | 2,810.4 img/s @ 8 threads | 2,991.1 img/s @ 8 threads | 3,255.6 img/s @ 16 threads |
| QDQ U8/S8 tensor-wise | 1,279.5 img/s @ 8 threads | 2,498.0 img/s @ 4 threads | 3,004.9 img/s @ 28 threads | 3,301.3 img/s @ 8 threads |
| QOperator U8/S8 tensor-wise | 1,093.9 img/s @ 1 thread | 1,741.6 img/s @ 4 threads | 1,707.3 img/s @ 4 threads | 1,602.9 img/s @ 4 threads |
| QDQ S8/S8 per-channel | 762.3 img/s @ 1 thread | 1,051.0 img/s @ 1 thread | 1,179.8 img/s @ 4 threads | 1,244.5 img/s @ 4 threads |
| Preprocessed QDQ U8/S8 per-channel | 1,266.9 img/s @ 2 threads | 2,243.7 img/s @ 4 threads | 3,071.3 img/s @ 8 threads | 3,519.5 img/s @ 28 threads |
| Preprocessed QDQ U8/S8 tensor-wise | 1,308.4 img/s @ 16 threads | 2,443.7 img/s @ 4 threads | 2,883.6 img/s @ 8 threads | 3,476.6 img/s @ 8 threads |
| Preprocessed QOperator U8/S8 tensor-wise | 1,422.4 img/s @ 8 threads | 2,564.3 img/s @ 16 threads | 3,758.9 img/s @ 8 threads | 3,993.4 img/s @ 8 threads |
| Preprocessed QDQ S8/S8 per-channel | 402.8 img/s @ 4 threads | 677.1 img/s @ 28 threads | 834.2 img/s @ 16 threads | 920.7 img/s @ 16 threads |

QOperator per-channel failed in ONNX Runtime quantization for this graph with a per-channel weight broadcast error. Preprocessing improved the QOperator tensor-wise high-batch result, but it still reached only `3,993.4 img/s` at batch 128, far below the FP32 baseline. Keep `MobileNetV3-Small 112 FP32` as the CPU throughput default.

## Artifact Paths

```text
runs/mobilenetv3_real_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_large128_distill_efficientnet_v2_s_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_large128_distill_gender_efficientnet_v2_s_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_large128_distill_gender_priority_efficientnet_v2_s_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_large128_distill_light_efficientnet_v2_s_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_large112_real_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_large112_distill_gender_efficientnet_v2_s_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_large112_distill_efficientnet_v2_s_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_real_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  int8-tuning/cpu-thread-sweep-summary.json
  int8-tuning-preprocessed/cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_imdb_facecrop_real_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_imdb_source_balanced_facecrop_real_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_imdb_pretrain_finetune_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/efficientnet_v2_s_128_imdb_source_balanced_gender_priority_real_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  model_card.md

runs/mobilenetv3_small112_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_large128_imdb_source_balanced_distill_gender_priority_efficientnet_v2_s_imdb_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_distill_efficientnet_b0_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_distill_efficientnet_v2_s_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_distill_gender_efficientnet_v2_s_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small128_real_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_distill_large112_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_distill_light_large112_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_lagenda_real_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/mobilenetv3_small112_lagenda_facecrop_real_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/efficientnet_b0_128_real_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/efficientnet_b0_128_gender_priority_real_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/efficientnet_v2_s_128_gender_priority_real_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  model_card.md

runs/efficientnet_v2_s_128_real_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/resnet18_128_real_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/convnext_tiny_128_real_fairface_utkface/
  best.pt
  last.pt
  config.resolved.yaml
  metrics.jsonl
  evaluation_val.json
  model_fp32.onnx
  model_int8_static.onnx
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  cpu-thread-sweep-summary.json
  model_card.md

runs/swin_t_128_real_fairface_utkface/
  best.pt
  model_fp32.onnx
  model_int8_static.onnx
  evaluation_val.json
  benchmark_fp32_cpu.json
  benchmark_int8_static_cpu.json
  model_card.md
```
