# Recipes, provenance, and compatibility

## Historical experiments versus this release

Historical source: [LabelMix commit 814678aed0481ecc2c8046fae6419cd9323e3471](https://github.com/Ejafa/LabelMix/tree/814678aed0481ecc2c8046fae6419cd9323e3471). Its embedded timm version is `1.0.25.dev0`; it is a development fork, not an installable version pin. Its model, dataset, optimizer, training, and experiment sources remain available at that commit.

The fork descends from verified upstream timm commit [370df821](https://github.com/huggingface/pytorch-image-models/commit/370df82133b13cad31458b9525e067011f4bae5c). Against that base, model constructors and optimizers are unchanged; the model checkpoint helper was changed to step-based resume, and the HF dataset reader gained a local Arrow cache path. Neither modification is needed by the release's ImageFolder reader and independent epoch-boundary checkpoint implementation.

This release uses the paper's stable log-domain PL formulation exclusively. The historical code used probability-space subtraction/clamping and processed repeated target entries separately. The new PL requires distinct active labels. Its natural-frequency `unmixed` fallback is explicit and preserves original counts but changes augmentation for repeated-label groups. Existing paper accuracy/calibration numbers are not attributed to this release, and no historical training runs have been repeated here.

Geometry, group rendering, balanced row sampling, natural-mixed index scheduling, and alpha/K schedules were compared against historical functions on controlled inputs. Their controlled outputs match exactly. The new map-style wrapper assigns deterministic per-row/per-group seeds, shuffles views through a rolling index buffer, and uses a bounded group cache. Worker/rank execution and random streams differ from the historical iterable loader; bitwise reproduction of its full training stream is not claimed.

Sparse SCE values and gradients match historical SCE, including repeated labels. Stable PL and mixed loss agree with historical losses on well-conditioned distinct-label float32/float64 inputs. Extreme-logit correctness is checked against independent remaining-set mathematics rather than the historical clamped calculation.

## Selected training configurations

The YAML examples retain the main paper's key ImageNet-1K choices: 110 total epochs, 10 warmup epochs, 256-pixel inputs, global batch 1024, NAdamW, base learning rate 0.0015, cosine decay, and Dirichlet alpha 0.1 to 0.5. Balanced ImageNet recipes use 1280 constructed rows per epoch. SCE/mixed use K=4; PL uses K=6. Mixed beta is 0.5. Seeds 42, 43, and 44 must be run separately for three-seed summaries.

Filtering settings are sourced from `experiments/generate_jobs.py`: minimum side disabled, maximum aspect ratio 20, and 200 rejection attempts. The release samples directly rather than maintaining historical quantized sampling pools. Filtering conditions the weight distribution; unfiltered Dirichlet moment proofs do not describe the filtered sampler exactly. The fallback is uniform weights followed by a valid stripe layout if needed.

The example is a compact reference trainer. Historical runs used a broader timm training pipeline, update-based scheduling, model EMA where configured, and deployment-specific loader settings. These YAMLs are selected release recipes, not a promise of bit-for-bit historical experiment reproduction. Original full recipes and baseline code remain in LabelMix.

The default model is ViT-Wee. The four historical constructor argument sets match their definitions in tested timm 1.0.30:

| Backbone | timm name | Weight decay |
| --- | --- | --- |
| Wee | `vit_wee_patch16_reg1_gap_256` | 0.06 |
| Little | `vit_little_patch16_reg1_gap_256` | 0.06 |
| Medium | `vit_medium_patch16_reg1_gap_256` | 0.06 |
| Betwixt | `vit_betwixt_patch16_reg1_gap_256` | 0.08 |

Use the matching model name and weight decay when adapting a configuration. Identical constructor arguments do not guarantee identical implementation behavior across timm versions. The environment used for release validation is pinned in `requirements-tested.txt`.

## Historical configuration mapping

| Historical option | Release field |
| --- | --- |
| `labelmix_mix_k` | `k` |
| `labelmix_alpha_min/max` | `alpha_min/max` |
| `labelmix_loss=soft_ce/pl_loss/mixed` | `loss=sce/pl/mixed` |
| `labelmix_mixed_alpha` | `beta` |
| `balanced_mode=1280` | `grouping=balanced, cycles=1280` |
| `balanced_mode=natural` with natural-mixed scheduling | `grouping=natural` |
| `labelmix_sampling_max_aspect` | `max_aspect` |

For balanced data, sample count means the constructed class-balanced stream, which may resample original images. For natural data, every provided source is included once; incomplete rows shorter than K pass through unmixed. Incomplete balanced groups are padded from earlier row entries and retain only their primary output count. Partial cyclic orbits do not have the full-orbit balancing guarantee.

## Verification scope

CPU tests cover independent loss values/gradients, repeated labels, padding, extreme logits, half/bfloat16 loss calculations, complete canvas coverage, cyclic target exposure, sample accounting, and matching outputs with zero versus two workers. A two-process Gloo test checks DDP gradients against a single-process global-valid-sample reference. A file-store Gloo launcher also completed the full timm training/validation smoke example on two processes. The timm smoke example exercises forward/backward, unmixed validation, and checkpoint resume.

GPU, NCCL, full ImageNet/COCO training, and historical accuracy reproduction require data/hardware unavailable in this validation environment. Windows PyTorch's torchrun rendezvous can fail when its TCPStore requests unavailable libuv; this is distinct from the passing file-store Gloo gradient check. Availability and measured synthetic loader results are recorded in the release validation record. Synthetic measurements are not ImageNet training speed claims.
