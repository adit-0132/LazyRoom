# Port report: ImageToolbox "AI tools" (+ neural-tools library)

Modules: `src/lazyroom/ai/` — `catalog.json` (model catalogue), `models.py`
(catalogue, name resolution, checksum-verified downloads), `runtime.py` (ONNX
Runtime sessions, cancellation, tiler, generic image-to-image pipeline),
`processors.py` (model-specific pre/post-processing), `ocr.py` (PaddleOCR),
`segmentation.py` (subject mask).  Operations: `src/lazyroom/ops/ai_ops.py`;
tools: `src/lazyroom/tools/ai_tools.py`; tests: `tests/test_ai.py`; user
documentation: `docs/MODELS.md`.

## Sources actually read

* `feature/ai-tools/.../domain/model/NeuralModel.kt` (parsed mechanically into the
  catalogue: name, title, type, description string id -> English
  `strings.xml`, download link, size, checksum, speed tier, AI-detector
  contracts), `AiDetection.kt`, `NeuralParams.kt`, `DepthParams.kt`.
* `feature/ai-tools/.../data/`: `AiProcessor.kt`, `AndroidAiToolsRepository.kt`
  (dispatch, session options), `model/{ModelInfo,ImageTensor,TileGrid,TensorSize}.kt`,
  `utils/{AiExtensions,OnnxConversion}.kt`, `DepthProcessor.kt`,
  `StyleTransferProcessor.kt`, `OptimizationStyleTransferProcessor.kt`,
  `AiImageDetectorProcessor.kt`; presentation `AiToolsControls.kt` and
  `DepthParamsSelector.kt` for parameter ranges/defaults.
* `lib/neural-tools/.../bgremover/*` (generic remover, trained sizes, YOLO-seg),
  `inpaint/{LaMaProcessor,WatermarkRemoverProcessor}.kt`, `scans/UvDocUnwarper.kt`,
  `ocr/PaddleOCR.kt`, `OnnxUtils.kt`; `core/data/.../CanvasUtils.kt` (`healAlpha`);
  `feature/filters/.../SpotHealFilter.kt` (LaMa usage).
* The Hugging Face repo tree of `T8RIN/imagetoolbox-models` (sizes + LFS SHA-256
  of all 131 catalogue files matched the Kotlin values exactly; used for the
  checksums of the extra files and of the watermark detector, whose Kotlin
  checksum is empty).

## Status

| Feature (ImageToolbox) | LazyRoom | Status | Notes |
|---|---|---|---|
| `NeuralModel.entries` (131 models, 12 types) | `ai/catalog.json`, `models.list_models()` | faithful | Same names, English titles/descriptions, URLs, sizes, checksums, speed tiers, detector contracts; ordered by type like `sortedBy { type.ordinal }`.  Added: per-model upstream project + weight licence + `restricted`, `task`, and 15 extra files ITB downloads outside the list (LaMa 512 / 512-FAST, 13 PaddleOCR packs). |
| Model download / import / delete | `lazyroom models download NAME --yes`, `models.download_model`, `delete_model` | approximate | SHA-256 verified (ITB only verifies style packs), atomic install, zip packs verified + extracted with a `.complete` marker, dependency download (watermark -> LaMa).  Importing arbitrary user ONNX files ("Imported" models) is **not ported**. |
| `AiProcessor` generic pipeline (upscale, de-JPEG, denoise, artifacts, enhance, anime, scans, colorize) | `runtime.process_image` | faithful | `TileGrid` formula, tile order, edge-extension padding to multiples of 8, SCUNet chunk cap 256 / min-side rules, NAFNet min side 512, fixed-size inputs, `[1,1]` strength control inputs (= strength/100), FP16 I/O, gray (1-channel) models, smooth-step overlap blending of the left/top band (`min` of both ramps), alpha nearest-upscaled, session optimisation levels (`.ort` none, `fbcnn_` extended, `scunet_` none).  Deviations: float pipeline (ITB quantises tiles to 8-bit PNGs); the output scale is taken from the model output (ITB parses `x4`/`4x` from the name — used here only for the size pre-check); tiles are processed sequentially (ITB "parallel workers" not ported; ORT uses intra-op threads); fixed-size non-chunkable models on larger images are resized to the model size and back (ITB would crop). Verified with real Real-ESRGAN x4v3, Zero-DCE++, SPAN anime and synthetic models. |
| Background removers U2Net/U2NetP/ISNet/MODNet/RMBG/InSPyReNet/BiRefNet-tiny (`GenericBackgroundRemover`) | `processors.background_mask`, `ai.remove_background` | faithful | Same trained sizes, bilinear resize, input /255 without normalisation, first output, mask resized back, `healAlpha`.  Verified with real U2NetP only.  Note: upstream U²-Net/rembg normalise with ImageNet mean/std; ITB does not, and neither do we. |
| YOLO11-seg remover (`YoloSegRemover`) | `processors._yolo_mask` | faithful (unverified) | Score 0.45, class-aware NMS 0.5, top 8, prototype masks, 0.35..0.55 alpha ramp.  Not run with the real 249 MB model. |
| `LaMaProcessor` / spot heal | `processors.lama_inpaint`, `ai.inpaint` | approximate | 512x512 whole-image inpainting, mask = value > 0 (nearest), FAST model output x255.  Resampling uses OpenCV Lanczos-4 in sRGB instead of Aire Lanczos3 in LAB.  `keep_unmasked` (default true) composites the result only inside the mask; set it false to get ITB's behaviour (whole image replaced by the upscaled 512 px result).  No mask -> input unchanged (like `SpotHealFilter`).  Verified only with a synthetic model (real LaMa is 208 MB). |
| `WatermarkRemoverProcessor` | `processors.watermark_mask`, `ai.remove_watermark` | faithful (unverified) | ImageNet-normalised 512 px input, threshold 0.18, nearest mask upscaling, then LaMa; same `keep_unmasked` option.  Not run with the real 340 MB detector. |
| `UvDocUnwarper` | `processors.uvdoc_unwarp` (`ai.restore_scans` + `uvdoc_grid.onnx`) | faithful | 496x720 input, align-corners bilinear grid sampling, clamped bilinear source sampling (OpenCV `remap`, 1/32 px sub-pixel precision).  Verified with the real model. |
| `DepthProcessor` inference (DA2 disparity, DA3 direct + confidence background suppression, DA3-mono sky) | `processors.infer_depth` | faithful | Input sizing (14-px alignment, 518/504 max side, letterbox with edge clamp for fixed inputs, 7-px mono inset), output selection, percentile normalisation.  Verified with real Depth Anything V2 Small Q4 only; the DA3 contracts are untested. |
| 13 depth effects + custom depth map | `processors.render_depth_effect`, `depth_from_image`, `ai.depth_map` | faithful | Same formulas, constants, palettes, blur frame (<= 1536 px box blur, nearest sampling), sampler; float maths instead of 8-bit integer rounding. |
| `StyleTransferProcessor` (Arbitrary, MicroAST, AesFA) | `processors.style_transfer`, `ai.style_transfer` | faithful | Same content sizing rule (images <= 2048 px are processed at 500 px), style preparation, embedding interpolation (Arbitrary), padding multiples (4 / 8), ImageNet normalisation (AesFA), output/source mixing by strength.  All three verified with the real packs. |
| `OptimizationStyleTransferProcessor` (VGG19) | `processors.vgg19_style_transfer`, `lbfgs` | faithful (partly verified) | Numpy port of the L-BFGS + strong-Wolfe line search (history 10, 240 steps), working size budget 384², mean-centred 0..255 input.  Ran end-to-end with the real pack for 3 steps (I/O contract: targets, gradient, loss; image changed); a full 240-step run was not executed (~3.5 s per gradient evaluation on this CPU).  The optimiser itself is unit-tested on a quadratic. |
| `AiImageDetectorProcessor` | `processors.detect_ai_generated`, tool `detect_ai_generated` | faithful (unverified with real weights) | Stretch / shortest-side centre crop, per-model mean/std, softmax over AI class indices or sigmoid, verdict thresholds 0.3 / 0.7.  Tested with a synthetic model only. |
| `PaddleOCR` (det/cls/rec, hOCR) | `ai/ocr.py`, tool `ocr_paddle` | faithful | DB detection (960 px, /32 alignment, B,G,R channel order with R,G,B mean/std as upstream), 8-connected components, ITB's own convex hull / min-area rectangle / unclip / box ordering, perspective crop + 90° rotation of tall crops, angle classifier, CTC decoding, low-confidence re-classification, 0.5 confidence filter, hOCR.  Verified with the real PP-OCRv5 English pack (2 lines, 98-99 % confidence, one O/0 confusion). |
| DDColor / ColorizerV2 / other colourisers | `ai.colorize` (generic pipeline, non-chunkable for DDColor) | faithful (unverified) | Same generic path as ITB; no colouriser was downloaded (DDColor files are 220-911 MB). |
| `subject` mask type | `segmentation.subject_mask` | new | Uses the task default or smallest installed background-removal model. |
| Session options | `runtime.load_session` | approximate | Threads = 3/4 of cores, inter-op 4 as in ITB; CPU provider by default, accelerators opt-in via `LAZYROOM_AI_PROVIDER`.  KleidiAI config entries are ARM-only and omitted. |
| Progress notifications, keep-alive service, model selection persistence | — | not ported | Android-specific UI plumbing; cancellation is via `ctx.check()` + `RunOptions.terminate`. |

## Operation mapping

`ai.upscale` (UPSCALE, geometry), `ai.remove_background` (REMOVE_BG),
`ai.colorize` (COLORIZE), `ai.dejpeg` (DE_JPEG, strength for FBCNN),
`ai.denoise` (DENOISE), `ai.remove_artifacts` (ARTIFACTS without the
watermark detector), `ai.enhance` (ENHANCE), `ai.restore_anime` (ANIME),
`ai.restore_scans` (SCANS incl. UVDoc), `ai.style_transfer` (STYLE_TRANSFER),
`ai.depth_map` (DEPTH, 13 effects), `ai.inpaint` (LaMa), `ai.remove_watermark`;
tools `detect_ai_generated` (AI_DETECTION; named without the `ai.` prefix
because tool names are snake_case identifiers) and `ocr_paddle`.

Each op declares `requires=["onnxruntime", "model:@<task>"]` rather than only
`onnxruntime`: the extra virtual requirement makes the op report itself
unavailable until a compatible model is downloaded (otherwise the shared smoke
test, which runs every *available* op with defaults, would fail).  Inside the op
the chosen model is checked and a missing one raises `BACKEND_UNAVAILABLE` with
the exact `lazyroom models download NAME --yes` command.  Consequence:
`ai.depth_map` with a custom depth image still needs one depth model installed.

## Licence research

146 catalogue entries (131 ImageToolbox `NeuralModel` entries + LaMa 512,
LaMa 512 FAST and 13 PaddleOCR packs).  Weight licences found: Apache-2.0 54,
CC-BY-NC-SA-4.0 25, CC-BY-NC-4.0 13, WTFPL 10, MIT 9, BSD-3-Clause 7,
CC-BY-SA-4.0 4, CC-BY-4.0 4, CC0-1.0 1, GPL-3.0 2, AGPL-3.0 1 (YOLO11),
BRIA RMBG-1.4 non-commercial 1, MIT + NVIDIA SegFormer non-commercial 1
(watermark detector), conflicting CC-BY-NC/Apache 1 (DA3-Large-1.1 INT4),
unknown 13.  **57 entries are `restricted`** (41 non-commercial, 13
unverifiable, 3 GPL/AGPL) — 56 of the 131 ImageToolbox models.  Every task
default is a permissive model except `remove_watermark`, whose only model is
restricted.

Findings worth knowing (from the research notes stored in `license_notes`, not
re-verified by hand): the `ai_detector_community_forensics_vit_small_384`
ONNX reportedly contains randomly-initialised weights (0 of 75 checked tensors
match OwensLab/commfor-model-384), so its output is likely meaningless; six
community "1x" models (NMKD Brighten/BrightenRedux/Detoon, Repainter,
ReDetail v2, DeJumbo) could only be traced to an unlicensed mirror; `1x-WB-Denoise`
is a probable (not filename-confirmed) match to loganavter's "1x BW Denoise".

Method: OpenModelDB's data repository (community models), Hugging Face model
cards and GitHub `LICENSE` files (research models), checked October 2026.
Where a licence could not be verified the catalogue says `unknown` and marks the
model restricted.  `restricted` also covers non-commercial (CC-BY-NC*, BRIA,
NVIDIA research) and GPL/AGPL weights.

## Tests

`tests/test_ai.py`: catalogue integrity (131 ITB entries, unique names, types,
checksums, licences, restricted consistency, task defaults), name resolution,
op/tool registration and clean `BACKEND_UNAVAILABLE` reporting with an empty
model directory, `TileGrid` formula/coverage, tiled reconstruction with a
hand-encoded identity ONNX model (random sizes, chunk sizes, overlaps, RGBA),
nearest x2 model == `np.repeat`, overlap band == smooth-step mix of the two
tiles, strength control tensor, FP16 and fixed-size models, cancellation,
LaMa / detector contracts and `ai.inpaint` end-to-end with synthetic models,
checksum-verified atomic downloads and zip packs (`file://` URLs), all 13 depth
effects, custom depth map round trip, L-BFGS port, OCR geometry helpers, and two
`slow` tests with real downloaded models (U2NetP background removal + subject
mask, Real-ESRGAN x4v3 tiled vs single pass) that skip when the models are not
present in `LAZYROOM_AI_TEST_HOME` (default `/tmp/pm_ai_home`).

Run: `LAZYROOM_ITB_ONLY=none .venv/bin/python -m pytest tests/test_ai.py -q -x`.

Additionally verified manually with real downloads (then deleted): PaddleOCR
v5 English, Depth Anything V2 Small Q4 (all 13 effects), MicroAST, Arbitrary
and AesFA style transfer, VGG19 optimisation (3 steps), Zero-DCE++, SPAN anime,
UVDoc.

## Not verified / open points

* Models never run with real weights: all detectors, DDColor/colourisers,
  FBCNN, SCUNet, NAFNet, Restormer, Swin2SR/SwinIR/DAT/RealPLKSR/SPAN upscalers,
  ESRGAN 1x community models, RMBG/ISNet/MODNet/InSPyReNet/BiRefNet/U2Net/YOLO,
  LaMa, watermark detector, Depth Anything 3, PaddleOCR CJK/other packs/v6.
  They share the verified code paths, but model-specific I/O surprises are
  possible (each failure surfaces as `OPERATION_FAILED` with the ORT message).
* The `restricted` flag is conservative and not legal advice.
