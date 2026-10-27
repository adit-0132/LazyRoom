# AI models

LazyRoom's AI operations (category `ai`, plus the `detect_ai_generated` and
`ocr_paddle` tools) run neural networks with [ONNX Runtime](https://onnxruntime.ai)
(`pip install 'lazyroom[ai]'`).  They are a port of ImageToolbox's "AI tools"
and use the **same model files** ImageToolbox downloads from
[`huggingface.co/T8RIN/imagetoolbox-models`](https://huggingface.co/T8RIN/imagetoolbox-models).

**No model weights are bundled with LazyRoom.**  Every model is downloaded only
when a user (or an agent acting for a user) explicitly asks for it, and each
model keeps the licence of its original authors — which is *not* LazyRoom's
Apache-2.0 licence and is often more restrictive.

## Listing and downloading (consent)

```bash
lazyroom models list                                  # name, task, licence, restricted?, size, available
lazyroom models download u2netp --yes                 # explicit confirmation is required
lazyroom models download @upscale --yes               # @<task> = the recommended default for a task
```

* Without `--yes` the CLI refuses with `APPROVAL_REQUIRED`.  Names are matched
  exactly, case-insensitively or without the extension (`u2netp` ->
  `u2netp.onnx`); `@<task>` aliases resolve to the task default listed below.
* The file is streamed from the catalogue URL into a temporary `.part-<pid>`
  file, its **SHA-256 is verified** against the catalogue (the checksums
  hard-coded in ImageToolbox's `NeuralModel.kt`; for the watermark detector,
  whose Kotlin checksum is empty, and for the LaMa / PaddleOCR files, the
  SHA-256 published by Hugging Face's LFS metadata), and only then moved into
  place with an atomic rename.  A mismatch raises `DECODE_FAILED` and installs
  nothing.  Free disk space is checked first.
* Zip packs (style transfer, PaddleOCR) are verified, extracted to a temporary
  directory, the inner files are checked (sizes + SHA-256 from ImageToolbox for
  the style packs), and the directory is renamed into place with a `.complete`
  marker.  Archive members are flattened (no path traversal) and capped at 4 GB.
* Models that need another model download it in the same confirmed step
  (`watermark_mit_b5_sigmoid.onnx` also fetches `LaMa_512.onnx`).
* Downloading a **restricted** model works, but the result carries a `warning`
  with its licence.  You are responsible for complying with each model's
  licence; this page is a best-effort summary, not legal advice.
* Files live in `<data dir>/models` — `$LAZYROOM_HOME/data/models` when
  `LAZYROOM_HOME` is set, otherwise `~/.local/share/lazyroom/models` (Linux)
  or `%LOCALAPPDATA%\LazyRoom\models` (Windows).  Each download gets a
  `<file>.meta.json` provenance sidecar.  Delete the file (or call
  `lazyroom.ai.models.delete_model(name)`) to remove a model.

### How operations behave without models

Each AI operation lists its compatible models in a `model` choice.  Until at
least one compatible model is downloaded the operation reports
`available: false` with an install hint (`requires` contains `onnxruntime` and
`model:@<task>`).  Running it with a model that is not downloaded fails with
`BACKEND_UNAVAILABLE`; `details.command` / `hint` contain the exact command,
e.g. `lazyroom models download RealESRGAN-x4v3.ort --yes`, plus the licence
and download size.

## Runtime

* **CPU by default.**  `LAZYROOM_AI_PROVIDER=auto|cuda|directml|coreml` opts
  into an accelerator; it is used only if `onnxruntime` reports that provider as
  available (e.g. `onnxruntime-gpu` / `onnxruntime-directml` installed), otherwise
  CPU is used.  No accelerator is ever required.
* Large images are processed in tiles exactly like ImageToolbox (`chunk_size`
  default 512, `overlap` 16, smooth-step blending of the overlap); models that
  ImageToolbox marks non-chunkable (background removal, DDColor, depth, style
  transfer, UVDoc, watermark, detectors) run in one pass at their own input size.
* Limits: `LAZYROOM_AI_MAX_OUTPUT_PIXELS` (default 200 MP) for upscaled
  outputs, `LAZYROOM_AI_MAX_UNTILED_PIXELS` (default 16 MP) for single-pass
  inference of dynamic-size models, `LAZYROOM_AI_MAX_SESSIONS` (default 2)
  cached sessions.  Inference is cancellable and honours render timeouts (the
  native run is terminated).
* Speed: the catalogue keeps ImageToolbox's speed tier (`VeryFast` ... `VerySlow`).
  Measured on the development machine (12-thread x86 CPU, CPU provider, warm
  session, machine shared with other jobs): U2NetP background removal on 512x512
  ≈ 0.9 s; Real-ESRGAN x4v3 on 160x120 ≈ 1.3 s and on 512x512 (-> 2048x2048,
  tiled) ≈ 10 s; Depth Anything V2 Small Q4 + effect on 512x512 ≈ 1.3 s;
  PaddleOCR (en) on a 650x250 text image ≈ 2.2 s.  The VGG19 optimisation style
  transfer needs ≈ 3-4 s per gradient evaluation at its 384x384 working size,
  i.e. 15+ minutes for its 240 L-BFGS steps.

## Licences and the `restricted` flag

The licence column is the licence of the **weights**, researched per model
(October 2026) from the original authors' model cards, `LICENSE` files and the
[OpenModelDB](https://openmodeldb.info) database; the link points to where the
statement was found.  When no licence could be verified the entry says
`unknown`.  `restricted: true` means the weights are under a non-commercial or
research-only licence, a strong copyleft licence (GPL/AGPL), or a licence that
could not be verified — do not use them for commercial work without checking.
Attribution licences (CC-BY, CC-BY-SA, BSD, MIT, Apache) are not restricted but
still require attribution when you redistribute results or weights as their
terms say.  The full metadata (URL, SHA-256, size, speed, upstream URL, licence
source and notes) is in `src/lazyroom/ai/catalog.json`.

## Models

<!-- TABLE START -->
<!-- generated from src/lazyroom/ai/catalog.json — 146 models, 57 restricted -->

### upscale — `ai.upscale` (20 models; default `RealESRGAN-x4v3.ort`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `RealESRGAN-x4v3.ort` | RealESRGAN x4v3 | 2.6 | [Real-ESRGAN (xinntao)](https://github.com/xinntao/Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | no |
| `RealESRGAN_x2plus.ort` | RealESRGAN x2 Plus | 35 | [Real-ESRGAN (xinntao)](https://github.com/xinntao/Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | no |
| `RealESRGAN_x4plus.ort` | RealESRGAN x4 Plus | 35 | [Real-ESRGAN (xinntao)](https://github.com/xinntao/Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | no |
| `RealESRGAN_x4plus_anime_6B.ort` | RealESRGAN x4 Plus Anime 6B | 9.5 | [Real-ESRGAN (xinntao)](https://github.com/xinntao/Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | no |
| `RealESRNet_x4plus.ort` | RealESRNet x4 Plus | 35 | [Real-ESRGAN (xinntao)](https://github.com/xinntao/Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | no |
| `RealESRGAN_x4plus_anime_4B32F.ort` | RealESRGAN x4 Plus Anime 4B | 5.2 | [lightweight-real-ESRGAN-anime (xiongjie / xiong-jie-y), third-party Real-ESRGAN anime variant (4 blocks, 32 feats)](https://huggingface.co/xiongjie/lightweight-real-ESRGAN-anime) | [MIT](https://huggingface.co/xiongjie/lightweight-real-ESRGAN-anime/blob/main/README.md) | no |
| `x4-UltraSharpV2_fp32_op17.ort` | UltraSharp x4 V2 | 81 | [UltraSharpV2 by kim2091 (OpenModelDB 4x-UltraSharpV2)](https://openmodeldb.info/models/4x-UltraSharpV2) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/4x-UltraSharpV2.json) | **yes** — non-commercial / research-only licence |
| `x4-UltraSharpV2_Lite_fp16_op17.ort` | UltraSharp x4 V2 Lite | 16 | [4x-UltraSharpV2_Lite by Kim2091](https://huggingface.co/Kim2091/UltraSharpV2) | [CC-BY-NC-SA-4.0](https://huggingface.co/Kim2091/UltraSharpV2) | **yes** — non-commercial / research-only licence |
| `2xLiveActionV1_SPAN_490000.onnx` | LiveAction V1 SPAN x2 | 1.7 | [2xLiveActionV1_SPAN by jcj83429 (OpenModelDB 2x-LiveActionV1-SPAN)](https://openmodeldb.info/models/2x-LiveActionV1-SPAN) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/2x-LiveActionV1-SPAN.json) | **yes** — non-commercial / research-only licence |
| `2xVHS2HD-RealPLKSR.onnx` | VHS to HD x2 | 30 | [VHS2HD by asterixcool (OpenModelDB 2x-VHS2HD)](https://openmodeldb.info/models/2x-VHS2HD) | [CC-BY-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/2x-VHS2HD.json) | no |
| `2x_Text2HD_v.1-RealPLKSR.onnx` | Text2HD x2 | 30 | [Text2HD v.1 by asterixcool (OpenModelDB 2x-Text2HD-v-1)](https://openmodeldb.info/models/2x-Text2HD-v-1) | [CC-BY-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/2x-Text2HD-v-1.json) | no |
| `4x-FrankendataPretrainer_SRFormer400K.onnx` | SRFormer x4 | 121 | [FrankendataPretainer_SRFormer by crustaceous-d (OpenModelDB 4x-FrankendataPretainer-SRFormer)](https://openmodeldb.info/models/4x-FrankendataPretainer-SRFormer) | [CC-BY-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/4x-FrankendataPretainer-SRFormer.json) | no |
| `4xRealWebPhoto_v2_rgt_s.onnx` | RealWebPhoto v2 RGT-S x4 | 49 | [4xRealWebPhoto_v2_rgt_s by helaman (Philip Hofmann / Phhofm) (OpenModelDB 4x-RealWebPhoto-v2-rgt-s)](https://openmodeldb.info/models/4x-RealWebPhoto-v2-rgt-s) | [CC-BY-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/4x-RealWebPhoto-v2-rgt-s.json) | no |
| `4xRealWebPhoto_v4_dat2.onnx` | RealWebPhoto v4 DAT2 x4 | 49 | [4xRealWebPhoto_v4_dat2 by helaman (Philip Hofmann / Phhofm) (OpenModelDB 4x-RealWebPhoto-v4-dat2)](https://openmodeldb.info/models/4x-RealWebPhoto-v4-dat2) | [CC-BY-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/4x-RealWebPhoto-v4-dat2.json) | no |
| `DAT_2_x2.onnx` | DAT2 x2 | 50 | [DAT - Dual Aggregation Transformer (Zheng Chen et al.)](https://github.com/zhengchen1999/DAT) | [Apache-2.0](https://github.com/zhengchen1999/DAT/blob/main/LICENSE) | no |
| `DAT_2_x3.onnx` | DAT2 x3 | 50 | [DAT - Dual Aggregation Transformer (Zheng Chen et al.)](https://github.com/zhengchen1999/DAT) | [Apache-2.0](https://github.com/zhengchen1999/DAT/blob/main/LICENSE) | no |
| `DAT_2_x4.onnx` | DAT2 x4 | 50 | [DAT - Dual Aggregation Transformer (Zheng Chen et al.)](https://github.com/zhengchen1999/DAT) | [Apache-2.0](https://github.com/zhengchen1999/DAT/blob/main/LICENSE) | no |
| `swin2SR-x4-bsrgan-psnr.onnx` | Swin2SR BSRGAN PSNR x4 | 54 | [Swin2SR (mv-lab)](https://github.com/mv-lab/swin2sr) | [Apache-2.0](https://github.com/mv-lab/swin2sr/blob/main/LICENSE) | no |
| `BSRGAN_SwinIR-M_x4_GAN.onnx` | SwinIR-M BSRGAN GAN x4 | 61 | [SwinIR (Jingyun Liang et al.)](https://github.com/JingyunLiang/SwinIR) | [Apache-2.0](https://github.com/JingyunLiang/SwinIR/blob/main/LICENSE) | no |
| `RealESR-AnimeVideo-x4v3.onnx` | RealESR AnimeVideo x4v3 | 2.5 | [Real-ESRGAN (xinntao)](https://github.com/xinntao/Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | no |

### remove_background — `ai.remove_background / 'subject' mask` (8 models; default `u2netp.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `RMBG_1.4.ort` | RMBG 1.4 | 89 | [RMBG-1.4 (BRIA AI)](https://huggingface.co/briaai/RMBG-1.4) | [LicenseRef-bria-rmbg-1.4](https://huggingface.co/briaai/RMBG-1.4) | **yes** — non-commercial / research-only licence |
| `inspyrenet.onnx` | InSPyReNet | 395 | [InSPyReNet (plemeri)](https://github.com/plemeri/InSPyReNet) | [MIT](https://github.com/plemeri/InSPyReNet/blob/main/LICENSE) | no |
| `u2net.onnx` | U2Net | 176 | [U^2-Net (Xuebin Qin et al.)](https://github.com/xuebinqin/U-2-Net) | [Apache-2.0](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE) | no |
| `u2netp.onnx` | U2NetP | 4.6 | [U^2-Net (Xuebin Qin et al.) - U2NetP small](https://github.com/xuebinqin/U-2-Net) | [Apache-2.0](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE) | no |
| `birefnet_swin_tiny.ort` | BiRefNet | 247 | [BiRefNet (ZhengPeng7) - general, swin_v1_tiny backbone](https://github.com/ZhengPeng7/BiRefNet) | [MIT](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE) | no |
| `modnet_portrait_matting.onnx` | MODNet | 26 | [MODNet (ZHKKKe)](https://github.com/ZHKKKe/MODNet) | [Apache-2.0](https://github.com/ZHKKKe/MODNet#license) | no |
| `isnet-general-use.onnx` | ISNet | 179 | [DIS / IS-Net (Xuebin Qin et al.)](https://github.com/xuebinqin/DIS) | [Apache-2.0](https://github.com/xuebinqin/DIS/blob/main/LICENSE.md) | no |
| `yolo11x-seg.onnx` | YOLO v11 | 249 | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | **yes** — strong copyleft licence (GPL/AGPL); check the obligations before use |

### colorize — `ai.colorize` (5 models; default `ddcolor_paper_tiny.ort`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `1x_ColorizerV2_22000G-fp16.onnx` | Colorizer | 33 | [BS_Colorizer/Vapourizer by blackscout (OpenModelDB 1x-BS-Colorizer)](https://openmodeldb.info/models/1x-BS-Colorizer) | [GPL-3.0-only](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-BS-Colorizer.json) | **yes** — strong copyleft licence (GPL/AGPL); check the obligations before use |
| `1x-SpongeColor-Lite-fp16.onnx` | SpongeColor Lite | 10 | [SpongeColor Lite by joey (OpenModelDB 1x-SpongeColor-Lite)](https://openmodeldb.info/models/1x-SpongeColor-Lite) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-SpongeColor-Lite.json) | **yes** — non-commercial / research-only licence |
| `ddcolor_paper_tiny.ort` | DDColor Tiny | 221 | [DDColor (piddnad)](https://github.com/piddnad/DDColor) | [Apache-2.0](https://huggingface.co/piddnad/ddcolor_paper_tiny) | no |
| `ddcolor_modelscope.onnx` | DDColor | 912 | [DDColor (piddnad)](https://github.com/piddnad/DDColor) | [Apache-2.0](https://huggingface.co/piddnad/ddcolor_modelscope) | no |
| `ddcolor_artistic.onnx` | DDColor Artistic | 912 | [DDColor (piddnad)](https://github.com/piddnad/DDColor) | [Apache-2.0](https://huggingface.co/piddnad/ddcolor_artistic) | no |

### dejpeg — `ai.dejpeg` (15 models; default `fbcnn_color_fp16.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `fbcnn_color_fp16.onnx` | FBCNN Color | 144 | [FBCNN (Jiaxi Jiang et al.)](https://github.com/jiaxi-jiang/FBCNN) | [Apache-2.0](https://github.com/jiaxi-jiang/FBCNN/blob/main/LICENSE) | no |
| `fbcnn_grey_fp16.onnx` | FBCNN Grayscale | 144 | [FBCNN (Jiaxi Jiang et al.)](https://github.com/jiaxi-jiang/FBCNN) | [Apache-2.0](https://github.com/jiaxi-jiang/FBCNN/blob/main/LICENSE) | no |
| `fbcnn_gray_double_fp16.onnx` | FBCNN Grayscale Strong | 144 | [FBCNN (Jiaxi Jiang et al.)](https://github.com/jiaxi-jiang/FBCNN) | [Apache-2.0](https://github.com/jiaxi-jiang/FBCNN/blob/main/LICENSE) | no |
| `1x_JPEGDestroyerV2_96000G-fp16.onnx` | JPEG Destroyer V2 | 33 | [JPEGDestroyer by blackscout (OpenModelDB 1x-JPEGDestroyer)](https://openmodeldb.info/models/1x-JPEGDestroyer) | [GPL-3.0-only](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPEGDestroyer.json) | **yes** — strong copyleft licence (GPL/AGPL); check the obligations before use |
| `1x_JPEG_00-20.ort` | JPEG 0-20 | 35 | [JPG (00-20%) by blueamulet (OpenModelDB 1x-JPG-00-20)](https://openmodeldb.info/models/1x-JPG-00-20) | [CC-BY-NC-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-00-20.json) | **yes** — non-commercial / research-only licence |
| `1x_JPEG_20-40.ort` | JPEG 20-40 | 35 | [JPG (20-40%) by blueamulet (OpenModelDB 1x-JPG-20-40)](https://openmodeldb.info/models/1x-JPG-20-40) | [CC-BY-NC-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-20-40.json) | **yes** — non-commercial / research-only licence |
| `1x_JPEG_40-60.ort` | JPEG 40-60 | 35 | [JPG (40-60%) by blueamulet (OpenModelDB 1x-JPG-40-60)](https://openmodeldb.info/models/1x-JPG-40-60) | [CC-BY-NC-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-40-60.json) | **yes** — non-commercial / research-only licence |
| `1x_JPEG_60-80.ort` | JPEG 60-80 | 35 | [JPG (60-80%) by blueamulet (OpenModelDB 1x-JPG-60-80)](https://openmodeldb.info/models/1x-JPG-60-80) | [CC-BY-NC-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-60-80.json) | **yes** — non-commercial / research-only licence |
| `1x_JPEG_80-100.ort` | JPEG 80-100 | 35 | [JPG (80-100%) by blueamulet (OpenModelDB 1x-JPG-80-100)](https://openmodeldb.info/models/1x-JPG-80-100) | [CC-BY-NC-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-80-100.json) | **yes** — non-commercial / research-only licence |
| `1x_artifacts_jpg_00_20_alsa-fp16.ort` | JPEG Artifacts 0-20 | 35 | [JPG (00-20%) by alsa (OpenModelDB 1x-JPG-00-20-alsa)](https://openmodeldb.info/models/1x-JPG-00-20-alsa) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-00-20-alsa.json) | **yes** — non-commercial / research-only licence |
| `1x_artifacts_jpg_20_40_alsa-fp16.ort` | JPEG Artifacts 20-40 | 35 | [JPG (20-40%) by alsa (OpenModelDB 1x-JPG-20-40-alsa)](https://openmodeldb.info/models/1x-JPG-20-40-alsa) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-20-40-alsa.json) | **yes** — non-commercial / research-only licence |
| `1x_artifacts_jpg_40_60_alsa-fp16.ort` | JPEG Artifacts 40-60 | 35 | [JPG (40-60%) by alsa (OpenModelDB 1x-JPG-40-60-alsa)](https://openmodeldb.info/models/1x-JPG-40-60-alsa) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-40-60-alsa.json) | **yes** — non-commercial / research-only licence |
| `1x_artifacts_jpg_60_80_alsa-fp16.ort` | JPEG Artifacts 60-80 | 35 | [JPG (60-80%) by alsa (OpenModelDB 1x-JPG-60-80-alsa)](https://openmodeldb.info/models/1x-JPG-60-80-alsa) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-60-80-alsa.json) | **yes** — non-commercial / research-only licence |
| `1x_artifacts_jpg_80_100_alsa-fp16.ort` | JPEG Artifacts 80-100 | 35 | [JPG (80-100%) by alsa (OpenModelDB 1x-JPG-80-100-alsa)](https://openmodeldb.info/models/1x-JPG-80-100-alsa) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-JPG-80-100-alsa.json) | **yes** — non-commercial / research-only licence |
| `1x_SBDV-DeJPEG-Lite_130000_G.ort` | SBDV DeJPEG | 11 | [SBDV-DeJPEG Lite by joey (OpenModelDB 1x-SBDV-DeJPEG-Lite)](https://openmodeldb.info/models/1x-SBDV-DeJPEG-Lite) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-SBDV-DeJPEG-Lite.json) | **yes** — non-commercial / research-only licence |

### denoise — `ai.denoise` (16 models; default `scunet_color_25_fp16.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `scunet_color-GAN.onnx` | SCUNet Color GAN | 91 | [SCUNet (cszn)](https://github.com/cszn/SCUNet) | [Apache-2.0](https://github.com/cszn/SCUNet/blob/main/LICENSE) | no |
| `scunet_color-PSNR.onnx` | SCUNet Color PSNR | 91 | [SCUNet (cszn)](https://github.com/cszn/SCUNet) | [Apache-2.0](https://github.com/cszn/SCUNet/blob/main/LICENSE) | no |
| `scunet_gray_15_fp16.onnx` | SCUNet Grayscale 15 | 38 | [SCUNet (cszn)](https://github.com/cszn/SCUNet) | [Apache-2.0](https://github.com/cszn/SCUNet/blob/main/LICENSE) | no |
| `scunet_gray_25_fp16.onnx` | SCUNet Grayscale 25 | 38 | [SCUNet (cszn)](https://github.com/cszn/SCUNet) | [Apache-2.0](https://github.com/cszn/SCUNet/blob/main/LICENSE) | no |
| `scunet_gray_50_fp16.onnx` | SCUNet Grayscale 50 | 38 | [SCUNet (cszn)](https://github.com/cszn/SCUNet) | [Apache-2.0](https://github.com/cszn/SCUNet/blob/main/LICENSE) | no |
| `scunet_color_15_fp16.onnx` | SCUNet Color 15 | 43 | [SCUNet (cszn)](https://github.com/cszn/SCUNet) | [Apache-2.0](https://github.com/cszn/SCUNet/blob/main/LICENSE) | no |
| `scunet_color_25_fp16.onnx` | SCUNet Color 25 | 43 | [SCUNet (cszn)](https://github.com/cszn/SCUNet) | [Apache-2.0](https://github.com/cszn/SCUNet/blob/main/LICENSE) | no |
| `scunet_color_50_fp16.onnx` | SCUNet Color 50 | 43 | [SCUNet (cszn)](https://github.com/cszn/SCUNet) | [Apache-2.0](https://github.com/cszn/SCUNet/blob/main/LICENSE) | no |
| `1x-RGB-max-Denoise-fp16.onnx` | RGB Max Denoise | 0.3 | [1x RGB max Denoise by loganavter (OpenModelDB 1x-RGB-max-Denoise)](https://openmodeldb.info/models/1x-RGB-max-Denoise) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-RGB-max-Denoise.json) | **yes** — non-commercial / research-only licence |
| `1x-WB-Denoise-fp16.onnx` | WB Denoise | 0.3 | [1x BW Denoise by loganavter (OpenModelDB 1x-BW-Denoise)](https://openmodeldb.info/models/1x-BW-Denoise) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-BW-Denoise.json) | **yes** — non-commercial / research-only licence |
| `1x_NoiseToner-Poisson-Detailed_108000_G-fp16.onnx` | Noise Toner Poisson Detailed | 33 | [NoiseToner Poisson Detailed by mutin-choler (OpenModelDB 1x-NoiseToner-Poisson-Detailed)](https://openmodeldb.info/models/1x-NoiseToner-Poisson-Detailed) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-NoiseToner-Poisson-Detailed.json) | no |
| `1x_NoiseToner-Poisson-Soft_101000_G-fp16.onnx` | Noise Toner Poisson Soft | 33 | [NoiseToner Poisson Soft by mutin-choler (OpenModelDB 1x-NoiseToner-Poisson-Soft)](https://openmodeldb.info/models/1x-NoiseToner-Poisson-Soft) | [unknown (not stated)](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-NoiseToner-Poisson-Soft.json) | **yes** — licence of the weights could not be verified |
| `1x_NoiseToner-Uniform-Detailed_100000_G-fp16.onnx` | Noise Toner Uniform Detailed | 33 | [NoiseToner Uniform Detailed by mutin-choler (OpenModelDB 1x-NoiseToner-Uniform-Detailed)](https://openmodeldb.info/models/1x-NoiseToner-Uniform-Detailed) | [unknown (not stated)](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-NoiseToner-Uniform-Detailed.json) | **yes** — licence of the weights could not be verified |
| `1x_NoiseToner-Uniform-Soft_100000_G-fp16.onnx` | Noise Toner Uniform Soft | 33 | [NoiseToner Uniform Soft by mutin-choler (OpenModelDB 1x-NoiseToner-Uniform-Soft)](https://openmodeldb.info/models/1x-NoiseToner-Uniform-Soft) | [unknown (not stated)](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-NoiseToner-Uniform-Soft.json) | **yes** — licence of the weights could not be verified |
| `1x_ISO_denoise_v1.ort` | ISO Denoise v1 | 35 | [ISO denoise v1 by alpha (OpenModelDB 1x-ISO-denoise-v1)](https://openmodeldb.info/models/1x-ISO-denoise-v1) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-ISO-denoise-v1.json) | no |
| `deblurring_nafnet_2025may.onnx` | NAFNET deblurring | 92 | [NAFNet (megvii-research); ONNX by OpenCV Zoo](https://github.com/megvii-research/NAFNet) | [MIT](https://github.com/opencv/opencv_zoo/blob/main/models/deblurring_nafnet/LICENSE) | no |

### remove_artifacts — `ai.remove_artifacts` (11 models; default `1x_Bendel_Halftone-fp32.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `1x-BroadcastToStudio_Compact-fp16.onnx` | Broadcast To Studio | 1.2 | [BroadcastToStudio Compact by kim2091 (OpenModelDB 1x-BroadcastToStudio-Compact)](https://openmodeldb.info/models/1x-BroadcastToStudio-Compact) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-BroadcastToStudio-Compact.json) | **yes** — non-commercial / research-only licence |
| `1x-Anti-Aliasing-fp16.onnx` | Anti-Aliasing | 33 | [Anti Aliasing by twittman (OpenModelDB 1x-Anti-Aliasing)](https://openmodeldb.info/models/1x-Anti-Aliasing) | [CC-BY-NC-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-Anti-Aliasing.json) | **yes** — non-commercial / research-only licence |
| `1x_DeEdge-fp16.onnx` | DeEdge | 33 | [DeEdge by mutin-choler (OpenModelDB 1x-DeEdge)](https://openmodeldb.info/models/1x-DeEdge) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-DeEdge.json) | no |
| `1x-cinepak-fp16.onnx` | Cinepak | 33 | [cinepak by twittman (OpenModelDB 1x-cinepak)](https://openmodeldb.info/models/1x-cinepak) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-cinepak.json) | **yes** — non-commercial / research-only licence |
| `1x_Bandage-Smooth-fp16.onnx` | Bandage Smooth | 33 | [Bandage Smooth by mutin-choler (OpenModelDB 1x-Bandage-Smooth)](https://openmodeldb.info/models/1x-Bandage-Smooth) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-Bandage-Smooth.json) | no |
| `1x_Bendel_Halftone-fp32.onnx` | Bendel Halftone | 8.7 | [Bendel_Halftone by bendel (OpenModelDB 1x-Bendel-Halftone)](https://openmodeldb.info/models/1x-Bendel-Halftone) | [CC-BY-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-Bendel-Halftone.json) | no |
| `1x_NMKD-h264Texturize-fp16.onnx` | NMKD H264 Texturize | 33 | [NMKD h264Texturize by nmkd (OpenModelDB 1x-NMKD-h264Texturize)](https://openmodeldb.info/models/1x-NMKD-h264Texturize) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-NMKD-h264Texturize.json) | no |
| `1x_artifacts_dithering_alsa-fp16.onnx` | Artifacts Dithering ALSA | 33 | [Artifacts Dithering alsa by alsa (OpenModelDB 1x-artifacts-dithering-alsa)](https://openmodeldb.info/models/1x-artifacts-dithering-alsa) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-artifacts-dithering-alsa.json) | **yes** — non-commercial / research-only licence |
| `1x_DeJumbo.ort` | DeJumbo | 35 | [1x_DeJumbo (author unknown)](https://huggingface.co/utnah/esrgan/blob/main/1x_DeJumbo.pth) | [unknown](https://huggingface.co/utnah/esrgan) | **yes** — licence of the weights could not be verified |
| `1x-Fatality-DeBlur.onnx` | Fatality DeBlur | 33 | [Fatality DeBlur by twittman (OpenModelDB 1x-Fatality-DeBlur)](https://openmodeldb.info/models/1x-Fatality-DeBlur) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-Fatality-DeBlur.json) | **yes** — non-commercial / research-only licence |
| `1x-UnResize-V3.onnx` | UnResize V3 | 33 | [UnResize V3 by twittman (OpenModelDB 1x-UnResize-V3)](https://openmodeldb.info/models/1x-UnResize-V3) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-UnResize-V3.json) | **yes** — non-commercial / research-only licence |

### enhance — `ai.enhance` (21 models; default `1xOverExposureCorrection_compact-fp16.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `1xOverExposureCorrection_compact-fp16.onnx` | Overexposure Correction | 2.4 | [OverExposureCorrection compact by helaman (Philip Hofmann / Phhofm) (OpenModelDB 1x-OverExposureCorrection-compact)](https://openmodeldb.info/models/1x-OverExposureCorrection-compact) | [CC-BY-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-OverExposureCorrection-compact.json) | no |
| `zero_dcepp.onnx` | Zero-DCE++ | 0.1 | [Zero-DCE++ (Chongyi Li et al., TPAMI 2021)](https://github.com/Li-Chongyi/Zero-DCE_extension) | [CC-BY-NC-4.0](https://github.com/Li-Chongyi/Zero-DCE_extension/blob/main/README.md) | **yes** — non-commercial / research-only licence |
| `1x_DeSharpen-fp16.onnx` | DeSharpen | 33 | [DeSharpen by loinne (OpenModelDB 1x-DeSharpen)](https://openmodeldb.info/models/1x-DeSharpen) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-DeSharpen.json) | **yes** — non-commercial / research-only licence |
| `1x_GainresV4-fp16.onnx` | GainRes | 33 | [GainRESV4 by cf2lter (OpenModelDB 1x-GainRESV4)](https://openmodeldb.info/models/1x-GainRESV4) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-GainRESV4.json) | **yes** — non-commercial / research-only licence |
| `1x-DeBink-v4.onnx` | DeBink v4 | 33 | [DeBink v4 by kim2091 (OpenModelDB 1x-DeBink-v4)](https://openmodeldb.info/models/1x-DeBink-v4) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-DeBink-v4.json) | **yes** — non-commercial / research-only licence |
| `1x-DeBink-v5.onnx` | DeBink v5 | 33 | [DeBink v5 by kim2091 (OpenModelDB 1x-DeBink-v5)](https://openmodeldb.info/models/1x-DeBink-v5) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-DeBink-v5.json) | **yes** — non-commercial / research-only licence |
| `1x-DeBink-v6.onnx` | DeBink v6 | 33 | [DeBink v6 by kim2091 (OpenModelDB 1x-DeBink-v6)](https://openmodeldb.info/models/1x-DeBink-v6) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-DeBink-v6.json) | **yes** — non-commercial / research-only licence |
| `1x-NMKD-Jaywreck3-Lite-fp16.onnx` | NMKD Jaywreck3 Lite | 10 | [NMKD Jaywreck3 (Lite) by nmkd (OpenModelDB 1x-NMKD-Jaywreck3-Lite)](https://openmodeldb.info/models/1x-NMKD-Jaywreck3-Lite) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-NMKD-Jaywreck3-Lite.json) | no |
| `1x_BCGone-DetailedV2_40-60_115000_G-fp16.onnx` | BCGone Detailed V2 | 33 | [BCGone Detailed V2 40-60 by mutin-choler (OpenModelDB 1x-BCGone-DetailedV2-40-60)](https://openmodeldb.info/models/1x-BCGone-DetailedV2-40-60) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-BCGone-DetailedV2-40-60.json) | no |
| `1x_BCGone_Smooth_110000_G-fp16.onnx` | BCGone Smooth | 33 | [BCGone Smooth by mutin-choler (OpenModelDB 1x-BCGone-Smooth)](https://openmodeldb.info/models/1x-BCGone-Smooth) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-BCGone-Smooth.json) | no |
| `1x_DitherDeleterV3-Smooth-fp16.onnx` | Dither Deleter V3 Smooth | 33 | [DitherDeleterV3 Smooth by mutin-choler (OpenModelDB 1x-DitherDeleterV3-Smooth)](https://openmodeldb.info/models/1x-DitherDeleterV3-Smooth) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-DitherDeleterV3-Smooth.json) | no |
| `VHS-Sharpen-1x_46000_G-fp16.onnx` | VHS Sharpen | 33 | [VHS-Sharpen by rtx-2080-ti-hoarder (OpenModelDB 1x-VHS-Sharpen)](https://openmodeldb.info/models/1x-VHS-Sharpen) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-VHS-Sharpen.json) | **yes** — non-commercial / research-only licence |
| `1x_NMKD-BrightenRedux_200k-fp16.onnx` | NMKD Brighten Redux | 33 | [1x_NMKD-BrightenRedux_200k (NMKD per filename; author not verified)](https://huggingface.co/utnah/esrgan/blob/main/1x_NMKD-BrightenRedux_200k.pth) | [unknown](https://huggingface.co/utnah/esrgan) | **yes** — licence of the weights could not be verified |
| `1x_nmkdbrighten_10000_G-fp16.onnx` | NMKD Brighten | 33 | [1x_nmkdbrighten_10000_G (NMKD per filename; author not verified)](https://huggingface.co/utnah/esrgan/blob/main/1x_nmkdbrighten_10000_G.pth) | [unknown](https://huggingface.co/utnah/esrgan) | **yes** — licence of the weights could not be verified |
| `1x_NMKDDetoon_97500_G-fp16.onnx` | NMKD Detoon | 33 | [1x_NMKDDetoon_97500_G (NMKD per filename; author not verified)](https://huggingface.co/utnah/esrgan/blob/main/1x_NMKDDetoon_97500_G.pth) | [unknown](https://huggingface.co/utnah/esrgan) | **yes** — licence of the weights could not be verified |
| `1x_Repainter_20000_G-fp16.onnx` | Repainter | 33 | [1x_Repainter_20000_G (author unknown)](https://huggingface.co/utnah/esrgan/blob/main/1x_Repainter_20000_G.pth) | [unknown](https://huggingface.co/utnah/esrgan) | **yes** — licence of the weights could not be verified |
| `1x-Debandurh-FS-Ultra-lite-fp16.onnx` | Debandurh FS Ultra Lite | 1.4 | [Debandurh_FS Ultra-lite by twittman (OpenModelDB 1x-Debandurh-FS-Ultra-lite)](https://openmodeldb.info/models/1x-Debandurh-FS-Ultra-lite) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-Debandurh-FS-Ultra-lite.json) | **yes** — non-commercial / research-only licence |
| `1x_DeBLR.ort` | DeBLR | 35 | [DeBLR by blackscout (OpenModelDB 1x-DeBLR)](https://openmodeldb.info/models/1x-DeBLR) | [unknown (not stated)](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-DeBLR.json) | **yes** — licence of the weights could not be verified |
| `1x_ReDetail_v2_126000_G-fp16.ort` | ReDetail v2 | 35 | [1x_ReDetail_v2_126000_G (author unknown)](https://huggingface.co/utnah/esrgan/blob/main/1x_ReDetail_v2_126000_G.pth) | [unknown](https://huggingface.co/utnah/esrgan) | **yes** — licence of the weights could not be verified |
| `1x-ITF-SkinDiffDetail-Lite-v1.ort` | ITF Skin DiffDetail Lite | 11 | [ITF SkinDiffDetail Lite v1 by intheflesh (OpenModelDB 1x-ITF-SkinDiffDetail-Lite-v1)](https://openmodeldb.info/models/1x-ITF-SkinDiffDetail-Lite-v1) | [CC-BY-NC-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-ITF-SkinDiffDetail-Lite-v1.json) | **yes** — non-commercial / research-only licence |
| `restormer_deraining.onnx` | Restormer Deraining | 107 | [Restormer (Syed Waqas Zamir et al. / swz30)](https://github.com/swz30/Restormer) | [MIT](https://github.com/swz30/Restormer/blob/main/LICENSE.md) | no |

### restore_anime — `ai.restore_anime` (2 models; default `1x-span-anime-pretrain-fp16.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `1x-AnimeUndeint-Compact-fp16.onnx` | Anime Undeint | 2.4 | [AnimeUndeint Compact by zarxrax (OpenModelDB 1x-AnimeUndeint-Compact)](https://openmodeldb.info/models/1x-AnimeUndeint-Compact) | [WTFPL](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-AnimeUndeint-Compact.json) | no |
| `1x-span-anime-pretrain-fp16.onnx` | SPAN Anime Pretrain | 0.8 | [1x-span_anime_pretrain by kim2091 (OpenModelDB 1x-span-anime-pretrain)](https://openmodeldb.info/models/1x-span-anime-pretrain) | [CC0-1.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-span-anime-pretrain.json) | no |

### restore_scans — `ai.restore_scans` (3 models; default `1xBook-Compact-fp16.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `1xBook-Compact-fp16.onnx` | Book Scan | 1.2 | [Book Compact by asterixcool (OpenModelDB 1x-Book-Compact)](https://openmodeldb.info/models/1x-Book-Compact) | [CC-BY-SA-4.0](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-Book-Compact.json) | no |
| `1x-KDM003-scans-fp16.onnx` | KDM003 Scans | 33 | [KDM003 scans by boxdrop (OpenModelDB 1x-KDM003-scans)](https://openmodeldb.info/models/1x-KDM003-scans) | [unknown (not stated)](https://github.com/OpenModelDB/open-model-database/blob/main/data/models/1x-KDM003-scans.json) | **yes** — licence of the weights could not be verified |
| `uvdoc_grid.onnx` | UVDOC Grid | 32 | [UVDoc (Tanguy Magne et al.)](https://github.com/tanguymagne/UVDoc) | [MIT](https://github.com/tanguymagne/UVDoc/blob/main/LICENSE) | no |

### style_transfer — `ai.style_transfer` (4 models; default `microast_style_transfer_onnx.zip`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `arbitrary_style_transfer_onnx.zip` | Arbitrary Style Transfer | 11 | [Magenta arbitrary-image-stylization (Ghiasi et al.), distilled MobileNet-v2 style net + separable-conv transformer by Reiichiro Nakano; ONNX port by patlevin/pdn-styletransfer](https://github.com/reiinakano/arbitrary-image-stylization-tfjs) | [Apache-2.0](https://github.com/reiinakano/arbitrary-image-stylization-tfjs/blob/master/LICENSE) | no |
| `microast_style_transfer_onnx.zip` | MicroAST | 1.8 | [MicroAST (Zhizhong Wang et al., AAAI 2023)](https://github.com/EndyWon/MicroAST) | [MIT](https://github.com/EndyWon/MicroAST/blob/main/LICENSE) | no |
| `aesfa_style_transfer_onnx.zip` | AesFA | 12 | [AesFA (Sooyoung Kim et al., AAAI 2024)](https://github.com/Sooyyoungg/AesFA) | [MIT](https://github.com/Sooyyoungg/AesFA/blob/main/LICENSE) | no |
| `vgg19_optimization_style_transfer_onnx.zip` | VGG19 Optimization | 48 | [torchvision VGG19 IMAGENET1K_V1 weights (vgg19-dcbb9e9d.pth); NST pipeline after nazianafis/Neural-Style-Transfer](https://github.com/pytorch/vision) | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | no |

### depth — `ai.depth_map` (14 models; default `depth_anything_v2_vits_q4.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `depth_anything_v2_vits_q4.onnx` | Depth Anything V2 Small (Q4) | 27 | [depth-anything/Depth-Anything-V2-Small (ONNX from onnx-community/depth-anything-v2-small)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) | [Apache-2.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/README.md) | no |
| `depth_anything_v2_vits_fp32.onnx` | Depth Anything V2 Small (FP32) | 99 | [depth-anything/Depth-Anything-V2-Small (ONNX from onnx-community/depth-anything-v2-small)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) | [Apache-2.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/README.md) | no |
| `depth_anything_v2_vitb_q4.onnx` | Depth Anything V2 Base (Q4) | 102 | [depth-anything/Depth-Anything-V2-Base (ONNX from onnx-community/depth-anything-v2-base)](https://huggingface.co/depth-anything/Depth-Anything-V2-Base) | [CC-BY-NC-4.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/README.md) | **yes** — non-commercial / research-only licence |
| `depth_anything_v2_vitb_fp32.onnx` | Depth Anything V2 Base (FP32) | 389 | [depth-anything/Depth-Anything-V2-Base (ONNX from onnx-community/depth-anything-v2-base)](https://huggingface.co/depth-anything/Depth-Anything-V2-Base) | [CC-BY-NC-4.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/README.md) | **yes** — non-commercial / research-only licence |
| `depth_anything_v2_vitl_q4.onnx` | Depth Anything V2 Large (Q4) | 318 | [depth-anything/Depth-Anything-V2-Large (ONNX from onnx-community/depth-anything-v2-large)](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) | [CC-BY-NC-4.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/README.md) | **yes** — non-commercial / research-only licence |
| `depth_anything_v2_vitl_fp32.onnx` | Depth Anything V2 Large (FP32) | 1337 | [depth-anything/Depth-Anything-V2-Large (ONNX from onnx-community/depth-anything-v2-large)](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) | [CC-BY-NC-4.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/README.md) | **yes** — non-commercial / research-only licence |
| `depth_anything_v3_small_int8.onnx` | Depth Anything 3 Small (INT8) | 29 | [depth-anything/DA3-SMALL (ONNX from en970/depth-anything-v3-small-onnx)](https://huggingface.co/depth-anything/DA3-SMALL) | [Apache-2.0](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/README.md) | no |
| `depth_anything_v3_small_fp32.onnx` | Depth Anything 3 Small (FP32) | 105 | [depth-anything/DA3-SMALL (ONNX from onnx-community/depth-anything-v3-small)](https://huggingface.co/depth-anything/DA3-SMALL) | [Apache-2.0](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/README.md) | no |
| `depth_anything_v3_base_int4.onnx` | Depth Anything 3 Base (INT4) | 118 | [depth-anything/DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | [Apache-2.0](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/README.md) | no |
| `depth_anything_v3_base_fp32.onnx` | Depth Anything 3 Base (FP32) | 413 | [depth-anything/DA3-BASE (ONNX from onnx-community/depth-anything-v3-base)](https://huggingface.co/depth-anything/DA3-BASE) | [Apache-2.0](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/README.md) | no |
| `depth_anything_v3_large_int4.onnx` | Depth Anything 3 Large (INT4) | 306 | [depth-anything/DA3-LARGE-1.1 (ONNX from mindqtrl/DA3-LARGE-1.1-ONNX-STRIPPED)](https://huggingface.co/depth-anything/DA3-LARGE-1.1) | [conflicting: CC-BY-NC-4.0 (official GitHub model table) vs Apache-2.0 (official HF model card)](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/README.md) | **yes** — non-commercial / research-only licence |
| `depth_anything_v3_large_fp32.onnx` | Depth Anything 3 Large (FP32) | 1383 | [depth-anything/DA3-LARGE (original, deprecated; ONNX from onnx-community/depth-anything-v3-large)](https://huggingface.co/depth-anything/DA3-LARGE) | [CC-BY-NC-4.0](https://huggingface.co/depth-anything/DA3-LARGE) | **yes** — non-commercial / research-only licence |
| `depth_anything_v3_mono_large_q4.onnx` | Depth Anything 3 Mono Large (Q4) | 293 | [depth-anything/DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE) | [Apache-2.0](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/README.md) | no |
| `depth_anything_v3_mono_large_fp32.onnx` | Depth Anything 3 Mono Large (FP32) | 1341 | [depth-anything/DA3MONO-LARGE (ONNX from yuvraj108c/Depth-Anything-3-Onnx)](https://huggingface.co/depth-anything/DA3MONO-LARGE) | [Apache-2.0](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/README.md) | no |

### inpaint — `ai.inpaint` (2 models; default `LaMa_512.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `LaMa_512.onnx` | LaMa 512 | 208 | [LaMa (advimman); ONNX export by Carve (Carve/LaMa-ONNX)](https://huggingface.co/Carve/LaMa-ONNX) | [Apache-2.0](https://huggingface.co/Carve/LaMa-ONNX) | no |
| `LaMa_512_FAST.onnx` | LaMa 512 Fast | 183 | [LaMa dilated/multidilated-conv variant (advimman/lama architecture); exporter unidentified](https://github.com/advimman/lama) | [unknown](https://github.com/advimman/lama/blob/main/LICENSE) | **yes** — licence of the weights could not be verified |

### remove_watermark — `ai.remove_watermark` (1 models; default `watermark_mit_b5_sigmoid.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `watermark_mit_b5_sigmoid.onnx` | Watermark Remover | 341 | [Diffusion-Dynamics/watermark-segmentation (clear.photo) checkpoint best_watermark_model_mit_b5_best.pth (smp Unet + MiT-B5 encoder)](https://github.com/Diffusion-Dynamics/watermark-segmentation) | [MIT AND LicenseRef-NVIDIA-Source-Code-License-SegFormer (non-commercial)](https://github.com/Diffusion-Dynamics/watermark-segmentation/blob/main/LICENSE) | **yes** — non-commercial / research-only licence |

### detect_ai_generated — `tool detect_ai_generated` (11 models; default `ai_detector_distilled_vit_small_fp32.onnx`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `ai_detector_mirage_siglip2_base_fp16.onnx` | Mirage SigLIP2 Base (FP16) | 173 | [prithivMLmods/Mirage-Photo-Classifier](https://huggingface.co/prithivMLmods/Mirage-Photo-Classifier) | [Apache-2.0](https://huggingface.co/prithivMLmods/Mirage-Photo-Classifier) | no |
| `ai_detector_mirage_siglip2_base_fp32.onnx` | Mirage SigLIP2 Base (FP32) | 345 | [prithivMLmods/Mirage-Photo-Classifier](https://huggingface.co/prithivMLmods/Mirage-Photo-Classifier) | [Apache-2.0](https://huggingface.co/prithivMLmods/Mirage-Photo-Classifier) | no |
| `ai_detector_lpx_swinv2_large_fp32.onnx` | LPX AI Detector SwinV2 Large | 905 | [haywoodsloan/ai-image-detector-deploy (ONNX copied from LPX55/detection-model-1-ONNX)](https://huggingface.co/haywoodsloan/ai-image-detector-deploy) | [Apache-2.0](https://huggingface.co/haywoodsloan/ai-image-detector-deploy) | no |
| `ai_detector_convnext_tiny_fp32.onnx` | ConvNeXT Tiny AI Detector | 111 | [mmanikanta/ConvNeXT_AI_image_detector](https://huggingface.co/mmanikanta/ConvNeXT_AI_image_detector) | [Apache-2.0](https://huggingface.co/mmanikanta/ConvNeXT_AI_image_detector) | no |
| `ai_detector_resnet50_fp32.onnx` | ResNet-50 AI Detector | 94 | [mmanikanta/ResNet_AI_image_detector](https://huggingface.co/mmanikanta/ResNet_AI_image_detector) | [Apache-2.0](https://huggingface.co/mmanikanta/ResNet_AI_image_detector) | no |
| `ai_detector_ai_vs_deepfake_vs_real_fp32.onnx` | AI vs Deepfake vs Real | 343 | [prithivMLmods/AI-vs-Deepfake-vs-Real (ONNX from prithivMLmods/AI-vs-Deepfake-vs-Real-ONNX)](https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real) | [Apache-2.0](https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real) | no |
| `ai_detector_capcheck_vit_base_fp32.onnx` | CapCheck AI Image Detection | 343 | [capcheck/ai-image-detection (now redirects to aaronkantrowitz/ai-image-detection); weights originate from dima806/ai_vs_real_image_detection](https://huggingface.co/aaronkantrowitz/ai-image-detection) | [Apache-2.0](https://huggingface.co/aaronkantrowitz/ai-image-detection) | no |
| `ai_detector_deepfake_v2_vit_base_fp32.onnx` | Deep Fake Detector V2 | 343 | [prithivMLmods/Deep-Fake-Detector-v2-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | [Apache-2.0](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) | no |
| `ai_detector_distilled_vit_small_fp32.onnx` | AI Image Detect Distilled | 58 | [jacoballessio/ai-image-detect-distilled](https://huggingface.co/jacoballessio/ai-image-detect-distilled) | [MIT](https://huggingface.co/jacoballessio/ai-image-detect-distilled) | no |
| `ai_detector_community_forensics_vit_small_384_fp32.onnx` | Community Forensics ViT-S/384 | 144 | [Intended: OwensLab/commfor-model-384 (Park & Owens, Community Forensics); the shipped weights are not from it](https://huggingface.co/OwensLab/commfor-model-384) | [unknown](https://huggingface.co/OwensLab/commfor-model-384) | **yes** — licence of the weights could not be verified |
| `ai_detector_smogy_swin_base_fp32.onnx` | SMOGY Swin Base | 352 | [Smogy/SMOGY-Ai-images-detector](https://huggingface.co/Smogy/SMOGY-Ai-images-detector) | [CC-BY-NC-4.0](https://huggingface.co/Smogy/SMOGY-Ai-images-detector) | **yes** — non-commercial / research-only licence |

### ocr — `tool ocr_paddle` (13 models; default `paddleocr_v5.zip`)

| Model (`name`) | Title | Size MB | Upstream project | Weight licence | Restricted? |
|---|---|---:|---|---|---|
| `paddleocr_v5.zip` | PaddleOCR CJK | 19 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_korean.zip` | PaddleOCR Korean | 13 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_latin.zip` | PaddleOCR Latin | 12 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_eslav.zip` | PaddleOCR East Slavic | 12 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_th.zip` | PaddleOCR Thai | 12 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_el.zip` | PaddleOCR Greek | 12 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_en.zip` | PaddleOCR English | 12 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_cyrillic.zip` | PaddleOCR Cyrillic | 12 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_arabic.zip` | PaddleOCR Arabic | 13 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_devanagari.zip` | PaddleOCR Devanagari | 13 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_ta.zip` | PaddleOCR Tamil | 13 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v5_te.zip` | PaddleOCR Telugu | 13 | [PaddleOCR PP-OCRv5 (PaddlePaddle)](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec) | no |
| `paddleocr_v6_medium.zip` | PaddleOCR v6 (medium) | 101 | [PaddleOCR PP-OCRv6_medium (PaddlePaddle)](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det) | [Apache-2.0](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det) | no |
<!-- TABLE END -->
