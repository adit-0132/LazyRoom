# Resampling filters & scale colour spaces (pic-scale 0.6.6 via aire)

ImageToolbox resizes with aire's `ScalePipelines.scale(bitmap, w, h, ResizeFunction, ScaleColorSpace)`.
This goes through `aire-filters/src/lib.rs` (`resizeImpl`) to the Rust crate **pic-scale 0.6.6**, and from
there to **colorutils-rs 0.7** for colour conversions. LazyRoom's port:

* `src/lazyroom/ops/resample.py`: `KERNELS` (49 `ResizeFunction` entries), `SCALE_COLOR_SPACES`
  (19 `ScaleColorSpace` entries), `resample(img, width, height, method, colorspace="srgb")`,
  `weight_matrix(n_in, n_out, filter)`, `encode_space` / `decode_space`.
* `src/lazyroom/ops/geometry.py`: `geometry.resize` gains a `colorspace` choice (default `srgb`),
  passed through `resize_buffer`.
* Tests: `tests/test_resample.py`.

## Weight generation vs pic-scale `scaler.rs::generate_weights`

Both implementations are separable: one 1-D weight set per output row and per output column. Weights are
normalised to sum 1 and applied in float32. Image borders are handled the same way in both: out-of-range
taps are dropped and the remaining weights renormalised, with no edge extension. pic-scale runs the
vertical pass first; we run the vertical pass first unless upscaling vertically. The two passes commute,
so this only saves time.

| Step | pic-scale 0.6.6 | LazyRoom |
|---|---|---|
| scale | `in/out`; `cutoff = max(scale, 1)`, or 1 for fixed kernels (spline16/36/64) | same |
| support | `kernel_size = round(min_kernel_size * cutoff)` taps, `radius = kernel_size / 2` | `radius = R * cutoff`, where **R is the kernel's real support** (table below). `min_kernel_size` is the support radius the kernels were written for (Lanczos3 = 3, cubic = 2, Lagrange3 = 3); pic-scale halves it, so Lanczos3 gets 3 taps (radius 1.5) at upscale and every cubic loses its negative lobes |
| tap window | `start = floor(cx - radius)`, `end = min(ceil(cx + radius), start + kernel_size)` with `cx = (i + 0.5) * scale`. This set is **biased half a pixel to the left**: it can include a pixel up to `radius + 0.5` away on the left and then drops the rightmost in-range pixel. Example: Lanczos3 2x upscale, output 10 uses inputs 3,4,5 (dx = -1.75, -0.75, 0.25) instead of 4,5,6 | all `k` with `abs(k - c) <= radius`, `c = (i + 0.5) * scale - 0.5`. The window is symmetric, and distances are computed exactly from integers (`dx = ((2k+1) n_out - (2i+1) n_in) / (2 n_out)`), so mirrored images get mirrored weights bit-for-bit (and ties on a discontinuous kernel's support edge are decided identically on both sides) |
| weight | `kernel(abs(dx) * filter_scale)` | same: `kernel(abs(dx) / cutoff)` |
| windowed kernels (ginseng, haasn_soft, ewa_hanning, ewa_lanczos_*, ewa_ginseng) | `x = abs(dx)`; `x /= blur` if `blur > 0`; `x = 0 if x <= taper else (x - taper)/(1 - taper)`; `xk = x * filter_scale`; `w = window(xk * window_size) if x < window_size else 0`; `weight = w * kernel(xk)` | same formula, except the cutoff compares `xk < window_size` (kernel units) instead of the *unscaled* `x`. Upstream's cutoff truncates the window at `window_size` **input pixels**, so these filters alias more and more as the downscale factor grows. Identical at upscale. `taper` is 0 for every entry, so the taper step is the identity in both |
| EWA smoothing (`is_ewa`) | `w'[0] = w[0]`, `w'[j] = 0.7 w[j] + 0.3 w'[j-1]` over the taps left to right: a causal one-pole filter whose centroid shift is 0.3/0.7 ≈ 0.43 taps, which **shifts the image** (≈ 0.9 output px at 2x upscale) | the same recurrence run left-to-right and right-to-left, averaged (`_ewa_smooth`). Same softening (weights still renormalised), zero phase |
| zero / non-positive weight sum | row left at 0, so **black output pixels**. This happens for EWA-Hanning (jinc(0) = 0), HaasnSoft at non-integer upscales (no tap within 0.37 px), and Bartlett at 2x upscale | the nearest input pixel gets weight 1 (the limit of a kernel narrower than the tap spacing) |
| Area | upscale: OpenCV INTER_AREA upscale coefficients (`sx = floor(i*scale)`, `fx = (i+1) - (sx+1)/scale`, weights `1-fx', fx'`). Downscale: a flat box over `round(2*scale)` taps (twice INTER_AREA's width, despite the "replicates INTER_AREA" doc comment) | upscale: verbatim port. Downscale: exact pixel coverage = OpenCV INTER_AREA. Both match `cv2.resize(INTER_AREA)` to 2e-7 (tested) |
| Nearest | `src = trunc((x + 0.5) * scale - 0.5)`, i.e. floor of the centre, a half-pixel bias | `src = floor((x + 0.5) * scale)` = `cv2.INTER_NEAREST_EXACT` (tested bit-exact) |

## Kernels

"Status" refers to the kernel function itself; the weight generation above applies to all of them.
*mks* = pic-scale `min_kernel_size` (upstream radius = mks/2). *R* = support radius used here, in kernel
units (multiplied by `max(scale, 1)` for resizable kernels). Upstream `sinc`/`jinc` are the unnormalised
`sin(x)/x` and `J1(x)/x` (pxfm). They are ported as `np.sinc(x/pi)` and `scipy.special.j1(x)/x`.

| ResizeFunction | name | upstream kernel (math/*.rs) | mks | R | status | notes |
|---|---|---|---|---|---|---|
| Bilinear (0) | `bilinear` | `1-abs(x)` | 2 | 1 | faithful | antialiased (scaled) at downscale, unlike cv2 INTER_LINEAR |
| Nearest (1) | `nearest` | point sampler | – | – | approximate | centre rounding fixed (see above) |
| Cubic (2) | `cubic` | cubic B-spline | 2 | 2 | faithful | identical to `b_spline` |
| MitchellNetravalli (3) | `mitchell_netravalli` | BC(1/3, 1/3) | 2 | 2 | faithful | |
| CatmullRom (4) | `catmull_rom` | BC(0, 1/2) | 2 | 2 | faithful | |
| Hermite (5) | `hermite` | BC(0, 0) | 2 | 1 | faithful | zero beyond 1 |
| BSpline (6) | `b_spline` | BC(1, 0) | 2 | 2 | faithful | |
| Hann (7) | `hann` | `cos²(πx/4)/4`, abs(x) ≤ 2 | 3 | 2 | faithful | a window used as kernel (all positive, soft) |
| Bicubic (8) | `bicubic` | Keys a = −0.5 | 2 | 2 | faithful | cv2 INTER_CUBIC uses a = −0.75 |
| Hamming (9) | `hamming` | `0.54+0.46cos(πx)`, x < 1 | 2 | 1 | faithful | window as kernel, discontinuous at 1 |
| Hanning (10) | `hanning` | `0.5+0.5cos(πx)`, x < 1 | 2 | 1 | faithful | |
| Blackman (11) | `blackman` | `sinc(x)·blackman_window(x/2)`, x < 2 | 2 | 2 | faithful, **quirk kept** | The window is evaluated on [0, 1) without centring, so the kernel is **ring-shaped** (≈ 0.0003 at the centre, peak at abs(x) = 1). sinc has no π. Weights stay ≥ 0, so normalisation holds and the formula is kept. Output looks soft and double-edged. Candidate for a fix if parity isn't wanted |
| Welch (12) | `welch` | `1-x²`, x < 1 | 2 | 1 | faithful | |
| Quadric (13) | `quadric` | quadratic B-spline | 2 | 1.5 | faithful | |
| Gaussian (14) | `gaussian` | `exp(-x/(2σ²)²)/(√(2π)σ)`, σ = 0.35 | 2 | 2 | faithful, **quirk kept** | Upstream exponent is `-x/0.06` (Laplacian, not `-x²/2σ²`), so the filter is close to nearest-neighbour |
| Sphinx (15) | `sphinx` | `3(sin y − y cos y)/y³`, y = πx | 2 | 2 | faithful | unbounded; truncated at mks |
| Bartlett (16) | `bartlett` | `2x` on [0,1], `2-2x` beyond | 2 | 1 | **approximate (bug fixed)** | Upstream is a Bartlett window on [0, 2] that isn't centred: 0 at the tap centre, jumps 2 → 0 at x = 1, negative beyond. Weight sums are 0, giving black pixels in pic-scale at 2x upscale. Replaced by the centred window `1-abs(x)`, the same as `bilinear` |
| Robidoux (17) | `robidoux` | BC(12/(19+9√2), 13/(58+216√2)) | 2 | 2 | faithful, **quirk kept** | Robidoux's published C is 113/(58+216√2) ≈ 0.311; upstream has 13/... ≈ 0.036. BC-splines stay normalised, so the formula is kept |
| RobidouxSharp (18) | `robidoux_sharp` | BC(6/(13+7√2), 7/(2+12√2)) | 2 | 2 | faithful | |
| Spline16 (19) | `spline16` | Panorama Tools spline | 2 | 2 | faithful | fixed kernel (not scaled), so it **aliases at downscale**, as upstream |
| Spline36 (20) | `spline36` | " | 4 | 3 | faithful | fixed kernel; polynomial is not zero beyond its support 3, so it's cut there |
| Spline64 (21) | `spline64` | " | 6 | 4 | faithful | fixed kernel; cut at support 4 |
| Kaiser (22) | `kaiser` | `I0(6.33·√(1-x²))/I0(6.33)`, x ≤ 1 | 2 | 1 | faithful | `np.i0` instead of the 1e-12 series |
| BartlettHann (23) | `bartlett_hann` | `0.62 − 0.4832·abs(x−0.5) + 0.38cos(2π·abs(x−0.5))`, abs(x) ≤ 2 | 2 | 2 | **approximate (bug fixed)** | Upstream window is centred at 0.5 instead of 0 (0 at the tap centre, −0.48 at 2). For output pixels aligned with an input pixel the weight sum is ≤ 0 (−0.003 in pic-scale's window, −0.97 over abs(x) ≤ 2), so normalisation inverts the kernel. Replaced by the centred Bartlett–Hann window over abs(x) ≤ 2: `fac = abs(x)/4`, canonical 0.48 coefficient |
| Box (24) | `box` | constant 1 | 2 | 0.5 | faithful | Upstream support is the tap window (radius 1 → averages 2·scale pixels; 2 neighbours at upscale). R = 0.5 makes it a true box (= nearest at upscale, pixel-centre box at downscale) |
| Bohman (25) | `bohman` | `(1-abs(x))cos(πabs(x)) + sin(πabs(x))/π` | 2 | 1 | faithful | |
| Lanczos2/3/4/6 (26–28, 46) | `lanczos2` … `lanczos6` | `sinc(πx)·sinc(πx/a)` | a | a | faithful | |
| Lanczos2/3/4/6Jinc (29–31, 47) | `lanczos2_jinc` … | `jinc(πx)·jinc(πx/a)`, abs(x) < a | a | a | **approximate (bug fixed)** | Upstream returns 0 at x = 0 and jinc(0) = 0, which leaves a hole at the centre tap. We use the limit (1/4). Applied separably, like upstream |
| EwaHanning (32) | `ewa_hanning` | kernel `jinc`, window `hanning`, size 2, blur 0 | 1 | 0.5 | approximate (jinc(0)) | `hanning(2x)` is 0 beyond 0.5 → near-nearest at upscale. Upstream returned black pixels when a tap hits dx = 0 |
| EwaRobidoux (33) | `ewa_robidoux` | robidoux + EWA smoothing | 2 | 2 | faithful kernel | smoothing made zero-phase |
| EwaBlackman (34) | `ewa_blackman` | blackman + EWA | 2 | 2 | faithful, quirk kept | ring kernel (see Blackman) |
| EwaQuadric (35) | `ewa_quadric` | quadric + EWA | 2 | 1.5 | faithful kernel | |
| EwaRobidouxSharp (36) | `ewa_robidoux_sharp` | robidoux_sharp + EWA | 2 | 2 | faithful kernel | |
| EwaLanczos3Jinc (37) | `ewa_lanczos3_jinc` | lanczos3_jinc + EWA | 3 | 3 | approximate (jinc(0)) | |
| Ginseng (38) | `ginseng` | kernel `sinc(x)`, window `jinc`, size 3, blur 1 | 3 | 3 | approximate (jinc(0)) | The window `jinc(3x)` carries the lobes (zeros at 1.28, 2.34). The no-π sinc is a broad taper |
| EwaGinseng (39) | `ewa_ginseng` | kernel `sinc`, window `jinc`, size 3.2383, blur 1 | 3 | 3.238 | approximate (jinc(0)) | |
| EwaLanczosSharp (40) | `ewa_lanczos_sharp` | jinc · jinc window, size 3.2383, blur 0.98125 | 3 | 3.178 | approximate (jinc(0)) | lobes from `jinc(3.24x)`: zeros at 1.18, 2.17, 3.14 (×blur), close to true jinc's 1.22, 2.23, 3.24 |
| EwaLanczos4Sharpest (41) | `ewa_lanczos4_sharpest` | jinc · jinc window, size 4.2411, blur 0.88451 | 4 | 3.751 | approximate (jinc(0)) | |
| EwaLanczosSoft (42) | `ewa_lanczos_soft` | jinc · jinc window, size 3.2383, blur 1.01647 | 3 | 3.292 | approximate (jinc(0)) | |
| HaasnSoft (43) | `haasn_soft` | kernel `jinc`, window `hanning`, size 3, blur 1.11 (not EWA) | 3 | 0.37 | faithful kernel | `hanning(3x)` is 0 beyond 1/3, so the support is 1.11/3 px: nearest at upscale (upstream: black pixels when no tap is within 0.37), a narrow Hann at downscale |
| Lagrange2/3 (44–45) | `lagrange2`, `lagrange3` | Lagrange interpolation of order 2·support | 2/3 | 2/3 | faithful | |
| Area (48) | `area` | see above | 2 | – | approximate | = OpenCV INTER_AREA both ways |

The window argument `xk * window_size` (instead of `xk / window_size`) and the missing π in `jinc`/`sinc` look
unintended upstream. They don't break normalisation, though, and the resulting shapes are usable jinc-like
filters, so they are kept.

## Scale colour spaces

aire maps `ScaleColorSpace` values to pic-scale scalers (`aire-filters/src/lib.rs`). Every scaler converts
to the space, resizes in f32 with the same filter, then converts back. Alpha is premultiplied *in the
working space* (pic-scale multiplies e.g. L\*, a\*, b\* by α), filtered, clipped to [0, 1] and
unpremultiplied (α > 1e-6). pic-scale skips premultiplication when alpha is constant 1, which gives the
same result.

| ScaleColorSpace | name | upstream | LazyRoom | status |
|---|---|---|---|---|
| SRGB (0) | `srgb` | `Scaler` on encoded values | identity | faithful |
| LAB (1) | `lab` | `LabScaler`: sRGB TF → XYZ D65 → CIE L\*a\*b\* (ε 0.008856, 7.787, white 95.047/100/108.883) | same constants | faithful |
| LINEAR (2) | `linear` | `LinearScaler` (sRGB TF, 12-bit fixed-point linear) | float32 linear | faithful (more precise) |
| LUV (3) | `luv` | `LuvScaler`: CIE L\*u\*v\* D65 | same formulas | faithful |
| SIGMOIDAL (4) | `sigmoidal` | `1/(1+e^-x)` of the *encoded* values, inverse `ln(y/(1-y))` (0 if y ≥ 1 or y ≤ 0) | same | faithful. On [0, 1] this is nearly linear (range 0.5–0.73), not ImageMagick's contrast sigmoid |
| XYZ (5) | `xyz` | `XYZScaler`: linear (sRGB TF) → XYZ D65 | same | faithful. Mathematically equal to `linear`, since XYZ is a linear map of linear RGB |
| LINEAR_F32_SRGB/REC709/GAMMA_2_2/GAMMA_2_8 (6–9) | `linear_f32_*` | aire uses `LinearScaler::new_with_transfer` (the 12-bit scaler, despite the "F32" name) with sRGB / Rec.709 / pure 2.2 / pure 2.8 | float32 linear with that TF | faithful |
| LCH (10) | `lch` | `LChScaler`: CIE **LCh(uv)** (from Luv). Hue in radians is filtered as a plain scalar | same; hue wrap-around artefacts at ±π are upstream behaviour | faithful |
| OKLAB_* (11–14) | `oklab_srgb/rec709/gamma_2_2/gamma_2_8` | `OklabScaler(TF)`: linearise with TF → Ottosson Oklab | same matrices | faithful |
| JZAZBZ_* (15–18) | `jzazbz_*` | `JzazbzScaler(200 nits, TF)`: linearise → XYZ D65 × 200 → Jzazbz (colorutils' combined LMS matrix, PQ p = 134.034375, d0 = 1.6295e-11) → inverse ÷ 200 | same constants, computed in float64 | faithful |

Deviations shared by all spaces:

* colorutils clamps transfer functions to [0, 1] and Luv L to [0, 100]. For its 8-bit input those are
  no-ops. We drop the clamps and mirror the sign through power functions, so float/HDR values and filter
  overshoot survive (export clamps later). The PQ steps keep upstream's `x <= 0 → 0` and `NaN → 0` guards.
* Pure-gamma decoding (`x^(1/2.2)`, `x^(1/2.8)`) has infinite slope at 0. With Oklab/Jzazbz, float32
  noise on channels that should be exactly 0 can come back as up to ≈ 2/255. Upstream's f32 path has the
  same property.

## `geometry.resize` wiring

New parameter `colorspace` (choices = `SCALE_COLOR_SPACES`, default `srgb`). Routing in `resize_buffer`:

* `srgb`: unchanged. `auto`/`nearest`/`bilinear`/`bicubic`/`lanczos`/`area` use OpenCV; every other name
  uses `resample()`.
* `linear`, `linear_f32_*`: OpenCV names use the OpenCV kernel on linear light (`encode_space` →
  premultiply → `cv2.resize` → `decode_space`). Other names use `resample()`.
* All other spaces: every name goes through `resample()`. OpenCV names map to the pic-scale versions:
  `nearest` (= INTER_NEAREST_EXACT), `bilinear` (scaled tent, antialiased at downscale), `bicubic`
  (Keys a = −0.5), `lanczos` → `lanczos4` (`ALIASES`), `area` (= INTER_AREA). `auto` still picks
  area or lanczos.

## Performance

Measured on the dev box with load average ≈ 10 (other ports running in parallel). An earlier, less-loaded
prototype of the same sparse passes ran the 12 MP → 3 MP Lanczos3 case in ≈ 0.9 s.

| case | time |
|---|---|
| 4000×3000 → 2000×1500 `lanczos3`, srgb, RGB | 1.2–1.5 s |
| same, RGBA | ≈ 2.3 s |
| same, `oklab_srgb` / `lab` / `jzazbz_srgb` | ≈ 1.7 s / 3.7 s / 5 s |
| 2000×1500 → 4000×3000 `lanczos3` | ≈ 1.6 s |
| 3000×2000 → 1500×1000 `lanczos3` (test) | ≈ 0.6–1.1 s |

The filtering passes are scipy CSR × dense matrix products. Colour conversions and (un)premultiplication
run per pixel over row blocks in a thread pool. scipy's sparse product holds the GIL, so threading it gave
nothing.

## Tests (`tests/test_resample.py`)

* Tables equal the aire enums, in order.
* Every kernel down- and upscales RGBA without NaNs; alpha stays in [0, 1].
* Constant images (RGB and RGBA) stay constant for every kernel, including a 0.8 scale with output centres
  exactly between inputs (exercises the nearest fallback).
* Weight matrix rows sum to 1 for tiny and odd sizes.
* Every kernel commutes with horizontal and vertical flips (symmetric windows, zero-phase EWA).
* 2x up/down round trip stays close (< 0.012 mean) for 24 smooth kernels.
* `area` / `nearest` match OpenCV INTER_AREA / INTER_NEAREST_EXACT.
* Spot values of ported kernels, including the fixed bugs and the kept Blackman quirk.
* Every colour space: encode/decode round trip, resampling RGBA with 4 kernels, constant preservation.
* `resize_buffer` routing for srgb / linear / other spaces, and the `geometry.resize` op with the new param.
* Speed: 3000×2000 → 1500×1000 `lanczos3` in < 3 s. It runs in about 1 s, so it isn't marked `slow`.
