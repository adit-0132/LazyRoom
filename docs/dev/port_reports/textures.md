# Port report: ImageToolbox "Texture Generation" (177 presets)

Every entry of ImageToolbox's `TextureFilterType` enum is ported. That covers 122 fast-noise textures, 28 GLSL patterns, 15 ray-marched 3D shapes, 9 JH Labs filters and 3 G'MIC commands.

| Area | Location |
|---|---|
| Preset registry and API | `src/lazyroom/textures/` (`render`, `list_presets`, `get_preset`) |
| Tools | `texture`, `texture_presets` in `src/lazyroom/tools/textures.py` |
| Operation | `generate.texture` in `src/lazyroom/ops/texture_layer.py` |
| Tests | `tests/test_textures.py` |

## Backends and how each was verified

| Backend | Upstream implementation | Port | Verification |
|---|---|---|---|
| `fast-noise` (122) | ImageToolboxLibs `fast-noise/cpp/textures/ProceduralTextures.cpp` on FastNoiseLite 1.1.1 (MIT) | numba, line-by-line (`fastnoise_lite.py`, `_noisebank.py`, `_texmath.py`, `_procedural_*.py`) | The upstream C++ was compiled with g++ (`-ffp-contract=off`) into a ctypes harness that runs the same loops as the JNI entry. Every type was compared pixel by pixel with ImageToolbox defaults, then with non-default params, other seeds and non-square sizes. |
| `pattern` (28) | `PatternTextureShaders.kt` (GLSL via GPUImage) | numba kernels (`_patterns_impl.py`) | No GPU reference was available. GLSL formulas were hand-evaluated at spot pixels, and contact sheets were reviewed for orientation and appearance. |
| `raymarch` (15) | `RaymarchTextureGenerator.kt` (GLSL via GPUImage) | numba kernel (`_raymarch_impl.py`) | Visual review, plus an equirectangular environment-image test. No GPU reference was available. |
| `jhlabs` (9) | ImageToolboxLibs `libs/jhlabs` **Rust** re-implementation (`lib.rs`), which is what ImageToolbox actually runs. It is not the original Java JH Labs code. | numba (`_jhlabs_impl.py`), with f32 maths and Rust/Java cast semantics | No Rust toolchain was available. Each kernel was checked against an independent Python transcription of the Rust: 0 differences on Perlin `noise3`, Cellular, `box_blur_exact` and Quilt. Java `Random(0)` reproduces its known values. |
| `gmic` (3) | ImageToolboxLibs `gmic/filters/TextureFilters.kt` | Exact command strings via `run_gmic`, on a black canvas | Runs with the dev G'MIC CLI and is deterministic (`srand <seed>`, single-threaded). |

FastNoiseLite itself (`fastnoise_lite.py`, 2D only: all six noise types, the FBm/Ridged/PingPong fractals, and all cellular distance functions and return types) was checked on 20k random points over every configuration. Values are within 1e-3 of the compiled header. The only exception is the CellValue return type, where 0.005–0.01% of points sit on a nearest-cell tie and resolve differently in float32 and float64. Domain warp and 3D noise are not ported, because no texture uses them.

## Common notes

- **Seed.** `seed=0` reproduces ImageToolbox's default output; other values change it as follows:
  - fast-noise: noise seed = 1337 + seed.
  - Patterns that have a seed parameter: seed % 10001.
  - jhlabs: Brushed Metal uses `java.util.Random(seed)`, Plasma's seed field becomes 567 + seed, and the Perlin permutation seed becomes `0x4a484c616273 + seed`.
  - G'MIC: `srand <seed>`. Upstream G'MIC textures are unseeded, so they are not reproducible there.
  - Presets with no randomness upstream ignore the seed.
- **Precision.**
  - The fast-noise, pattern and raymarch ports run in float64; upstream uses float32 (C++ `float` or GPU highp).
  - Where float64 visibly changed results, the fast-noise kernels evaluate in float32 like upstream. This applies to the `textures::hash` speckle, the fractal escape-time loops, and the lattices that feed `floor` or truncation.
  - The GLSL `fract(sin(x)*43758.5453)` hash cannot match any particular GPU bit for bit, because GPU `sin` precision is implementation-defined. The same caveat applies to an arm64 build of the C++, which may use fused multiply-add.
- **Parameters.**
  - Parameters mirror the ImageToolbox UI: same sliders, ranges, defaults (`withDefaultsFor` / `FastNoiseTextureParams.defaultFor`) and English titles.
  - Values the UI hides are fixed at their upstream defaults; for example, Ringed Planet's scale and pattern colours beyond `colorCount`.
  - Out-of-range options are rejected (`INVALID_PARAMS`), whereas ImageToolbox clamps them.
- **Output.** Output is straight-alpha RGBA. The alpha channel is dropped when the result is fully opaque.
- **`generate.texture`.**
  - It renders the texture at the full-resolution-equivalent size (buffer / render scale / `scale`) and resamples it, premultiplied, to the buffer. This keeps preview proxies consistent with exports.
  - The render is capped at 64 MP.
  - The texture is composited with `layer_over`, so transparent texture areas leave the image unchanged.
- **Performance.**
  - Every non-G'MIC preset renders 1024x1024 in under 1.2 s once the JIT cache is warm.
  - The very first run compiles the numba kernels, which takes 5–30 s per kernel module. They are cached on disk in LazyRoom's numba cache.
  - G'MIC Organic Fibers takes about 7.4 s at 1024².

## Presets

| Preset | id | backend | status | notes |
|---|---|---|---|---|
| Lava | lava | fast-noise (C++->numba) | faithful | vs compiled upstream C++: 99.96% identical, 100% ≤1 LSB |
| Cloud | cloud | fast-noise (C++->numba) | faithful | 99.99% identical, 100% ≤1 LSB vs upstream C++ |
| Smoke | smoke | fast-noise (C++->numba) | faithful | 99.97% identical, 100% ≤1 LSB vs upstream C++ |
| Stone | stone | fast-noise (C++->numba) | faithful | 99.94% identical, 100% ≤1 LSB vs upstream C++ |
| Advanced Wood | advanced_wood | fast-noise (C++->numba) | faithful | 99.83% identical, 100% ≤1 LSB vs upstream C++ |
| Camouflage | camouflage | fast-noise (C++->numba) | faithful | 99.99% identical, 100% ≤1 LSB vs upstream C++ |
| Paper | paper | fast-noise (C++->numba) | faithful | 99.99% identical, 100% ≤1 LSB vs upstream C++ |
| Rust | rust | fast-noise (C++->numba) | faithful | 99.99% identical, 100% ≤1 LSB vs upstream C++ |
| Fabric | fabric | fast-noise (C++->numba) | faithful | weave-cell parity evaluated in float32 like upstream; 99.91% identical, 100% ≤1 LSB vs upstream C++ |
| Topography | topography | fast-noise (C++->numba) | faithful | 99.79% identical, 100% ≤1 LSB vs upstream C++ |
| Cell | cell | fast-noise (C++->numba) | faithful | 99.99% identical, 100% ≤1 LSB vs upstream C++ |
| Crack | crack | fast-noise (C++->numba) | faithful | 99.93% identical, 100% ≤1 LSB vs upstream C++ |
| Water Ripple | water_ripple | fast-noise (C++->numba) | faithful | 99.93% identical, 100% ≤1 LSB vs upstream C++ |
| Foliage | foliage | fast-noise (C++->numba) | faithful | 99.98% identical, 100% ≤1 LSB vs upstream C++ |
| Brick | brick | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ |
| Terrain | terrain | fast-noise (C++->numba) | faithful | 99.97% identical, 100% ≤1 LSB vs upstream C++ |
| Ice | ice | fast-noise (C++->numba) | faithful | 98.6-99.2% ≤1 LSB vs upstream C++; only sparkle specks differ (float32 hash at noise-warped coordinates is chaotic under 1e-7 coordinate differences; sparkle=0 gives 100% ≤1 LSB) |
| Sand | sand | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ |
| Nebula | nebula | fast-noise (C++->numba) | faithful | 99.2% ≤1 LSB vs upstream C++; only star positions differ (float32 hash at warped coordinates; stars=0 gives 100% ≤1 LSB) |
| Honeycomb | honeycomb | fast-noise (C++->numba) | faithful | 99.06% identical, 100% ≤1 LSB vs upstream C++ |
| Grass | grass | fast-noise (C++->numba) | faithful | 97.6% identical, 100% ≤1 LSB vs upstream C++ |
| Dirt | dirt | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ (float32 products emulated for the pebble hash) |
| Leather | leather | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ |
| Concrete | concrete | fast-noise (C++->numba) | faithful | 99.4% ≤1 LSB vs upstream C++; only aggregate specks differ (float32 hash at warped coordinates; aggregate=0 gives 100% ≤1 LSB) |
| Asphalt | asphalt | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ |
| Moss | moss | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ |
| Fire | fire | fast-noise (C++->numba) | faithful | 99.97% identical, 100% ≤1 LSB vs upstream C++ |
| Aurora | aurora | fast-noise (C++->numba) | faithful | 99.8-99.95% ≤1 LSB vs upstream C++; only star positions differ (float32 hash at warped coordinates; stars=0 gives 100% ≤1 LSB) |
| Oil Slick | oil_slick | fast-noise (C++->numba) | faithful | 99.68% identical, 100% ≤1 LSB at defaults; a handful of fract() phase-wrap pixels (<0.005%) can differ strongly with other params |
| Watercolor | watercolor | fast-noise (C++->numba) | faithful | same algorithm, but the largest measured deviation of any preset: 79.4-91.7% ≤1 LSB, max diff 4-6/255 vs upstream C++, because the paper-fibre layer samples the float32 hash at warped coordinates (chaotic); paper amount=0 gives 100% ≤1 LSB |
| Abstract Flow | abstract_flow | fast-noise (C++->numba) | faithful | 99.82% identical, 100% ≤1 LSB vs upstream C++ |
| Opal | opal | fast-noise (C++->numba) | faithful | 99.96% identical, 100% ≤1 LSB at defaults; a few fract() phase-wrap pixels can differ with other params |
| Damascus Steel | damascus_steel | fast-noise (C++->numba) | faithful | 99.91% identical, 100% ≤1 LSB vs upstream C++ |
| Lightning | lightning | fast-noise (C++->numba) | faithful | 99.64% identical, 100% ≤1 LSB vs upstream C++ |
| Velvet | velvet | fast-noise (C++->numba) | faithful | 99.92% identical, 100% ≤1 LSB vs upstream C++ |
| Ink Marbling | ink_marbling | fast-noise (C++->numba) | faithful | 99.82% identical, 100% ≤1 LSB vs upstream C++ |
| Holographic Foil | holographic_foil | fast-noise (C++->numba) | faithful | 98.7-99.66% identical, 100% ≤1 LSB at defaults; a few fract() phase-wrap pixels can differ with other params |
| Bioluminescence | bioluminescence | fast-noise (C++->numba) | faithful | 99.84% identical, 100% ≤1 LSB vs upstream C++ |
| Cosmic Vortex | cosmic_vortex | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ |
| Lava Lamp | lava_lamp | fast-noise (C++->numba) | faithful | 98.5% identical, 100% ≤1 LSB vs upstream C++ |
| Event Horizon | event_horizon | fast-noise (C++->numba) | faithful | 99.95% identical, 100% ≤1 LSB vs upstream C++ |
| Fractal Bloom | fractal_bloom | fast-noise (C++->numba) | faithful | 99.99% identical, 100% ≤1 LSB vs upstream C++ |
| Chromatic Tunnel | chromatic_tunnel | fast-noise (C++->numba) | faithful | 99.94% identical, 100% ≤1 LSB vs upstream C++ |
| Eclipse Corona | eclipse_corona | fast-noise (C++->numba) | faithful | 99.99% identical, 100% ≤1 LSB vs upstream C++ |
| Strange Attractor | strange_attractor | fast-noise (C++->numba) | faithful | pixel-independent orbit samples tabulated once per image (identical values, ~8x faster); 99.92% identical, 100% ≤1 LSB vs upstream C++ |
| Ferrofluid Crown | ferrofluid_crown | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ |
| Supernova | supernova | fast-noise (C++->numba) | faithful | 99.99% identical, 100% ≤1 LSB vs upstream C++ |
| Iris | iris | fast-noise (C++->numba) | faithful | 99.99% identical, 100% ≤1 LSB vs upstream C++ |
| Peacock Feather | peacock_feather | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ |
| Nautilus Shell | nautilus_shell | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++ |
| Ringed Planet | ringed_planet | fast-noise (C++->numba) | faithful | 100% identical vs upstream C++; scale fixed at the default (not shown in the ImageToolbox UI) |
| Geode / Agate | geode | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs: defaults; random non-default params, seed 42, 200x300; seed -77, 300x170): 99.78-99.81% pixels identical, all within 1/255 |
| Prismatic Light | prismatic_light | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.93-99.98% identical, all within 1/255; no seed dependence (none upstream) |
| Stained Glass | stained_glass | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.79-99.90% identical, >=99.99% within 1/255; isolated cell-boundary pixels up to 60/255 (float64 vs float32) |
| Kelp Forest | kelp_forest | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.81-99.98% identical, all within 1/255 |
| Frost Fern | frost_fern | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.58-99.90% identical, all within 1/255 |
| Liquid Crystal | liquid_crystal | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.80-99.95% identical, >=99.99% within 1/255; isolated fract-wrap pixels up to 246/255 (float64 vs float32) |
| Dragon Scales | dragon_scales | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.98-100.00% identical, all within 1/255 |
| Firefly Swarm | firefly_swarm | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.98-99.99% identical, all within 1/255 |
| Mycelium | mycelium | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 98.73-99.02% identical, all within 1/255 |
| Kintsugi | kintsugi | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.77-99.98% identical, all within 1/255 |
| Carbon Fiber | carbon_fiber | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.91-99.95% identical, all within 1/255 |
| Circuit Board | circuit_board | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.99-100.00% identical, all within 1/255 |
| Soap Film | soap_film | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 97.37-98.40% identical, all within 1/255 |
| Moiré Guilloché | moire_guilloche | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.68-99.86% identical, all within 1/255; no seed dependence (none upstream) |
| Snake Skin | snake_skin | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.89-99.91% identical, all within 1/255 |
| Terrazzo | terrazzo | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.99% identical, >=99.99% within 1/255; isolated chip-threshold pixels up to 167/255 (float64 vs float32) |
| Galaxy Filaments | galaxy_filaments | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.58-99.89% identical, all within 1/255 |
| Volcanic Obsidian | volcanic_obsidian | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.95-99.97% identical, all within 1/255 |
| Motherboard Heatmap | motherboard_heatmap | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 100.00% identical (rounded), all within 1/255 |
| Microscopic Diatoms | microscopic_diatoms | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.99% identical, all within 1/255; no seed dependence (none upstream) |
| Reaction Diffusion | reaction_diffusion | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.17-99.45% identical, all within 1/255 |
| Coral Growth | coral_growth | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.98-99.99% identical, all within 1/255 |
| Slime Mold | slime_mold | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.44-99.88% identical, >=99.95% within 1/255, max 8/255 on isolated pixels |
| Dendritic Crystal | dendritic_crystal | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.89-99.91% identical, all within 1/255 |
| Electric Arc Field | electric_arc_field | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.99-100.00% identical, all within 1/255 |
| Cloud Chamber | cloud_chamber | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.95-99.98% identical, all within 1/255 |
| Turbulent Ink | turbulent_ink | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.13-99.42% identical, all within 1/255 |
| Cellular Embryo | cellular_embryo | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.99-100.00% identical, all within 1/255 |
| Neural Garden | neural_garden | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.99% identical, all within 1/255 |
| Magnetic Field | magnetic_field | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.71-99.94% identical, all within 1/255 |
| River Delta | river_delta | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.52-99.63% identical, >=99.99% within 1/255, max 3/255 |
| Lichen Colony | lichen_colony | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.76-99.87% identical, all within 1/255 |
| Bacterial Culture | bacterial_culture | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.97-100.00% identical, all within 1/255 |
| Fluid Vorticity | fluid_vorticity | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.92-99.94% identical, all within 1/255 |
| Crystal Growth | crystal_growth | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.93-99.98% identical, all within 1/255 |
| Galactic Web | galactic_web | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.99-100.00% identical, all within 1/255 |
| Veined Leaf | veined_leaf | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.89-99.97% identical, all within 1/255 |
| Porous Sponge | porous_sponge | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.55-99.93% identical, all within 1/255 |
| Rain On Glass | rain_on_glass | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.99-100.00% identical, all within 1/255 |
| Ember Field | ember_field | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.80-99.95% identical, all within 1/255 |
| Quantum Foam | quantum_foam | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.68-99.84% identical, all within 1/255 |
| Chladni Plate | chladni_plate | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.91-99.99% identical, all within 1/255; no seed dependence (none upstream) |
| Cymatic Rosette | cymatic_rosette | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.88-99.89% identical, all within 1/255; no seed dependence (none upstream) |
| Lichtenberg Figure | lichtenberg_figure | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.41-99.63% identical, >=99.95% within 1/255, max 5/255 |
| Quasicrystal | quasicrystal | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.90-99.96% identical, all within 1/255; no seed dependence (none upstream) |
| Mandelbrot | mandelbrot | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 100% identical; escape-time loop evaluated in float32 like upstream; no seed dependence (none upstream) |
| Burning Ship | burning_ship | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 100% identical; escape-time loop evaluated in float32 like upstream; no seed dependence (none upstream) |
| Julia Set | julia_set | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 100% identical; escape-time loop evaluated in float32 like upstream; no seed dependence (none upstream) |
| Kaleidoscope Crystal | kaleidoscope_crystal | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.61-99.66% identical, >=99.97% within 1/255, max 3/255; no seed dependence (none upstream) |
| Spectral Prism | spectral_prism | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.85-99.95% identical, all within 1/255; no seed dependence (none upstream) |
| Topological Knot | topological_knot | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.99-100.00% identical, all within 1/255; no seed dependence (none upstream) |
| X-Ray Botanical | x_ray_botanical | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.98-100.00% identical, all within 1/255 |
| Chromatophore | chromatophore | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.90-99.97% identical, all within 1/255 |
| Biomechanical Tissue | biomechanical_tissue | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.84-99.95% identical, all within 1/255 |
| Gilded Filigree | gilded_filigree | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.82-99.94% identical, all within 1/255; no seed dependence (none upstream) |
| Ancient Runes | ancient_runes | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 100% identical |
| Solar Granulation | solar_granulation | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 97.36-97.74% identical, >=99.99% within 1/255, max 2/255 |
| Lunar Ejecta | lunar_ejecta | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.87-99.89% identical, >=99.99% within 1/255, max 2/255 |
| Ocean Currents | ocean_currents | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.91-99.97% identical, all within 1/255 |
| Ink Wash Mountains | ink_wash_mountains | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 100.00% identical (rounded), all within 1/255 |
| Neon City | neon_city | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.97-99.98% identical, all within 1/255; window-column lattice evaluated in float32 like upstream |
| Phyllotaxis Bloom | phyllotaxis_bloom | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 100% identical; seed lattice evaluated in float32 like upstream; no seed dependence (none upstream) |
| Sierpiński Triangle | sierpinski_triangle | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 100% identical; no seed dependence (none upstream) |
| Apollonian Gasket | apollonian_gasket | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.78-99.99% identical, >=99.99% within 1/255, max 2/255; no seed dependence (none upstream) |
| Hyperbolic Tiling | hyperbolic_tiling | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.76-99.80% identical, all within 1/255; seed only rotates the tiling by ~0.0017 rad per step |
| Möbius Weave | moebius_weave | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.96-99.99% identical, all within 1/255; no seed dependence (none upstream) |
| Rorschach Inkblot | rorschach_inkblot | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.98% identical, all within 1/255 |
| Seismic Interference | seismic_interference | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.91-99.96% identical, all within 1/255 |
| Rayleigh–Bénard Convection | rayleigh_benard | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.45-99.89% identical, all within 1/255 |
| Origami Facets | origami_facets | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.90-99.96% identical, all within 1/255; no seed dependence (none upstream) |
| Fiber Optic Bundle | fiber_optic_bundle | fast-noise (C++->numba) | faithful | vs compiled upstream C++ (3 configs): 99.91-99.94% identical, all within 1/255; no seed dependence (none upstream) |
| Random triangles | random_triangles | pattern (GLSL->numba) | faithful | triangle vertices precomputed once per image (same values as per-pixel); seed from render seed; GPU sin-hash precision caveat |
| Hex cubes | hex_cube_pattern | pattern (GLSL->numba) | faithful | |
| Square looper | square_looper | pattern (GLSL->numba) | faithful | |
| Moire interference | moire_interference | pattern (GLSL->numba) | faithful | |
| Truchet arcs | truchet_arcs | pattern (GLSL->numba) | faithful | seed from render seed; GPU sin-hash precision caveat |
| Circuit pattern | circuit_pattern | pattern (GLSL->numba) | faithful | seed from render seed; GPU sin-hash precision caveat |
| Rosaces | rosace_pattern | pattern (GLSL->numba) | faithful | |
| Rosette | rosette_pattern | pattern (GLSL->numba) | faithful | |
| Light array | light_array | pattern (GLSL->numba) | faithful | seed from render seed; GPU sin-hash precision caveat |
| Voronoi gems | voronoi_gems | pattern (GLSL->numba) | faithful | seed from render seed; GPU sin-hash precision caveat |
| City skyline | city_skyline | pattern (GLSL->numba) | faithful | seed from render seed; GPU sin-hash precision caveat |
| Winter landscape | winter_landscape | pattern (GLSL->numba) | faithful | tree positions precomputed once per image (same values); seed from render seed; GPU sin-hash precision caveat |
| Circle ripple illusion | circle_ripple_pattern | pattern (GLSL->numba) | faithful | |
| Square ripple illusion | square_ripple_pattern | pattern (GLSL->numba) | faithful | |
| Hatch | hatch_pattern | pattern (GLSL->numba) | faithful | |
| Smooth hatch | smooth_hatch | pattern (GLSL->numba) | faithful | |
| XOR pattern | xor_pattern | pattern (GLSL->numba) | faithful | shader loop is capped at 12 iterations although intensity allows 16 (same as upstream) |
| Square spiral | square_spiral_pattern | pattern (GLSL->numba) | faithful | |
| Cross stitch | cross_stitch_pattern | pattern (GLSL->numba) | faithful | |
| Plaid | plaid_pattern | pattern (GLSL->numba) | faithful | seed from render seed; GPU sin-hash precision caveat |
| Generated waves | generated_waves | pattern (GLSL->numba) | faithful | seed from render seed; GPU sin-hash precision caveat |
| Voronoi hatch | voronoi_hatch | pattern (GLSL->numba) | faithful | seed from render seed; GPU sin-hash precision caveat |
| Loops | loop_pattern | pattern (GLSL->numba) | faithful | loop centres precomputed once per image (same values); seed from render seed; GPU sin-hash precision caveat |
| Scintillating grid | scintillating_grid | pattern (GLSL->numba) | faithful | |
| Circle spiral | circle_spiral_pattern | pattern (GLSL->numba) | faithful | |
| Test chart | test_chart | pattern (GLSL->numba) | faithful | no shape parameters upstream; HSV quadrant early return ported |
| Bent rows | bent_rows | pattern (GLSL->numba) | faithful | |
| Diamonds illusion | diamonds_illusion | pattern (GLSL->numba) | faithful | |
| 3D Sphere | sphere_3d | raymarch (GLSL->numba) | faithful | float64 (GPU highp float32); no randomness, seed ignored; optional environment image only downscaled when > 2048 px (ImageToolbox's loader may also resample smaller ones) |
| 3D Cube | cube_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere |
| Cube Stair | cube_stair_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere |
| Cube Spheres | cube_spheres_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere |
| Emptied Cube | emptied_cube_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere |
| 3D Torus | torus_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere |
| Möbius Torus | mobius_torus_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere |
| Gyro Rings | gyro_rings_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere |
| Torus Gem | torus_gem_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere |
| Cube Octahedron | cube_octahedron_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere |
| Infinite Spheroids | infinite_spheroids_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere; at defaults the camera sits inside the 2.4 bounding box (upstream behaviour) |
| Infinite Structure | infinite_structure_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere; fills the frame at defaults (upstream behaviour), so output is RGB |
| Infinite Cylinders | infinite_cylinders_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere; camera inside the 2.4 bounding box at defaults (upstream behaviour) |
| 3D Truchet Rings | truchet_rings_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Sphere; per-cell fract(sin(dot)) hash evaluated in float32, tile orientations may differ from a given GPU |
| 3D Truchet Pipes | truchet_pipes_3d | raymarch (GLSL->numba) | faithful | same notes as 3D Truchet Rings |
| Brushed Metal | brushed_metal | jhlabs (Rust->numba) | faithful | java.util.Random(seed) noise + sine highlight, then box_blur_exact incl. its upstream bug of overwriting the first/last rows and columns with corner pixels when radius > 0 (reproduced) |
| Caustics | caustics | jhlabs (Rust->numba) | faithful | faithful to what ImageToolbox runs: the Rust version is ridged 3D Perlin turbulence, not the original ray-traced JH Labs caustics; `dispersion` (and hidden `samples`) have no effect upstream; f32 mix gives alpha 254 on some pixels, as upstream |
| Cellular | cellular | jhlabs (Rust->numba) | faithful | useColor=true on a `color`-filled bitmap: mix(black, color, smoothstep(F2, F1, (d2-d1)/(d2+d1))); amount, angle/gradient coefficients, gain and bias have no visible effect upstream; angle used as radians although the UI range is ±360; seed only matters with randomness > 0 on hexagonal/octagonal/triangular grids |
| Checkerboard | check | jhlabs (Rust->numba) | faithful | fuzziness/100, angle used as radians; no randomness, seed ignored |
| fBm | fbm | jhlabs (Rust->numba) | faithful | Rust FBMFilter: abs 3D Perlin turbulence with octaves fixed at 4 (the Java class has no getTurbulence), GrayscaleColormap, mixed with the transparent source by `amount` (alpha < 1 when amount < 1); hidden octaves/H/lacunarity/gain/bias are ignored upstream; angle as radians |
| Marble | marble | jhlabs (Rust->numba) | faithful | triangle(nx + turbulence_factor * turbulence3) through LinearColormap(black, white); angle as radians; seed offsets the Perlin permutation |
| Plasma | plasma | jhlabs (Rust->numba) | faithful | faithful to what ImageToolbox runs: the Rust version is Perlin turbulence (phase = 567e-4), not the original diamond-square plasma; useImageColors mix with `color` |
| Quilt | quilt | jhlabs (Rust->numba) | faithful | chaotic attractor hit counts through LinearColormap (f32 like Rust); exact hit positions depend on the platform's sinf rounding (same attractor and density, not bit-identical to a given device); seed ignored |
| Wood | wood | jhlabs (Rust->numba) | faithful | triangle(sqrt(nx²+ny²) + rings * turbulence3) through LinearColormap(#E5C494, #987B51); gain/bias have no effect upstream; angle as radians; default rings=2 per withDefaultsFor |
| Organic Fibers | organic_fibers | gmic (G'MIC CLI) | faithful | exact `fx_organic_fibers` command on a black canvas, RGB processed (Preserve alpha mode); `srand <seed>` + single-threaded for determinism (upstream unseeded); ~7.4 s at 1024² |
| Reaction Diffusion (G'MIC) | gmic_reaction_diffusion | gmic (G'MIC CLI) | faithful | exact `fx_reaction_diffusion` command on a black RGBA canvas, G'MIC alpha kept (gmicProcessAlpha); title suffixed "(G'MIC)" to distinguish it from the fast-noise preset (upstream uses the same title for both) |
| Truchet | truchet | gmic (G'MIC CLI) | faithful | exact `fx_truchet` command on a black RGBA canvas, G'MIC alpha kept (transparent colour modes return alpha); `srand <seed>` |
