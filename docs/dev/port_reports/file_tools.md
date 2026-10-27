# Port report: ImageToolbox file utilities (cipher, archives, audio cover, code preview, screenshot framing, document scanner)

Module: `src/lazyroom/tools/file_tools.py` (10 tools) with private helpers
`_cipher.py`, `_archive.py`, `_audio_cover.py`, `_code_preview.py`, `_code_themes.py`
(generated data) and `_docscan.py`.  Tests: `tests/test_file_tools.py` (154 tests).
No core file was changed (see "Secrets" below for why none was needed).

## Sources actually read

* `feature/cipher`: `CryptographyManager.kt`, `AndroidCryptographyManager.kt`,
  `AndroidRandomStringGenerator.kt`, `CipherComponent.kt`, `CipherControls.kt` (output names,
  "already encrypted" warning), `androidTest/.../AndroidCryptographyManagerTest.kt`;
  `core/domain/.../CipherType.kt` (BROKEN list, entries), `SecureAlgorithmsMapping.kt` (OID names),
  `core/utils/.../EncryptedFileUtils.kt` (magic), `app/.../RegisterSecurityProviders.kt`
  (provider registration, extra excludes), `strings.xml` (error texts).
* `feature/archive-tools` (all of `domain/`, `data/AndroidArchiveManager.kt`, `ArchiveToolsComponent.kt`)
  and `lib/archive` (`ArchiveEngine.kt`, `ArchivePath.kt`, `ArchiveFormat.kt`,
  `ArchiveCompressionMethod.kt`, `ArchiveCompressionLevel.kt`, `ExtractionLimits`).
* `feature/audio-cover-extractor` (`AndroidAudioCoverRetriever.kt`, component, result model).
* `feature/code-preview`: `CodePreviewBitmapRenderer.kt`, `CodeHighlight.kt`, `CodeLanguage.kt`,
  `CodePreviewTheme.kt`, `CodePreviewParams.kt`, `CodePreviewComponent.kt` (ranges), the 256 theme
  CSS assets and `CodePreviewBitmapRendererTest.kt`; `core/ui/.../GradientBackgroundPreset.kt`.
* `feature/screenshot-framing`: renderer, params, image loader, both androidTests, `DomainAspectRatio.kt`.
* `feature/document-scanner` (`DocumentScannerComponent.kt`, ML Kit `ScanResult` usage) and
  `PdfCreationParams.kt`.

## Status

| ImageToolbox feature | LazyRoom tool | Status | Notes |
|---|---|---|---|
| Cipher: encrypt (versioned `ITBXCRYP` format) | `encrypt_file` | faithful | Byte-identical format: header, PBKDF2-HMAC-SHA256 (210,000 it., 16-byte salt, 128/256-bit key by the MD5List rule), nonce size rule (`ivSize`, 12 for GCM), header as AAD for GCM/CCM/Poly1305, PBE ciphers keyed with the hex PBKDF2 output. Verified both ways against the real BouncyCastle provider (see Tests). Default output `enc-<name>`; warns when the input is already encrypted. |
| Cipher: decrypt (versioned + legacy) | `decrypt_file` | faithful | Versioned files: algorithm read from the header, same header validation as `EncryptedFileHeader.decode`, ImageToolbox's error text. Legacy (headerless) files: key = MD5/SHA-256 of the password, IV = first block of the key, PBE legacy salt/iterations = encoded password, exactly as `decryptLegacy`. Default output `dec-<name>`. |
| Cipher algorithm list (100+ BouncyCastle/Conscrypt transformations) | `cipher_algorithms` | approximate list, faithful behaviour | The list is device-dependent (`Security.getAlgorithms("Cipher")` filtered by `BROKEN`); reconstructed from BouncyCastle 1.77 + Conscrypt with ImageToolbox's own filter/OID-mapping/sort/dedupe logic: **79 entries, 34 supported**, 45 unsupported with reasons (table below). |
| `generateRandomString(18)` | `generate_password` | faithful | CSPRNG over `[0-9A-Za-z]`, default length 18. |
| Archive: create (ZIP; Store/Deflate/BZIP2/LZMA/XZ/ZSTD; level Fast/Normal/Max; AES-256 password) | `archive_create` | partial | ZIP with Store/Deflate/BZIP2/LZMA, levels 1/6/9, flat entries named after the files with ImageToolbox's case-insensitive `name (2).ext` de-duplication, mode 0644. **XZ** (method 95) not implemented (zipfile lacks it); **ZSTD** only when the interpreter's zipfile has `ZIP_ZSTANDARD` (Python 3.14+); **password (WinZip AES-256)** not implemented (no permissive AES-ZIP library installed) - clear `BACKEND_UNAVAILABLE` errors. |
| Archive: list + encryption status + passphrase check | `archive_list` | faithful (ZIP) | `safeSegments`-normalised, de-duplicated entries; status none / password_required / unsupported (AES); `password_ok` like `verifyArchivePassphrase`. Unsafe/special entries are reported in `skipped_entries` (ImageToolbox silently drops them). |
| Archive: extract (create subfolder, preserve structure, skip hidden, selected entries, ZipCrypto/AES passwords) | `archive_extract` | faithful (ZIP) + hardened | Same options and semantics (unique `name (2)` folders/files, selection rules for directories, rollback of everything created on failure, whole extraction refused on an unsafe path). ZipCrypto extraction supported; WinZip AES not implemented. tar/7z/rar/cpio/iso/ar/raw-compressor formats of libarchive are **not ported** (ZIP containers such as .cbz/.jar/.apk/.epub work). |
| Audio cover extractor (`MediaMetadataRetriever.embeddedPicture ?: frameAtTime`, saved as `<name>.png`) | `audio_cover_extract` | faithful (own parsers) | MP3 ID3v2.2/2.3/2.4 (APIC/PIC, tag + frame unsynchronisation, extended headers, v2.3/2.4 zlib frames, data-length indicator), FLAC PICTURE (also after a leading ID3 tag), MP4/M4A `covr` (full-box and QuickTime `meta`, 64-bit sizes), Ogg Vorbis/Opus/FLAC-in-Ogg `METADATA_BLOCK_PICTURE` and legacy `COVERART`. Fallback to the first video frame via OpenCV. Output PNG by default; jpg/webp or the original bytes optional. |
| Code preview (`CodePreviewBitmapRenderer`) | `code_preview` | faithful geometry, approximate highlighting | All renderer constants and formulas ported (see below); ImageToolbox's 4 renderer tests reproduced. Highlighting: Pygments mapped to highlight.js scopes coloured by the same 256 themes (ImageToolbox runs highlight.js). |
| Screenshot framing | (existing `screenshot_framing` in `tools/itb_tools.py`) | already ported elsewhere - not duplicated | Another module already registers `screenshot_framing` with the same formulas; registering a second tool with that name would make `@tool` raise "Duplicate tool" once both modules are loaded. A test checks there is no collision and the ITB geometry (auto canvas keeps source size, 16:9 canvas). Gap found in that tool (not mine to change): it supports two gradient colours only, while ImageToolbox accepts N-colour palettes (`gradientUsesTheMiddlePaletteColors`). There is no window chrome / device frame in ImageToolbox's renderer. |
| Document scanner (ML Kit, Android-only) | `document_scan` | new (classical replacement) | OpenCV page detection (or user corners), perspective rectification, modes color / grayscale / bw / shadow_removal, images or one multi-page PDF (Pillow). |

## Cipher details

Verification method: BouncyCastle jars from the local Gradle cache (bcprov/bcpkix 1.77; ImageToolbox
ships 1.86) were driven by small Java programs (kept outside the repo) that replicate
`AndroidCryptographyManager` line by line (versioned `encrypt`, `decryptVersioned`, `createKey` +
`Cipher.init` fallbacks for legacy).  Results:

* every one of the 34 supported entries: BouncyCastle-encrypted file -> LazyRoom decrypts
  (versioned **and** legacy), and LazyRoom-encrypted file (non-ASCII password and data) ->
  BouncyCastle decrypts with ImageToolbox's code path;
* the 6 OID-named Camellia/Serpent CBC entries fail even inside BouncyCastle (see table) - this is an
  ImageToolbox bug, not a LazyRoom limitation.

The BouncyCastle-produced files are embedded in the test file (`BC_VERSIONED`, `BC_LEGACY`,
`BC_DEFAULT_210000` with the real 210,000 iterations).  Semantics established this way (and documented
in `_cipher.CIPHERS`): plain names (`AES`, `BLOWFISH`, `CAMELLIA`, `DESEDE`, `IDEA`, `SEED`, `SM4`,
`RIJNDAEL`) are BouncyCastle's default ECB/PKCS7 with the IV silently ignored; `ARC4` uses the whole
256-bit key; `DESEDE` keys are 128-bit (two-key 3DES); `IDEA` uses the first 128 key bits; the
`1.2.840.113549.3.7` OID pads with PKCS#7 despite the mapped name `DESede/CBC/NoPadding`; `GCM` is
AES-GCM; PBE schemes are PKCS#12 (SHA-1/SHA-256), PKCS#5 v1 (MD5/SHA-1, DES) and OpenSSL
`EVP_BytesToKey` (MD5, iteration count ignored by BouncyCastle).  Single DES runs as 3DES with
K1=K2=K3.  `AES_256/ECB/PKCS5PADDING` is Conscrypt-only (checked with an equivalent BouncyCastle
transformation).

Supported (34): AES/GCM/NoPadding (default), AES, AES/CTR/NOPADDING, AES/ECB/PKCS5PADDING,
AES/GCM/NOPADDING, AES_256/ECB/PKCS5PADDING, ARC4, BLOWFISH, CAMELLIA, DESEDE,
DESEDE/CBC/PKCS5PADDING, DESede/CBC/NoPadding, GCM, IDEA, PBEWITHMD5AND{128,192,256}BITAES-CBC-OPENSSL,
PBEWITHMD5ANDDES, PBEWITHSHA1ANDDES, PBEWITHSHA256AND{128,192,256}BITAES-CBC-BC,
PBEWITHSHAAND{128,192,256}BITAES-CBC-BC, PBEWITHSHAAND128BITRC2-CBC, PBEWITHSHAAND128BITRC4,
PBEWITHSHAAND40BITRC4, PBEWITHSHAAND2-KEYTRIPLEDES-CBC, PBEWITHSHAAND3-KEYTRIPLEDES-CBC,
PBEWITHSHAANDIDEA-CBC, RIJNDAEL, SEED, SM4.  `cipher_algorithms` also probes each at runtime
(`available`), because Blowfish/IDEA/SEED/RC2/RC4/DES need OpenSSL's legacy provider.

| Unsupported | Reason |
|---|---|
| ARIA, ARIA128/CFB, ARIA128/OFB, ARIA192/CFB, ARIA192/ECB, ARIA192/OFB, ARIA256/CFB, ARIA256/ECB, ARIA256/OFB, ARIAGCM, ASN1-module (= ARIA-128/ECB OID) | ARIA is not implemented by `cryptography` |
| CAMELLIA-128/192/256/CBC, Serpent-128/192/256/CBC (OID names) | Broken in ImageToolbox itself: `ivSize` returns 8 for names containing "CBC", BouncyCastle rejects the 8-byte IV, falls back to a random IV that is never stored, so the file cannot be decrypted (reproduced with BouncyCastle) |
| CAST6, GOST28147, NOEKEON, RC5, RC6, SERPENT, SHACAL-2, SKIPJACK, TEA, THREEFISH-256, TNEPRES, TWOFISH, XTEA, Serpent-*/ECB/CFB/OFB, PBEWITHSHAANDTWOFISH-CBC | Cipher not implemented by `cryptography` |
| PBEWITHMD2ANDDES | MD2 is not available in OpenSSL 3 |
| PBEWITHMD5ANDRC2, PBEWITHSHA1ANDRC2, PBEWITHSHAAND40BITRC2-CBC, RC2/CBC/PKCS5Padding (OID) | `cryptography`'s RC2 accepts 128-bit keys only (these use 64/40/256-bit keys) |

Already excluded by ImageToolbox (`BROKEN` + extra excludes): RSA/IES/KEM families, all *WRAP / KW
ciphers, ChaCha/Salsa/HC/Grain/ZUC/VMPC stream ciphers, CCM variants, DSTU7624, GOST3412-2015, DES,
DES/CBC, THREEFISH-512/1024, most Conscrypt AES CBC/ECB-NoPadding/GCM-SIV names.

## Code preview details

Ported exactly: `RenderScale = 3`, minimum card width 560 x 3, font size x 3, inner padding, 14 px
line-number gap, line-number width = advance of the last line number, wrap / no-wrap card width
(`desired + scale`), `StaticLayout` line height with the 1.55 multiplier (Android rounding, no extra
after the last line, 20 px default tab stops, greedy breaks after whitespace, character breaks for long
words), line numbers only on the first visual line of each logical line at 32 % alpha right-aligned,
42 px header with three 5 px window dots (#FF5F57 #FEBC2E #28C840, 17 px apart), bold 12 px title
(ellipsized with "…" before the language label), 10 px language label at 52 %, 1 px divider at 10 %,
shadow padding `ceil((blur + max(|dx|, |dy|)) x 3)`, Skia shadow sigma `0.57735 r + 0.5`, rotated
bounding box, `ceil(outerPadding x 3)` canvas padding, diagonal `LinearGradient` with evenly spaced
stops, rounded canvas, 65,545,216-pixel cap with the 0.99 factor.  Parameter ranges are the
component's `coerceIn` ranges; 30 background presets + custom N-colour gradients; 192 language keys
with ImageToolbox's display names/extensions; 256 themes.

Themes: `_code_themes.py` was generated from the 256 CSS assets: ImageToolbox's `resolveCssValues`
(CSS variables, `background` shorthand -> last hex colour), then a small CSS cascade (specificity,
order, `!important`, inheritance; compound and descendant class selectors) evaluated for 46
highlight.js scopes (incl. nested `meta keyword`, `tag name`).  Only colours are stored because the
renderer only applies `ForegroundColorSpan`s.

Deviations: highlighting is Pygments token types mapped to highlight.js scopes, not highlight.js
grammars (token boundaries/classes differ in places).  149 of the 192 keys get a Pygments lexer
(19 through aliases such as dos -> batch, x86asm -> nasm, pgsql -> postgresql, processing -> java,
oxygene -> delphi); 43 (1c, accesslog, angelscript, arcade, armasm, asciidoc, avrasm, axapta, cal, cos,
csp, dsconfig, dust, excel, fix, flix, gams, gauss, gml, hsp, isbl, jboss-cli, leaf, livecodeserver,
mel, mercury, mizar, mojolicious, parser3, pf, profile, purebasic, rib, roboconf, routeros,
ruleslanguage, sqf, step21, stylus, subunit, taggerscript, tp, xl) render uncoloured and report
`highlighter: "none"`.  Pygments is optional.  Font: the system monospace font
(fontconfig "monospace" -> Noto Sans Mono here; Menlo/DejaVu/Consolas fallbacks) instead of Android's
Droid Sans Mono, so line heights and widths depend on that font's metrics; bold uses the font's Bold
instance when it is a variable font.  The Android memory-based pixel budget is not modelled (only the
fixed cap).  Rotated cards are resampled bilinearly (Android draws them vectorially).  Added: `input`
parameter to read the code from a text file.

## Document scanner details

Detection on a <= 1000 px proxy: four binarisations (Canny of a text-suppressed image, Otsu, inverted
Otsu, Otsu of saturation) -> contours -> convex 4-gons from `approxPolyDP`, rejected unless they cover
>= 12 % of the image, have no corner far from square, and show a luminance step across all four
sides (or lie on the image border); scored by area x rectangularity.  Corners are refined on the full
image by sub-pixel edge search along each side's normal and robust line fits.  Rectified size = longest
opposite sides.  Modes: color, grayscale, bw (shadow-normalised adaptive threshold united with Otsu),
shadow_removal (per-channel division by a dilated + median background).  If no page is found the whole
image is used and the page carries `detected: false` and a warning.  PDF: one page per scan at
`pdf_dpi` (default 300; ML Kit's own PDF page sizing is not known), quality 85 like `PdfCreationParams`;
images default to JPG like `ImageFormat.Default`, named `<stem>_scan.<ext>`.

## Security considerations

* **Secrets.** Tool parameters are not logged anywhere: the command wrapper in `api.py` writes only
  `_summarise_args(kwargs)`, which reduces the `params` dict to `"<dict[N]>"` (and CLI, MCP and GUI
  call `tool_run` positionally, so even that is empty); `OperationLog`, jobs and project history never
  see tool params.  Therefore no `secret` flag was added to `core/params.py` and `api.py`/`workspace.py`
  are untouched.  The tools never echo passwords in results or errors (`DECODE_FAILED` uses
  ImageToolbox's fixed text; lone-surrogate passwords are rejected without quoting the character), and
  a test reads the real operation-log file after successful, wrong-password, invalid-choice and
  unknown-param runs (positional and keyword call forms) and asserts the passwords are absent.
  Remaining exposure outside this module: CLI arguments (`password=...`) are visible in shell history /
  process lists, and the GUI renders `Text` params unmasked; a `secret`/`writeOnly` schema hint would
  let GUI and MCP clients mask them if the lead wants it.
* **Cipher.** Header fields validated exactly like ImageToolbox before any key derivation; only the
  header-named algorithm is used; GCM tags checked; output never written on failure; output may not be
  the input (no in-place overwrite).  Non-authenticated algorithms are flagged in results.
* **Zip-slip.** `safeSegments` port + resolved-path containment check for every directory and file +
  `O_CREAT|O_EXCL` file creation + refusal to traverse existing symlinks; symlink and special entries
  skipped; one unsafe entry aborts the whole extraction and everything created is rolled back.
* **Zip bombs.** Entry count, per-entry and total limits enforced on bytes actually decompressed
  (declared sizes are checked too), plus a compression-ratio limit (entries > 1 MiB).  Defaults are
  stricter than ImageToolbox (8 GiB total / 4 GiB per entry / ratio 1000 vs 128 GiB / 64 GiB / none);
  the upper bounds of the parameters are ImageToolbox's values.
* All reads/writes go through `PathPolicy` (`check_read`, `check_write`, `check_write_dir`); archives
  are written atomically (temp file + `os.replace`); extraction never overwrites.

## Other deviations

* Archive entries keep the source files' modification times (libarchive entries carry none).  The LZMA
  ZIP method always uses preset 6 (zipfile ignores the level).
* `audio_cover_extract` picks the front cover (type 3) when several pictures exist, else the first
  (Android picks whichever its extractor reports); `index` selects any picture; all pictures are
  listed in the result.
* `decrypt_file` legacy decryption needs the algorithm (as in ImageToolbox, where the selected type is
  used); versioned files ignore it.

## Tests

`LAZYROOM_ITB_ONLY=none LAZYROOM_HOME=/tmp/pm_file_home .venv/bin/python -m pytest tests/test_file_tools.py -q -x`
-> **154 passed** (~14 s wall; the 1,200-line code preview at the 65.5 MP cap is ~5 s).

Covered: BouncyCastle versioned + legacy fixtures for all 34 supported algorithms and the
210,000-iteration default; encrypt -> decrypt round trips for every supported algorithm with header
structure checks; ImageToolbox's `AndroidCryptographyManagerTest` vectors (unique salt/nonce, header
cipher wins over the selected one, wrong password, corrupted header/ciphertext, legacy AES-GCM,
detection); header validation; tool naming/warnings; wrong password = clean `DECODE_FAILED` without
output; password absent from the operation log; missing `cryptography` -> `BACKEND_UNAVAILABLE` with
install hint; archives (4 methods, de-duplication, permissions, list/extract, unique folders, options,
selection, symlinks, 6 zip-slip variants with rollback, ratio/size/count limits, lying sizes, hand-built
ZipCrypto archive with wrong/right password, AES entry, path policy); audio covers from synthetic MP3
(v2.2 UTF-16, v2.3, v2.3 unsync, v2.3 zlib, v2.4 unsync + DLI + extended header), FLAC (plain and after
ID3), M4A, Ogg Vorbis and Opus (valid CRCs), picture selection, no-cover error, video-frame fallback;
code preview geometry formula, determinism, ImageToolbox's padding (+48), shadow (+300), empty-title,
preview/export ratio and long-code tests, themes/languages/titles, highlighting with and without
Pygments, wrapping and tab stops, tool output; document scanner corner accuracy (< 0.5 % of the
diagonal; ~0.4 px measured), rectified content vs. the source page, bw/grayscale/color modes, images
and 2-page PDF output, user corners, no-page warning; no tool-name collision with `itb_tools`.
