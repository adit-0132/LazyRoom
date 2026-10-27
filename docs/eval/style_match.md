# Reference style matching — evaluation

Ground truth: the reference is photo B graded with a built-in look; the ideal result is photo A graded with the same look (grain, borders and light leaks excluded). Metric: mean CIE76 ΔE to the ground truth (lower is better); SSIM of luminance vs the source measures content preservation.

| Method | mean ΔE (lower = better) | median ΔE |
|---|---|---|
| lazyroom | 13.17 | 12.82 |
| unedited | 14.03 | 12.70 |
| mkl | 33.88 | 28.77 |
| reinhard | 33.99 | 28.92 |
| histogram | 35.22 | 30.99 |

## Per case

| Look | A | B | LazyRoom ΔE | Reinhard | MKL | Histogram | Unedited | PM SSIM | PM similarity |
|---|---|---|---|---|---|---|---|---|---|
| warm-70s-film | astronaut.png | chelsea.png | 9.91 | 20.42 | 20.57 | 19.79 | 10.70 | 0.879 | 0.19 (low) |
| warm-70s-film | coffee.png | rocket.jpg | 12.30 | 52.19 | 52.24 | 52.44 | 10.24 | 0.928 | 0.35 (low) |
| warm-70s-film | chelsea.png | astronaut.png | 5.75 | 18.08 | 18.02 | 20.87 | 8.06 | 1.000 | 0.882 (high) |
| warm-70s-film | rocket.jpg | coffee.png | 19.39 | 49.15 | 49.99 | 49.06 | 15.87 | 0.986 | 0.0 (low) |
| teal-orange | astronaut.png | chelsea.png | 14.73 | 25.70 | 25.80 | 25.26 | 12.20 | 0.946 | 0.214 (low) |
| teal-orange | coffee.png | rocket.jpg | 14.51 | 70.13 | 69.94 | 71.27 | 15.84 | 0.970 | 0.109 (low) |
| teal-orange | chelsea.png | astronaut.png | 10.82 | 23.74 | 21.48 | 27.84 | 13.20 | 1.000 | 0.815 (high) |
| teal-orange | rocket.jpg | coffee.png | 7.14 | 60.77 | 62.33 | 60.13 | 11.20 | 0.999 | 0.201 (low) |
| faded-analog | astronaut.png | chelsea.png | 9.49 | 20.52 | 20.65 | 20.65 | 7.42 | 0.894 | 0.3 (low) |
| faded-analog | coffee.png | rocket.jpg | 16.64 | 52.06 | 52.01 | 51.76 | 9.14 | 0.943 | 0.391 (low) |
| faded-analog | chelsea.png | astronaut.png | 5.08 | 17.53 | 16.66 | 21.51 | 4.18 | 0.992 | 0.605 (medium) |
| faded-analog | rocket.jpg | coffee.png | 10.84 | 48.72 | 49.21 | 50.01 | 4.47 | 0.978 | 0.13 (low) |
| cool-urban | astronaut.png | chelsea.png | 18.93 | 22.89 | 23.06 | 21.83 | 19.91 | 0.956 | 0.74 (medium) |
| cool-urban | coffee.png | rocket.jpg | 28.62 | 39.93 | 39.96 | 43.12 | 28.65 | 0.955 | 0.436 (low) |
| cool-urban | chelsea.png | astronaut.png | 15.11 | 17.45 | 17.47 | 20.74 | 15.05 | 1.000 | 0.845 (high) |
| cool-urban | rocket.jpg | coffee.png | 14.16 | 41.10 | 41.46 | 41.91 | 11.96 | 0.999 | 0.597 (medium) |
| earthy-muted | astronaut.png | chelsea.png | 6.85 | 19.26 | 19.43 | 18.50 | 8.22 | 0.880 | 0.294 (low) |
| earthy-muted | coffee.png | rocket.jpg | 8.01 | 40.97 | 41.04 | 41.32 | 11.07 | 0.918 | 0.39 (low) |
| earthy-muted | chelsea.png | astronaut.png | 5.23 | 16.36 | 16.19 | 19.66 | 4.58 | 0.993 | 0.671 (medium) |
| earthy-muted | rocket.jpg | coffee.png | 17.90 | 39.30 | 39.88 | 40.25 | 15.30 | 0.977 | 0.147 (low) |
| bw-classic | astronaut.png | chelsea.png | 14.40 | 16.78 | 16.78 | 15.82 | 22.24 | 0.968 | 0.708 (medium) |
| bw-classic | coffee.png | rocket.jpg | 32.52 | 32.46 | 32.46 | 34.14 | 43.70 | 0.978 | 0.126 (low) |
| bw-classic | chelsea.png | astronaut.png | 13.78 | 13.68 | 13.70 | 17.75 | 23.48 | 1.000 | 0.695 (medium) |
| bw-classic | rocket.jpg | coffee.png | 14.61 | 31.74 | 31.74 | 34.18 | 24.57 | 0.990 | 0.676 (medium) |
| moody-dark | astronaut.png | chelsea.png | 8.29 | 19.94 | 20.05 | 19.23 | 9.95 | 0.871 | 0.469 (low) |
| moody-dark | coffee.png | rocket.jpg | 11.93 | 47.54 | 47.53 | 47.85 | 13.28 | 0.949 | 0.419 (low) |
| moody-dark | chelsea.png | astronaut.png | 8.05 | 16.67 | 16.24 | 20.27 | 9.58 | 0.986 | 0.479 (low) |
| moody-dark | rocket.jpg | coffee.png | 9.69 | 45.18 | 45.60 | 44.84 | 5.44 | 0.998 | 0.379 (low) |
| cross-process | astronaut.png | chelsea.png | 17.81 | 26.09 | 24.44 | 26.34 | 15.64 | 0.975 | 0.48 (low) |
| cross-process | coffee.png | rocket.jpg | 14.99 | 62.74 | 62.39 | 64.76 | 15.72 | 0.974 | 0.08 (low) |
| cross-process | chelsea.png | astronaut.png | 13.34 | 21.82 | 17.27 | 26.33 | 13.33 | 0.999 | 0.739 (medium) |
| cross-process | rocket.jpg | coffee.png | 10.66 | 56.86 | 58.41 | 57.55 | 14.91 | 0.998 | 0.255 (low) |
