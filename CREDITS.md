# Credits

## Shipped models and data

- `checkpoints/kodim03.checkpoint`: fitted to Kodak image 03 of the Kodak Lossless True Color Image Suite
  (Eastman Kodak), released for unrestricted use (<http://r0k.us/graphics/kodak/>).
- `checkpoints/hotdog.checkpoint`, `assets/hotdog/`: fitted to view r_14 of the Hotdog scene of the
  NeRF-Synthetic dataset (Mildenhall et al., ECCV 2020); its ground-truth depth, target cameras and source
  image come from that scene. The Hotdog model is by erickfree on Blend Swap, CC0
  (<https://creativecommons.org/publicdomain/zero/1.0/>). `depth_pro.npz` was estimated with Depth Pro
  (<https://github.com/apple/ml-depth-pro>).

## Media (`assets/`)

All media are renders of this code, without metadata and without overlaid text.
- `teaser.jpg`: the source photo of the Venice canal is **AI-generated** (made with an image-generation model,
  ChatGPT), not a photograph of a real scene. The right half is restyled after *The Starry Night*, Vincent van Gogh
  (1889), public domain. The provenance (C2PA) metadata of the generated image is not included in this file.
- `smudge.webp`: Kodak image 03 (see above).
- `cloth.webp`: *The Great Wave off Kanagawa*, Katsushika Hokusai (c. 1831), public domain.
- `mirror.webp`: *The Ambassadors*, Hans Holbein the Younger (1533), public domain.
- `lift.webp`: the NeRF-Synthetic Hotdog scene (see above).

## Datasets used for evaluation (not included)

Kodak-24 (<http://r0k.us/graphics/kodak/>); DIV2K (Agustsson and Timofte, CVPRW 2017,
<https://data.vision.ee.ethz.ch/cvl/DIV2K/>); NeRF-Synthetic (Mildenhall et al., ECCV 2020); DTU MVS
(Jensen et al., CVPR 2014, <https://roboimagedata.compute.dtu.dk/>); the OmniStyle-INR style sets.
