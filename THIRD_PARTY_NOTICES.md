# Third-party notices and provenance

This document records direct third-party software dependencies and published
work that informed GeoSurge. It does not grant rights beyond the applicable
upstream terms.

## Three.js

Generated visualizations may load **Three.js** at runtime from a CDN (version
`0.160.0`). Three.js source code is not bundled in this repository.

- Copyright © 2010–2026 three.js authors
- License: MIT
- Source and license: <https://github.com/mrdoob/three.js>

If a future release bundles Three.js source code, it must include the complete
upstream MIT license and copyright notice with that distribution.

## TotalSegmentator

Port B can use **TotalSegmentator** as an installed segmentation dependency;
its source code and model weights are not redistributed here.

- License: Apache License 2.0
- Source and license: <https://github.com/wasserth/TotalSegmentator>

## VISTA3D and other optional models

VISTA3D, MedSAM2, BiomedParse, model weights, and datasets are not distributed
with this repository. Users must obtain them separately and comply with their
respective upstream licenses, access conditions, and usage restrictions.

## 3DMedAgent reference

The Port B architecture and implementation work were informed by
[3DMedAgent](https://github.com/jinlab-imvr/3DMedAgent). This acknowledgement
does not imply affiliation with or endorsement by that project's authors.

At the time of this notice, the upstream repository does not display a
repository license. No source file from that repository may be copied,
modified, or redistributed in GeoSurge unless the applicable upstream license
or written permission is documented first.

## Bézier-surface liver-resection planning reference

The Bézier-surface and distance-map planning approach was informed by:

> Palomar R, Cheikh FA, Edwin B, Fretland Å, Beghdadi A, Elle OJ.
> *A novel method for planning liver resections using deformable Bézier
> surfaces and distance maps.* Computer Methods and Programs in Biomedicine.
> 2017;144:135–145. doi:[10.1016/j.cmpb.2017.03.019](https://doi.org/10.1016/j.cmpb.2017.03.019).

The publication is available under CC BY-NC-ND. GeoSurge does not reproduce
the paper's text, figures, supplementary material, or implementation. This
reference records methodological inspiration; it is not a license for any
third-party code.

## GeoSurge-authored code

Unless a file states otherwise, code authored for GeoSurge is licensed under
Apache License 2.0. The repository-wide license does not override any
third-party license, copyright notice, model term, data-use term, or patent
right.
