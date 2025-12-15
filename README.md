# Training-Free 4D Video Reconstruction from a Single 2D Video #
A training-free baseline for reconstructing dynamic 3D (4D) scenes from a single monocular video using off-the-shelf depth and view synthesis models.
## Approach #1: Training-free, modular 4D reconstruction pipeline  ##
| ![Original](videoes/original/babywalk-1.gif) | ![Novel View MI](videoes/dynamic_3d/novel_view_mi.gif) |
|----------------------------------------------|--------------------------------------------------------|
| *Original Video*                                  | *Single-Frame Depth (Midas)*                                            |
| ![Novel View DA](videoes/dynamic_3d/novel_view_da.gif) | ![Novel View MV](videoes/dynamic_3d/novel_view_mv.gif) |
| *Single-Frame Depth (DA-V2)*                                  | *Temporal Multi-Frame (DA-V2)*                                            |
