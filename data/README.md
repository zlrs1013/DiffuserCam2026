# Data Layout

External sample data should be kept under `data/external/`.

Current Waller-Lab DiffuserCam tutorial sample layout:

```text
data/external/tutorial/
  psf_sample.tif
  rawdata_hand_sample.tif

data/external/test_images/
  spiral_bw.gif
  ...
```

Notebook 04 also supports this alternative wrapped layout:

```text
data/external/waller_lab_diffusercam_tutorial/tutorial/
  psf_sample.tif
  rawdata_hand_sample.tif
```

The `test_images` files are display targets for future hardware capture. They are not ground truth for `rawdata_hand_sample.tif`.

The `data/external/3d/` folder is retained for future work only on this branch. The presentation should describe 3D as a limitation/next step, not as a finished result.
