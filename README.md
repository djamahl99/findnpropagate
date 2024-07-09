<!-- <img src="docs/open_mmlab.png" align="right" width="30%"> -->

# Find n' Propagate: Open-Vocabulary 3D Object Detection in Urban Environments

Official repository for Find n' Propagate: Open-Vocabulary 3D Object Detection in Urban Environments.


**Highlights**: 
* `Find n' Propagate` has been accepted at ECCV 2024!

## Overview
- [Model Zoo](#model-zoo)
- [Installation](docs/INSTALL.md)
- [Quick Demo](docs/DEMO.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Citation](#citation)


## Introduction



## Model Zoo

Use transfusion_lidar.yaml config for testing, need the normal class names so that OpenPCDet loads all the GT classes in the correct order.

### NuScenes 3D Object Detection - 6 Class Known / 4 Unknown Classes


| Method | VLM  | mAP   | NDS   | AP_B  | AP_N  | AR_N  | Download |
|--------|------|-------|-------|-------|-------|-------|----------|
| Ours   | OWL  | 42.52 | 45.13 | 53.09 | 26.66 | 60.10 |[model 94mb](https://drive.google.com/file/d/19whuGiz2rglDMyHm1e9VPV2OxiaeXFDn/view?usp=drive_link)|


### NuScenes 3D Object Detection - 3 Class Known / 7 Unknown Classes


| Method | Arch        | VLM  | mAP   | NDS   | AP_B  | AP_N  | AR_N  | Download |
|--------|-------------|------|-------|-------|-------|-------|-------|----------|
| Ours   | Transfusion | GLIP | 31.44 | 34.53 | 67.41 | 16.03  | 49.78  | [model 94mb](https://drive.google.com/file/d/1dEBL9fv4tePL_BKkouLGzyYC4Fzu4Jq3/view?usp=sharing) | 


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.


## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to get started with `OpenPCDet`.

### GLIP Predictions
Download [nuscenes_infos_train_mono3d.coco.json](https://drive.google.com/file/d/1At86saiwUzPlE7rgmCYtsD76AOuxvfJ9/view?usp=sharing) and [nuscenes_glip_train_pred.pth](https://drive.google.com/file/d/1PxOpy1esgV3fezf5XZVxLEYqSjLnJRzo/view?usp=sharing) to OpenPCDet/data/training_pred. This will be loaded by the PreprocessedGLIP class in pcdet/models/preprocessed_detector.py to generate pseudo-labels.

### Training Process

1.  Extract `Greedy Box Seeker` Boxes

```
python extract_pseudo_labels.py --cfg_file cfgs/nuscenes_box_seeker_proposals.yaml --folder ../data/pseudo_labels/nuscenes_box_seeker_proposals/
```

2. Run Self-training

```
python train_st.py  --cfg_file tools/cfgs/transfusion_lidar_st_nodisaugs_0.1sample_nomaxst_60confqueue_rotw_novelw_3sepinf_10fixcp_0.5unkw_1.0trans_0.785rot_0.2drop_glip_cpstonly_0.9997mom.yaml
```

3. Evaluate on all classes

```
python test.py --cfg_file cfgs/nuscenes_models/transfusion_lidar.yaml --ckpt ../output/transfusion_lidar_st_nodisaugs_0.1sample_nomaxst_60confqueue_rotw_novelw_3sepinf_10fixcp_0.5unkw_1.0trans_0.785rot_0.2drop_glip_cpstonly_0.9997mom/default/ckpt/checkpoint_epoch_20.pth
```



## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`OpenPCDet` is an open source project for LiDAR-based 3D scene perception that supports multiple
LiDAR-based perception models as shown above. Some parts of `PCDet` are learned from the official released codes of the above supported methods. 
We would like to thank for their proposed methods and the official implementation.   

We hope that this repo could serve as a strong and flexible codebase to benefit the research community by speeding up the process of reimplementing previous works and/or developing new methods.


## Citation 
If you find this project useful in your research, please consider citing:


```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

```
@inproceedings{findnpropagate2024,
    title={Find n' Propagate: Open Vocabulary 3D Object Detection in Urban Scenes},
    author={Djamahl Etchegaray and Zi Huang and Tatsuya Harada and Yadan Luo},
    title = {European Conference on Computer Vision (ECCV)},
    year = {2024},
    publisher = {Springer},
    year={2024},
    volume={}, # TODO: fill
    pages={},
}
```


