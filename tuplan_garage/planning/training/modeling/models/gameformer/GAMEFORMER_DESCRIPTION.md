<p align="center">
    <h1 align="center">GameFormer Planner</h1>
</p>

## Overview
- This is implementation of the GameFormer planner is based on the [official implementation](https://github.com/MCZhi/GameFormer-Planner)
- For Details regarding the model, please checkout the project page in the official repository
- Note: the GameFormer Planner model is different from the [GameFormer model](https://github.com/MCZhi/GameFormer). It is the nuPlan adaption which was submitted to the [nuPlan Planning Challenge](https://opendrivelab.com/challenge2023/) at the CVPR'23 End-to-End Autonomous Driving Workshop
- For Details regarding the method, please refer to the offical repositories or the papers linked below.
- For a fair comparison, the results reported for this model were generated after training on the standardized `train150k_split` rather than with the official checkpoint.

## Installation Notes
To run this model, you need to execute some additional steps after installing the `tuplan_garage` package:

* Install the libsuitesparse package 
    
    `sudo apt-get install libsuitesparse-dev` (ubuntu)
    `conda install -c conda-forge suitesparse` (mac)

* Make sure `nvcc` is installed (check `nvcc --version`). If it is not yet installed, consider installing `cudatoolkit` from `conda-forge`.

* GameFormer requires the `theseus-ai` package and a more recent `pytorch` version than the nuplan-devkit. Install them with:
    
    `pip install theseus-ai==0.1.3 torch==2.0.1 torchvision==0.15.2`

    **Note:** the remaining models were not developed nor tested with `torch==1.9.0`. While we found most of them to work with `torch==2.0.1`, we don't provide any guarantee nor support.

## Citation
```BibTeX
@InProceedings{Huang_2023_ICCV,
    author    = {Huang, Zhiyu and Liu, Haochen and Lv, Chen},
    title     = {GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3903-3913}
}
```

```BibTeX
@article{huang2023gameformer,
  title={Gameformer planner: A learning-enabled interactive prediction and planning framework for autonomous vehicles},
  author={Huang, Zhiyu and Liu, Haochen and Mo, Xiaoyu and Lyu, Chen},
  journal={URL https://opendrivelab. com/e2ead/AD23Challenge/Track\_4\_AID. pdf},
  year={2023}
}
```

## Acknowledgements
the GameFormer Planner Code is based on code from the following projects: 
- [GameFormer Planner](https://github.com/MCZhi/GameFormer-Planner). Thanks to the community for these great open-source contributions!
