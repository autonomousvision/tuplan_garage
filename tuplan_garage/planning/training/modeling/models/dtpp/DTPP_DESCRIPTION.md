<p align="center">
    <h1 align="center">DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation
for Tree Policy Planning in Autonomous Driving</h1>
</p>

## Overview
- This is implementation of the DTPP planner is based on the [official implementation](https://github.com/MCZhi/DTPP)
- For Details regarding the model, please checkout the project page in the official repository
- For Details regarding the method, please refer to the offical repositories or the papers linked below.
- For a fair comparison, the results reported for this model were generated after training on the standardized `train150k_split` rather than with the official checkpoint.

## Installation Notes
In order to run this model, you will have to upgrade the pytorch version.
To do so, run `pip install pytorch==2.0.1`.

**Note:** the remaining models were not developed nor tested with `torch==1.9.0`. While we found most of them to work with `torch==2.0.1`, we don't provide any guarantee nor support.

## Citation
```BibTeX
@inproceedings{huang2024dtpp,
  title={DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving},
  author={Huang, Zhiyu and Karkus, Peter and Ivanovic, Boris and Chen, Yuxiao and Pavone, Marco and Lv, Chen},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6806--6812},
  year={2024}
}
```

## Acknowledgements
the DTPP Planner Code is based on code from the following projects: 
- [DTPP](https://github.com/MCZhi/DTPP). Thanks to the community for these great open-source contributions!
