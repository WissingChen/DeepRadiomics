# DeepRadiomics
DeepRadiomics implements a deep learning generalization of the radiomics classification model using pytorch.

If you use this code, please cite: 
```
@article{Shi2020PredictionOK,
  title={Prediction of KRAS, NRAS and BRAF status in colorectal cancer patients with liver metastasis using a deep artificial neural network based on radiomics and semantic features.},
  author={Ruichuan Shi and W. Chen and Bo-Wen Yang and J. Qu and Yu Cheng and Z. Zhu and Y. Gao and Q. Wang and Y. Liu and Zhi Li and Xiujuan Qu},
  journal={American journal of cancer research},
  year={2020},
  volume={10 12},
  pages={
          4513-4526
        }
}
```

## Running Experiments

To run an experiment, you should run preprocess.py first, then you will get a h5py file with all data.

Make a .ini file including some parameter such network structure and train it from main.py
