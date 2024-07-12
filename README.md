# FastPCI: Motion-Structure Guided Fast Point Cloud Frame Interpolation (ECCV 2024) 

Tianyu Zhang, Guocheng Qian, Jin Xie and Jian Yang


## Requirements

Create a conda environment:
```
conda env create -f environment.yaml
conda activate fastpci

cd models/EMD/
python setup.py install
cp build/lib.linux-x86_64-cpython-39/emd_cuda.cpython-39-x86_64-linux-gnu.so .

cd ../pointnet2/
python setup.py install

cd ../../
```

## Dataset preparation

We utilize the [NL-Drive dataset](https://tongjieducn-my.sharepoint.com/:f:/g/personal/zhengzehan_tongji_edu_cn/Ej4AiwgJWp1MsAFwtWcxIFkBPDwsCW_3bWSRlpYf4XZw-w)
, which processed and integrated KITTI Odometry, Argoverse2sensor, and Nuscenes. 
Please download the NL-Drive dataset here. And put the NL-Drive dataset into `data/NL-Drive` .
We provide the split list of three datasets in `./data/NL-Drive/`.

## Instructions to training and testing

### Training

Training on KITTI Odometry dataset, Argoverse 2 sensor dataset, Nuscenes dataset, run separately:
```
bash train_fastpci_kitti.sh
bash train_fastpci_argoverse2.sh
bash train_fastpci_nuscenes.sh
```

### Testing

Testing on KITTI Odometry dataset, Argoverse 2 sensor dataset, Nuscenes dataset, run separately:
```
bash test_fastpci_kitti.sh
bash test_fastpci_argoverse2.sh
bash test_fastpci_nuscenes.sh
```

## Citation

If you find our code or paper useful, please cite:
```bibtex
@inproceedings{zhang2024fastpci,
  title     = {FastPCI: Motion-Structure Guided Fast Point Cloud Frame Interpolation},
  author    = {Zhang, Tianyu and Qian, Guocheng and Xie, Jin and Yang, Jian},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2024}
  }
```


## Acknowledgments
We thank the authors of

- [PointPWC-Net](https://github.com/DylanWusee/PointPWC)
- [PointINet](https://github.com/ispc-lab/PointINet)
- [NeuralPCI](https://github.com/ispc-lab/NeuralPCI)
- [EMA-VFI](https://github.com/mcg-nju/ema-vfi)

for open sourcing their methods.# FastPCI
