# Addressing Class Imbalance in Federated Learning 
This is the code for our AAAI-2021 paper: [Addressing Class Imbalance in Federated Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17219).

## Run the code
To run the monitoring scheme, you can
```
cd ./FEMNIST-monitor/
python3 main_nn.py
```

To load different loss functions on federated learning, you can
```
cd ./FEMNIST-4-Losses/
python3 main_nn.py --loss ce/focal/ratio/ghm
```

## Citation
If you find our work is helpful for your research, please cite our paper.
```
@inproceedings{wang2021addressing,
  title={Addressing Class Imbalance in Federated Learning},
  author={Wang, Lixu and Xu, Shichao and Wang, Xiao and Zhu, Qi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={11},
  pages={10165--10173},
  year={2021}
}
```
