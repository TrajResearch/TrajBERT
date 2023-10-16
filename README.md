# TrajBERT

**The parameter settings of different ablation experiments refer to the run.py file**



# Sample training command 

### TDrive dataset traing command

```bash
python run.py --data_type tdrive  --remask 0 --d_model 512 --if_posiemb 1 --loss spatial_loss --use_his 1 --is_training 1 --gpu 0  --epoch 50  --lr 1e-4
```



### Geolife dataset traing command

```bash
python run.py --data_type geolife  --remask 2 --d_model 512 --if_posiemb 1 --loss spatial_loss --use_his 1 --is_training 1 --gpu 0  --epoch 50  --lr 1e-4
```



### CDR dataset traing command

```bash
python run.py --data_type cdr  --remask 0 --d_model 512 --if_posiemb 1 --loss spatial_loss --use_his 1 --is_training 1 --gpu 0  --epoch 50  --lr 1e-4
```
### Citation
If you find this repo useful and would like to cite it, citing our paper as the following will be really appropriate:

```bash
@ARTICLE{10189092,
  author={Si, Junjun and Yang, Jin and Xiang, Yang and Wang, Hanqiu and Li, Li and Zhang, Rongqing and Tu, Bo and Chen, Xiangqun},
  journal={IEEE Transactions on Mobile Computing}, 
  title={TrajBERT: BERT-Based Trajectory Recovery with Spatial-Temporal Refinement for Implicit Sparse Trajectories}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMC.2023.3297115}}
```
