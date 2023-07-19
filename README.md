# TrajBERT

**The parameter settings of different ablation experiments refer to the run.py file**



# Sample training command 

###TDrive dataset traing command

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

