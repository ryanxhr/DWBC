# Discriminator-Weighted Offline Imitation Learning from Suboptimal Demonstrations

This is the code for reproducing the results of the paper Discriminator-Weighted Offline Imitation Learning from Suboptimal Demonstrations accepted at ICML'2022. The paper can be found [here](https://arxiv.org/abs/2207.10050).

### Usage
Paper results were collected with [MuJoCo 1.50](http://www.mujoco.org/) (and [mujoco-py 1.50.1.1](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.17.0](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Networks are trained using [PyTorch 1.4.0](https://github.com/pytorch/pytorch) and Python 3.6.

The paper results can be reproduced by running:
```
./run_dwbc.sh
```

You can also run DWBC on the setting used in [DemoDICE](https://openreview.net/forum?id=BrPdX1bDZkQ) and [SMODICE](https://arxiv.org/abs/2202.02433) by running `main_setting_demodice.py`:

 ```
python main_setting_demodice.py \
    --algorithm="DWBC" \  
    --env_e="hopper-expert-v2" \
    --env_o="hopper-random-v2" \
    --num_e=1 \  # expert trajectory num in D_e
    --num_o_e=200 \  # expert trajectory num in D_o
    --num_o_o=2000 \  # non-expert trajectory num in D_o
```


### Bibtex
```
@inproceedings{xu2022discriminator,
  title     = {Discriminator-Weighted Offline Imitation Learning from Suboptimal Demonstrations},
  author    = {Xu, Haoran and Zhan, Xianyuan and Yin, Honglei and Qin, Huiling},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  pages     = {24725-24742},
  year      = {2022},
}
```

