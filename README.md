# DRL-FlappyBird
Play Flappy Bird by DQN with PARL.  

<img src="https://s1.ax1x.com/2020/06/30/N4DLgU.gif" alt="flappybird" /> 

## Environment and Results
使用ple的FlappyBird环境默认配置(288×512的屏幕大小)，将原始画面预处理作为状态，在跳帧数和上下文长度设为4、探索概率0.1的情况下，训练20000步后，小鸟可以熟练通过大多数障碍，最终都在一处极限上升障碍(计算证明难以通过)上失利。  
<img src="https://s1.ax1x.com/2020/06/30/N4Dju4.gif" alt="fail" /> 

For more detail see <https://blog.csdn.net/u011189503/article/details/106969826>
## How to use
### Dependencies:
+ [paddlepaddle>=1.6.1](https://github.com/PaddlePaddle/Paddle)  
+ [parl==1.3.1](https://github.com/PaddlePaddle/PARL)  
+ gym  
+ [ple](https://github.com/ntasfi/PyGame-Learning-Environment)  
+ [gym-ple](https://github.com/lusob/gym-ple)  

```
pip install -r requirements.txt
```

### Start Training:
```
python train.py
```
### Test
```
python test.py
```