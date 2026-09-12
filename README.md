# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py

## Train logs

### Simple 

#### Config:

```commandline
PTS = 50
HIDDEN = 4
RATE = 0.5
```

```commandline
Epoch time: 0.24s
```

```commandline
Epoch  10  loss  32.44172484269226 correct 38
Epoch  20  loss  23.64445795575785 correct 50
Epoch  30  loss  11.809243877067876 correct 47
Epoch  40  loss  7.668494809659995 correct 48
Epoch  50  loss  4.113202768871685 correct 50
Epoch  60  loss  3.1108161768122016 correct 50
Epoch  70  loss  2.5360079504312103 correct 50
Epoch  80  loss  2.152687382932105 correct 50
Epoch  90  loss  1.8747224334913692 correct 50
Epoch  100  loss  1.6629159576621202 correct 50
Epoch  110  loss  1.4950958006079087 correct 50
Epoch  120  loss  1.358370703121657 correct 50
Epoch  130  loss  1.2445087325633537 correct 50
Epoch  140  loss  1.1479602710283296 correct 50
Epoch  150  loss  1.0648810989173143 correct 50
Epoch  160  loss  0.9924976458869289 correct 50
Epoch  170  loss  0.928781631710769 correct 50
Epoch  180  loss  0.872211219113664 correct 50
Epoch  190  loss  0.8216019140906096 correct 50
Epoch  200  loss  0.776073052133653 correct 50
Epoch  210  loss  0.7348118731165543 correct 50
Epoch  220  loss  0.6972326051851878 correct 50
Epoch  230  loss  0.6628568479260029 correct 50
Epoch  240  loss  0.6312886571328156 correct 50
Epoch  250  loss  0.6021970393397674 correct 50
Epoch  260  loss  0.5753027749267708 correct 50
Epoch  270  loss  0.5503683773382271 correct 50
Epoch  280  loss  0.5272028339310293 correct 50
Epoch  290  loss  0.5056245179137047 correct 50
Epoch  300  loss  0.4854794052965543 correct 50
Epoch  310  loss  0.466640851248063 correct 50
Epoch  320  loss  0.44897158763544914 correct 50
Epoch  330  loss  0.4323706719826295 correct 50
Epoch  340  loss  0.4167494036877975 correct 50
Epoch  350  loss  0.40202634336278137 correct 50
Epoch  360  loss  0.388130066181971 correct 50
Epoch  370  loss  0.3749965038799189 correct 50
Epoch  380  loss  0.36256800389825755 correct 50
Epoch  390  loss  0.3507925335659391 correct 50
Epoch  400  loss  0.3396230029223456 correct 50
Epoch  410  loss  0.3290166854532933 correct 50
Epoch  420  loss  0.31893471992510514 correct 50
Epoch  430  loss  0.30934168016769087 correct 50
Epoch  440  loss  0.30020520202921763 correct 50
Epoch  450  loss  0.2914956585643028 correct 50
Epoch  460  loss  0.28319775602222363 correct 50
Epoch  470  loss  0.27527703187569547 correct 50
Epoch  480  loss  0.2677058978382299 correct 50
Epoch  490  loss  0.26046380268994296 correct 50
Epoch  500  loss  0.2535315915500568 correct 50
```

### Diag 

#### Config:

```commandline
PTS = 50
HIDDEN = 2
RATE = 0.5
```

```commandline
Epoch time: 0.09s
```

```commandline
Epoch  10  loss  19.23348433236608 correct 44
Epoch  20  loss  17.886089235355314 correct 44
Epoch  30  loss  17.04952124233412 correct 44
Epoch  40  loss  15.956673701363886 correct 44
Epoch  50  loss  14.629678338877266 correct 44
Epoch  60  loss  12.96341918109833 correct 44
Epoch  70  loss  10.79876450383195 correct 44
Epoch  80  loss  8.537403407871453 correct 44
Epoch  90  loss  6.952786044212072 correct 48
Epoch  100  loss  6.081968966454233 correct 48
Epoch  110  loss  5.449637408307888 correct 48
Epoch  120  loss  4.891714567456854 correct 48
Epoch  130  loss  4.399257489794555 correct 48
Epoch  140  loss  3.964561439374164 correct 48
Epoch  150  loss  3.5808015976865133 correct 49
Epoch  160  loss  3.241930692185885 correct 49
Epoch  170  loss  2.942594380494506 correct 50
Epoch  180  loss  2.6780547384955913 correct 50
Epoch  190  loss  2.4441188489860624 correct 50
Epoch  200  loss  2.237074131666263 correct 50
Epoch  210  loss  2.053632702944919 correct 50
Epoch  220  loss  1.8908851303993797 correct 50
Epoch  230  loss  1.7462619546860545 correct 50
Epoch  240  loss  1.6175005640496587 correct 50
Epoch  250  loss  1.5026153222933174 correct 50
Epoch  260  loss  1.403119247029806 correct 50
Epoch  270  loss  1.3186489358939597 correct 50
Epoch  280  loss  1.2442707181205075 correct 50
Epoch  290  loss  1.1762112020248765 correct 50
Epoch  300  loss  1.1101510381864679 correct 50
Epoch  310  loss  1.05555680261604 correct 50
Epoch  320  loss  1.005330368567024 correct 50
Epoch  330  loss  0.9589552889803534 correct 50
Epoch  340  loss  0.9160082783450296 correct 50
Epoch  350  loss  0.8761346780132582 correct 50
Epoch  360  loss  0.8390322400856899 correct 50
Epoch  370  loss  0.8044399450788088 correct 50
Epoch  380  loss  0.7721300045826374 correct 50
Epoch  390  loss  0.7419019572703598 correct 50
Epoch  400  loss  0.7135781878811777 correct 50
Epoch  410  loss  0.687000442977765 correct 50
Epoch  420  loss  0.6620270640556368 correct 50
Epoch  430  loss  0.6385307496999184 correct 50
Epoch  440  loss  0.6163967166954968 correct 50
Epoch  450  loss  0.5955211681472832 correct 50
Epoch  460  loss  0.575810002252277 correct 50
Epoch  470  loss  0.55717771288451 correct 50
Epoch  480  loss  0.5395464453826632 correct 50
Epoch  490  loss  0.5228451796194586 correct 50
Epoch  500  loss  0.5070090187109791 correct 50
```

### Split 

#### Config:

```commandline
PTS = 50
HIDDEN = 8
RATE = 0.5
```

```commandline
Epoch time: 0.68s
```

```commandline
Epoch  10  loss  31.085489341188733 correct 31
Epoch  20  loss  29.15682704474763 correct 33
Epoch  30  loss  26.58831809281113 correct 37
Epoch  40  loss  34.657400113121646 correct 31
Epoch  50  loss  23.161364810608816 correct 33
Epoch  60  loss  19.847579021045725 correct 39
Epoch  70  loss  17.168952356808546 correct 41
Epoch  80  loss  12.54969467514591 correct 46
Epoch  90  loss  11.448684152192362 correct 46
Epoch  100  loss  8.5230307742014 correct 48
Epoch  110  loss  9.362143976610295 correct 45
Epoch  120  loss  7.723652146706051 correct 47
Epoch  130  loss  9.151830695377923 correct 45
Epoch  140  loss  5.481408333954987 correct 49
Epoch  150  loss  30.483814809918634 correct 41
Epoch  160  loss  4.356614946022407 correct 49
Epoch  170  loss  3.773114016025863 correct 50
Epoch  180  loss  3.5247369739348793 correct 50
Epoch  190  loss  4.215283902108867 correct 48
Epoch  200  loss  11.288326789306455 correct 43
Epoch  210  loss  3.5272579602610503 correct 49
Epoch  220  loss  3.07884334156068 correct 50
Epoch  230  loss  3.06973603298695 correct 50
Epoch  240  loss  3.049393106569659 correct 50
Epoch  250  loss  3.4362865927691275 correct 49
Epoch  260  loss  3.748516392923616 correct 49
Epoch  270  loss  4.054467028167399 correct 49
Epoch  280  loss  4.363702441109744 correct 49
Epoch  290  loss  3.8584108400462314 correct 49
Epoch  300  loss  3.0995161064680583 correct 49
Epoch  310  loss  2.90704530719047 correct 49
Epoch  320  loss  2.8405811322791745 correct 49
Epoch  330  loss  2.8080799128692084 correct 49
Epoch  340  loss  2.8161971583058727 correct 49
Epoch  350  loss  2.799548892178706 correct 49
Epoch  360  loss  2.7822895543067685 correct 49
Epoch  370  loss  2.7677469470441642 correct 49
Epoch  380  loss  2.754720580251902 correct 49
Epoch  390  loss  2.738769442825728 correct 49
Epoch  400  loss  2.7167806631695566 correct 49
Epoch  410  loss  2.6956961661281853 correct 49
Epoch  420  loss  2.674277311059044 correct 49
Epoch  430  loss  2.648607681650827 correct 49
Epoch  440  loss  2.6191716703678347 correct 49
Epoch  450  loss  2.5781895582294854 correct 49
Epoch  460  loss  2.5387350592591487 correct 49
Epoch  470  loss  2.4998512590436386 correct 49
Epoch  480  loss  2.4642168007027423 correct 49
Epoch  490  loss  2.4342687101436735 correct 49
Epoch  500  loss  2.401780647057112 correct 49
```

### Xor 

#### Config:

```commandline
PTS = 50
HIDDEN = 8
RATE = 0.5
```

```commandline
Epoch time: 0.64s
```

```commandline
Epoch  10  loss  35.17280923137794 correct 23
Epoch  20  loss  34.55979423947935 correct 32
Epoch  30  loss  34.42973818020691 correct 33
Epoch  40  loss  34.30194983046773 correct 33
Epoch  50  loss  34.11736532352274 correct 35
Epoch  60  loss  33.81200370584857 correct 33
Epoch  70  loss  33.5164515710135 correct 34
Epoch  80  loss  33.17389846925103 correct 33
Epoch  90  loss  32.77919521965919 correct 34
Epoch  100  loss  32.307351200038596 correct 34
Epoch  110  loss  31.77920001899437 correct 33
Epoch  120  loss  31.193409017110326 correct 32
Epoch  130  loss  30.49737888053913 correct 32
Epoch  140  loss  29.580457470852767 correct 33
Epoch  150  loss  26.058258999389157 correct 40
Epoch  160  loss  24.96107194440131 correct 41
Epoch  170  loss  25.255957953531542 correct 39
Epoch  180  loss  24.892463005830834 correct 36
Epoch  190  loss  24.451727630445255 correct 36
Epoch  200  loss  22.98652397532168 correct 38
Epoch  210  loss  22.13310132492082 correct 38
Epoch  220  loss  20.08563399694197 correct 40
Epoch  230  loss  19.23731264143342 correct 40
Epoch  240  loss  17.20900236411271 correct 42
Epoch  250  loss  16.27774295535769 correct 43
Epoch  260  loss  30.626909527369836 correct 32
Epoch  270  loss  13.698297075277852 correct 45
Epoch  280  loss  20.810371143082634 correct 42
Epoch  290  loss  15.268455771702994 correct 42
Epoch  300  loss  14.712298516344118 correct 43
Epoch  310  loss  13.708205428385114 correct 44
Epoch  320  loss  16.516557963999396 correct 40
Epoch  330  loss  11.043273305839392 correct 48
Epoch  340  loss  18.396970870179732 correct 38
Epoch  350  loss  10.25369205580093 correct 48
Epoch  360  loss  15.532837394941764 correct 40
Epoch  370  loss  10.568763856052216 correct 47
Epoch  380  loss  9.221429803920135 correct 49
Epoch  390  loss  8.562979346955768 correct 48
Epoch  400  loss  22.412725181351373 correct 40
Epoch  410  loss  7.737690045582363 correct 49
Epoch  420  loss  10.585587700938166 correct 45
Epoch  430  loss  9.43751380128255 correct 48
Epoch  440  loss  7.8175239550590305 correct 49
Epoch  450  loss  7.2046005470772005 correct 48
Epoch  460  loss  6.6648392241538525 correct 49
Epoch  470  loss  13.182173568584027 correct 45
Epoch  480  loss  6.428106928368289 correct 49
Epoch  490  loss  5.738626740953877 correct 49
Epoch  500  loss  5.0590744547201 correct 49

```
