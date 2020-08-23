# An implementation of DenseResNet

## What is DenseResNet ?

<span style="float: right">[<img alt="Architecture of DenseResNets" src="res/architecture.png" width="250px">](res/architecture.png)</span>

DenseResNet is a Density connected Residual convolutional neural Network for image recognition tasks.
An architecture of DenseResNets is a stack of Residual Blocks just like ResNets, though the circuit design is similar to DenseNets.
In order to improve the performance, DenseResNets use Gate Modules instead of elemet-wise additions or concatenations.
Gate Modules contain attention mechanisms which select useful features dynamically.
Experiental results show that DenseResNets achieve higher performance than conventional ResNets in image classification tasks.

［NOTE］This implementation concains some unefficient codes because the purpuse is performance evalutation of DenseResNets.

