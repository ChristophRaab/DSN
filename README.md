# Deep Spectral Network Pytorch implementation

Code to the Paper Domain Invariant Representations with Deep Spectral Alignment presented at ESANN 2020.
The proposed unsupervised deep domain adaptation method Deep Spectral Network is implemented in pytorch. The Deep Spectral Networks employs the spectral loss in the output layer of the Alexnet architecture.

## Training and Evaluation

- Download and extract the [Office dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code)

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE" -O office31.tar.gz && rm -rf /tmp/cookies.txt
tar -xzf office31.tar.gz
```

- Move the three folders composing the dataset in a new directory "data":
```
mkdir data
mv amazon dslr webcam ./dataset
```

- Train with default parameters:
```
python demo.py
```
After each training epoch, the model will be evaluated on source and target and the respective accuracies printed.
source and target accuracies, classification loss and Spectral loss are saved in `./log.pth`

- Construct Evaluation Results:
```
python study.py
```
For reproducing the results in the exerpimental part of the paper. After study.py is finished run merge the results must be merged:
```
python merge_results.py
```

- Construct plots of convergence behaviour:
```
python plot_accuracies_losses.py
```

## Acknowledgment
* Special Thanks to [DenishDsh](https://github.com/DenisDsh), who has made a reliable implementation of [CORAL](https://github.com/VisionLearningGroup/CORAL) public.
