# Deep Spectral Network Pytorch implementation

The repository is part of the submission Domain Invariant Representations with Deep Spectral Alignment for ESANN 2020.
This is a PyTorch implementation of the unsupervised deep domain adaptation method Deep Spectral Networks, which employs the spectral loss in the output layer of the Alexnet architecture.



### Experimental Results 
The method is evaluated on a standard domain adaptation benchmark: [the Office dataset][1].
The Office dataset contains 31 object categories from an office environment in 3 image domains: Amazon, DSLR, and Webcam.

To provide results comparable to the paper, have been considered:
- Source Domain: *Amazon*  
- Target Domain: *Webcam*

![Soft cross-entropy loss](./plots.png)

**(a)** CORAL loss significantly improves performances on target domain
 while maintaining almost the same accuracy on source domain.  
**(b)** Classification loss and CORAL loss while optimizing for both domain invariance and class discrimination.  
**(c)** CORAL distance without adaptation  
**(d)** Comparison between adaptation and no adaptation (lambda = 0) classification loss. Without domain adaptation the classification loss decreases slightly faster.


## Training and Evaluation

- Download and extract the [Office dataset][1]

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE" -O office31.tar.gz && rm -rf /tmp/cookies.txt
tar -xzf office31.tar.gz
```

- Move the three folders composing the dataset in a new directory "data":
```
mkdir data
mv amazon dslr webcam ./data
```

- Train with default parameters:
```
python demo.py
```
After each training epoch, the model will be evaluated on source and target and the respective accuracies printed.
source and target accuracies, classification loss and CORAL loss are saved in `./log.pth`

- Construct plots:
```
python plot_accuracies_losses.py
```

- Construct Evaluation Results:
```
python plot_accuracies_losses.py
```

## Acknowledgment



- Official Caffe implementation of Deep CORAL: https://github.com/VisionLearningGroup/CORAL
- Tracker to save and visualize accuracies and losses during training: https://github.com/Cyanogenoid/pytorch-vqa




[0]: https://arxiv.org/abs/1607.01719
[1]: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code


