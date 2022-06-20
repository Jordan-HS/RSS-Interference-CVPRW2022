# Does Interference Exist When Training a Once-For-All Network? (CVPRW, 2022)
# [[arXiv]]()
- Implementation based on "Once for All: Train One Network and Specialize it for Efficient Deployment" [[arXiv]](https://arxiv.org/abs/1908.09791) [[github]](https://github.com/mit-han-lab/once-for-all)  
- Highly reccomended to also look at OFA original paper and official repoisitory.  
- This repository is a work in progress.

## How to use
Adjust main run parameters at the top of the main.py file and run file. (There are additional parameters below in main.py)
```python
args.path = 'Trained_Networks/RSS-Net'  # Save path of model
args.task = 'RSS'                       # Task to run
args.phase = 2                          # Training phase for PS
base_epochs = 180                       # base training epochs
base_learning_rate = 0.01               # base learning rate for training
args.manual_seed = 0                    # manual seed for consistency
args.kd_ratio = 0                       # knowledge distillation use
args.kd_type = 'ce'                     # knowledge distillation type
```

## Task options
| Task | Use |
|----------------------|----------|
| `'super'` | Train the supernet only. |
| `'RSS` | Train the population using Random Subnet Sampling. |
| `'RSS Anchor'` | Train only a single subnet in the population, this single network is refered to as the 'anchor'. |
| `'eval subnets'` | Evaluate 100 random subnets in the population with respect to flops. |
| `'net flops'` | Get the flops for a single network. |
| `'flops bucket'` | Evaluate `n` number of subnets at the provided flops in the `buckets` list. |
| `'kernel'` | Train progressive shrinking's dyanmic kernel stage. |
| `'width'` | Train progressive shrinking's dynamic width stage. |
| `'depth'` | Train progressive shrinking's dynamic depth stage. |

## Citation
```BibTex
@inproceedings{
  shipard2022RSS,
  title={Does Interference Exist When Training a Once-For-All Network?},
  author={Jordan Shipard and Arnold Wiliem and Clinton Fookes},
  booktitle={Computer Vision and Pattern Recognition Workshop},
  year={2022},
  url={}
}
```

## Results
<img src="https://github.com/Jordan-HS/RSS-Interference-CVPRW2022/blob/main/figures/Training_speedup.png" width="45%" /> <img src="https://github.com/Jordan-HS/RSS-Interference-CVPRW2022/blob/main/figures/CIFAR100_Pop_performance_annotated.png" width="50%" />

## Requirments
- Python 3.6+
- PyTorch 1.4+
- MatPlotLib
