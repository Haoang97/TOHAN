# TOHAN: A One-step Approach towards Few-shot Hypothesis Adaptation
Hi, this is the core code of our rencent work "TOHAN: A One-step Approach towards Few-shot Hypothesis Adaptation" (NeurIPS 2021 Spotlight, https://arxiv.org/abs/2106.06326). This work is done by

- Haoang Chi (NUDT), haoangchi618@gmail.com
- Dr. Feng Liu (UTS), feng.liu@uts.edu.au
- Dr. Wenjing Yang (NUDT), wenjing.yang@nudt.edu.cn
- Dr. Long Lan (NUDT), long.lan@nudt.edu.cn
- Dr. Tongliang Liu (USYD), tongliang.liu@sydney.edu.au
- Dr. Bo Han (HKBU), bhanml@comp.hkbu.edu.hk
- Prof. William Cheung (HKBU), william@comp.hkbu.edu.hk
- Prof. James Kwok (HKUST), jamesk@cse.ust.hk

# Software version
Torch version is 1.7.1. Python version is 3.7.6. CUDA version is 11.0.

These python files, of cause, require some basic scientific computing python packages, e.g., numpy. I recommend users to install python via Anaconda (python 3.7.6), which can be downloaded from https://www.anaconda.com/distribution/#download-section . If you have installed Anaconda, then you do not need to worry about these basic packages.

After you install anaconda and pytorch (gpu), you can run codes successfully.

# TOHAN
Please feel free to test the TOHAN method by running main.py.

Specifically, please run

```
CUDA_VISIBLE_DEVICES=0 python main.py
```

in your terminal (using the first GPU device).

Our pre-trained models can be downloaded in the following link.

https://drive.google.com/drive/folders/1IIRJSDvJ9WYVGbZRNOpkU0STdVfmcjfb?usp=sharing

# Two bugs
We just found two bugs in the source codes regrettably.
1. We did not use `encoder.eval()` and `classifier.eval()` to completely freeze the source model in the generation process.
2. The implementation of the augmented L1 distance degraded into the naive L1 distance.

We have repaired the two bugs and the modified results are coming soon. Sorry for the trouble caused to you.

# Original reproducible results
For digital tasks,

| Task | Method | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
|------|--------|------|------|------|------|------|------|------|
| M2S  | S+FADA | 25.6 | 27.7 | 27.8 | 28.2 | 28.4 | 29.0 | 29.6 |
|      | T+FADA | 25.3 | 26.3 | 28.9 | 29.1 | 29.2 | 31.9 | 32.4 |
|      | TOHAN  | 26.7 | 28.6 | 29.5 | 29.6 | 30.5 | 32.1 | 33.2 |
| S2M  | S+FADA | 74.4 | 83.1 | 83.3 | 85.9 | 86.0 | 87.6 | 89.1 |
|      | T+FADA | 74.2 | 81.6 | 83.4 | 82.0 | 86.2 | 87.2 | 88.2 |
|      | TOHAN  | 76.0 | 83.3 | 84.2 | 86.5 | 87.1 | 88.0 | 89.7 |
| M2U  | S+FADA | 83.7 | 86.0 | 86.1 | 86.5 | 86.8 | 87.0 | 87.2 |
|      | T+FADA | 84.2 | 84.2 | 85.2 | 85.2 | 86.0 | 86.8 | 87.2 |
|      | TOHAN  | 87.7 | 88.3 | 88.5 | 89.3 | 89.4 | 90.0 | 90.4 |
| U2M  | S+FADA | 83.2 | 84.0 | 85.0 | 85.6 | 85.7 | 86.2 | 87.2 |
|      | T+FADA | 82.9 | 83.9 | 84.7 | 85.4 | 85.6 | 86.3 | 86.6 |
|      | TOHAN  | 84.0 | 85.2 | 85.6 | 86.5 | 87.3 | 88.2 | 89.2 |
| S2U  | S+FADA | 72.2 | 73.6 | 74.7 | 76.2 | 77.2 | 77.8 | 79.7 |
|      | T+FADA | 71.7 | 74.3 | 74.5 | 75.9 | 77.7 | 76.8 | 79.7 |
|      | TOHAN  | 75.8 | 76.8 | 79.4 | 80.2 | 80.5 | 81.1 | 82.6 |
| U2S  | S+FADA | 28.1 | 28.7 | 29.0 | 30.1 | 30.3 | 30.7 | 30.9 |
|      | T+FADA | 27.5 | 27.9 | 28.4 | 29.4 | 29.5 | 30.2 | 30.4 |
|      | TOHAN  | 29.9 | 30.5 | 31.4 | 32.8 | 33.1 | 34.0 | 35.1 |

For objective tasks,
|       | S+FADA | T+FADA | TOHAN |
|-------|--------|--------|-------|
| CF2SL | 72.1   | 71.3   | 72.8  |
| SL2CF | 56.9   | 55.8   | 56.6  |

# Citation
If you are using this code for your own researching, please consider citing

```
@inproceedings{chi2021tohan,
 author = {Chi, Haoang and Liu, Feng and Yang, Wenjing and Lan, Long and Liu, Tongliang and Han, Bo and Cheung, William and Kwok, James},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {TOHAN: A One-step Approach towards Few-shot Hypothesis Adaptation},
 year = {2021}
}
```
# Acknowledgment
This work was partially supported by the National Natural Science Foundation of China (No.
91948303-1, No. 61803375, No. 12002380, No. 62106278, No. 62101575, No. 61906210)
and the National Grand R&D Plan (Grant No. 2020AAA0103501). FL would also like to thank Dr.
Yanbin Liu for productive discussions.
