# TOHAN: A One-step Approach towards Few-shot Hypothesis Adaptation
Hi, this is the core code of our rencent work "TOHAN: A One-step Approach towards Few-shot Hypothesis Adaptation" (NeurIPS2021, https://arxiv.org/abs/2106.06326). This work is done by

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

# Citation
If you are using this code for your own researching, please consider citing

```
@article{chi2021tohan,
  author = {Haoang Chi and
               Feng Liu and
               Wenjing Yang and
               Long Lan and
               Tongliang Liu and
               Bo Han and
               William Cheung and
               James Kwok},
  title     = {TOHAN: A One-step Approach towards Few-shot Hypothesis Adaptation},
  journal   = {arXiv preprint arXiv:2106.06326},
  year      = {2021}
}
```
# Acknowledgment
This work was partially supported by the National Natural Science Foundation of China (No.
91948303-1, No. 61803375, No. 12002380, No. 62106278, No. 62101575, No. 61906210)
and the National Grand R&D Plan (Grant No. 2020AAA0103501). FL would also like to thank Dr.
Yanbin Liu for productive discussions.
