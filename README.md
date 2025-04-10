# CDEKI
Competitive Differential Evolution with Knowledge Inheritance for Single-Objective Human-Powered Aircraft Design

## Abstract
This paper introduces a novel and efficient optimizer derived from differential evolution (DE): Competitive differential evolution with knowledge inheritance (CDEKI). CDEKI is developed by introducing the competitive mechanism and incorporates a novel DE/winner-to-best/1 mutation strategy and a hybrid local search operation. Moreover, we emphasize the significance of a commonly neglected component in DE: the repair operation. We propose a novel repair operation with knowledge inheritance to accelerate optimization convergence, particularly when the generated offspring exceeds the search domain. Through comprehensive numerical experiments conducted on the CEC2020 benchmark functions, competing against sixteen state-of-the-art optimizers and advanced DE variants, our proposed CDEKI demonstrates significant superiority. Additionally, the ablation experiments are conducted to independently investigate the performance of the proposed strategies. Furthermore, we extend the application of CDEKI to real-world human-powered aircraft (HPA) design tasks, showcasing its extraordinary performance in practical scenarios. As an effective and efficient optimizer, CDEKI presents a compelling alternative evolutionary approach for addressing real-world applications across diverse domains. The source code of CDEKI is available in https://github.com/RuiZhong961230/CDEKI.

## Citation
@article{Zhong:25,  
title = {Competitive Differential Evolution with Knowledge Inheritance for Single-Objective Human-Powered Aircraft Design},  
journal = {The Journal of Supercomputing},  
volume = {81},  
pages = {721},  
year = {2025},  
author = {Rui Zhong and Yang Cao and Enzhi Zhang and Masaharu Munetomo},  
doi = {https://doi.org/10.1007/s11227-024-06859-3 }  
}  

## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively. The Human-Powered Aircraft (HPA) Design problems can be downloaded from https://github.com/Nobuo-Namura/hpa.

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp
