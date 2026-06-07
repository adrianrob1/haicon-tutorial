# haicon-tutorial
IPython notebooks to follow along for the Helmholtz AI Conference (HAICON) 2026 tutorial on 'Deep Learning with Bayesian Principles'. 

## Installation and Getting Started

Get `uv` and run

```shell
uv sync --locked
```

Then, try out the python notebooks to follow along :) 

## More IVON Examples
* Official repository of the IVON optimizer: [http://github.com/team-approx-bayes/ivon](http://github.com/team-approx-bayes/ivon)
* Many practical examples from the original paper: [http://github.com/team-approx-bayes/ivon-experiments](http://github.com/team-approx-bayes/ivon-experiments)
* Federated learning with IVON: [http://github.com/team-approx-bayes/bayes-admm](http://github.com/team-approx-bayes/bayes-admm)
* Low-rank adaptation with IVON: [https://github.com/team-approx-bayes/ivon-lora](https://github.com/team-approx-bayes/ivon-lora)
* IVON and Edge-of-Stability: [https://github.com/Avra98/variationallearning_eos](https://github.com/Avra98/variationallearning_eos)

## Recommended Papers

### Variational Learning / Bayesian Neural Networks

- [**Variational Learning is Effective for Large Deep Networks**](https://arxiv.org/abs/2402.17641)  
  Yuesong Shen, Nico Daheim, Bai Cong, Peter Nickl, Gian Maria Marconi, Clement Bazan, Rio Yokota, Iryna Gurevych, Daniel Cremers, Mohammad Emtiyaz Khan, Thomas Möllenhoff. ICML 2024.

- [**Keeping the Neural Networks Simple by Minimizing the Description Length of the Weights**](https://dl.acm.org/doi/10.1145/168304.168306)  
  Geoffrey E. Hinton and Drew van Camp. COLT 1993.  
  Classic early variational/MDL treatment of neural-network weights.

- [**Practical Variational Inference for Neural Networks**](https://www.cs.toronto.edu/~graves/nips_2011.pdf)  
  Alex Graves. NeurIPS 2011.

- [**Practical Deep Learning with Bayesian Principles**](https://arxiv.org/abs/1906.02506)  
  Kazuki Osawa, Siddharth Swaroop, Anirudh Jain, Runa Eschenhagen, Richard E. Turner, Rio Yokota, Mohammad Emtiyaz Khan. NeurIPS 2019.

### PAC-Bayes / Generalization

- [**User-friendly Introduction to PAC-Bayes Bounds**](https://arxiv.org/abs/2110.11216)  
  Pierre Alquier. Foundations and Trends in Machine Learning, 2024; arXiv version 2021.

- [**PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization**](https://arxiv.org/abs/2211.13609)  
  Sanae Lotfi, Marc Finzi, Sanyam Kapoor, Andres Potapczynski, Micah Goldblum, Andrew Gordon Wilson. NeurIPS 2022.

- [**Deep Learning is Not So Mysterious or Different**](https://arxiv.org/abs/2503.02113)  
  Andrew Gordon Wilson. ICML 2025 Position Paper.

### Laplace Approximation / Post-hoc Bayesian Deep Learning

- [**Laplace Redux — Effortless Bayesian Deep Learning**](https://arxiv.org/abs/2106.14806)  
  Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, Philipp Hennig. NeurIPS 2021.  
  Introduces and motivates the `laplace` PyTorch library.

- [**Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning**](https://arxiv.org/abs/2104.04975)  
  Alexander Immer, Matthias Bauer, Vincent Fortuin, Gunnar Rätsch, Mohammad Emtiyaz Khan. ICML 2021.

- [**Improving Predictions of Bayesian Neural Nets via Local Linearization**](https://arxiv.org/abs/2008.08400)  
  Alexander Immer, Maciej Korzepa, Matthias Bauer. AISTATS 2021.

- [**laplace-torch: Laplace Approximations for Deep Learning**](https://aleximmer.github.io/Laplace/)  
  Official documentation and implementation of the `laplace` PyTorch library.
