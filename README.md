# Optimal disease surveillance with graph-based Active Learning

​​**Joseph L.-H. Tsui**, **Mengyan Zhang**, Prathyush Sambaturu, Simon Busch-Moreno, Marc A. Suchard, Oliver G. Pybus, **Seth Flaxman**, **Elizaveta Semenova**, **Moritz U. G. Kraemer**

[![DOI](https://zenodo.org/badge/817429535.svg)](https://zenodo.org/doi/10.5281/zenodo.12198920)

---

This repository contains data and scripts used to generate results
presented in [citation]. Please note that some scripts may need small adjustments to run depending on local setup.

## Abstract

_Tracking the spread of emerging pathogens is critical to the design of timely and effective public health responses. Policymakers face the challenge of allocating finite resources for testing and surveillance across locations, with the goal of maximising the information obtained about the underlying trends in prevalence and incidence. We model this decision-making process as an iterative node classification problem on an undirected and unweighted graph, in which nodes represent locations and edges represent movement of infectious agents among them. To begin, a single node is selected for testing and determined to be either infected or uninfected. Test feedback is then used to update estimates of the probability of unobserved nodes being infected and to inform the selection of nodes for testing at the next iterations, until a certain resource budget is exhausted. Using this framework we evaluate and compare the performance of previously developed Active Learning policies, including node-entropy and Bayesian Active Learning by Disagreement. We explore the performance of these policies under different outbreak scenarios using simulated outbreaks on both synthetic and empirical networks. Further, we propose a novel policy that considers the distance-weighted average entropy of infection predictions among the neighbours of each candidate node. Our proposed policy outperforms existing ones in most outbreak scenarios, leading to a reduction in the number of tests required to achieve a certain predictive accuracy. Our findings could inform the design of cost-effective surveillance strategies for emerging and endemic pathogens, and reduce the uncertainties associated with early risk assessments in resource-constrained situations._

## Repository usage and structure

The structure of this repository is shown below:

```
├── code/
│   ├── binary_prevalence/
│       ├── active_allocation/
│       ├── passive_allocation/
│       ├── agents/
│       ├── evaluation/
│       ├── outbreak_simulation/
│       └── surrogate/
│   ├── run-scripts/
│   ├── tutorial_resources/
│   ├── utils/
│   └── configs/
├── materials/
│   ├── synthetic-graphs/
│       ├── SB-high_modularity/
│       └── SB-low_modularity/
│   ├── empirical-graphs/
│       ├── between-country-iata/
│       └── within-country-italy/
│   └── maps/
│       ├── Municipal_Boundaries_of_Italy_2019/
│       └── world-administrator-boundaries/
├── LICENSE.md
└── README.md
```

## Empirical human mobility data

Both empirical human mobility datasets which we used in the study are publicly available and can be downloaded from the following sources:
- [within-country mobility data at provincial level in Italy, 2020](https://data.humdata.org/dataset/covid-19-mobility-italy)
- [between-country air traffic data at country level, 2020](https://zenodo.org/records/7472836)

---