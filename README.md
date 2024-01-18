# INTEGRATIVE GRAPH CONVOLUTIONAL NETWORKS (IGCN)
<img width="1275" alt="IGCN_2" src="https://github.com/bozdaglab/IGCN/assets/71089544/149bdfd7-7d4e-40e6-937c-5b7c13c2240a">

Integrative Graph Convolutional Networks (IGCN) operates on multi-modal data structures. In IGCN, a multi-GCN module is initially employed to extract node embeddings from each network.
An attention module is then proposed to fuse the multiple node embeddings into a weighted form. 
In contrast to earlier research, the attention mechanism allocates distinct attention coefficients to each node or sample,
aiding in the identification of which data modalities are given greater emphasis for predicting a specific class type. Therefore,
we can fuse the multiple node embeddings using element-wise
multiplication denoted as ' $\otimes$ ' in the integration layer.
Finally, a prediction module utilizes fused node
embeddings on multiple similarity networks and takes advantage of multiple graph topologies to determine the final predictions.

# How to Run IGCN
After having all files in the directory, run `IGCN.py`.
### Inputs
TCGA-BRCA dataset was employed for the training and testing of IGCN. 
* The preprocessed multi-omics features and ground truth labels can be found in the dataset folder.
   * `1_.csv`: Gene expression features
   * `2_.csv`: miRNA expression features
   * `3_.csv`: DNA methylation features
   * `labels_.csv`: Ground truth labels
# Costumize IGCN
* We note that users can adjust the following parameters in `IGCN.py` (lines 12-15) for quick testing of IGCN
   * Reduce `xtimes1`, and `xtimes2` to a small number, like 3.
   * Retain only one element in the learning_rates list, for example, [0.001].
   * Retain only one element in the hd_sizes list, for example, [256].
* Users can also adjust the following parameters of `IGCN.py` and `main.py`
   * `max_epochs`: maximum number of epoch (line 9 of `IGCN.py`, default is 1000) 
   * `min_epochs`: minimum number of epoch line 10 of `IGCN.py`, default is 200)
   * `patients`: patience for early stopping (line 11 of `IGCN.py`, default is 30)
   *  `thereshold`: a threshold to construct the adjancecy matrices (line 10 of `main.py`, default is 3)

