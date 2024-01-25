# Spatial dependency-aware deep generative models

SpaVAE, spaPeakVAE, spaMultiVAE, spaLDVAE and spaPeakLDVAE are dependency-aware deep generative models for multitasking analysis of spatial genomics data. Different models are designed for different analytical tasks of spatial genomics data.<br/>
**spaVAE** is a negative binomial (NB) model-based variational autoencoder (VAE) with a hybrid embedding of Gaussian process (GP) prior and Gaussian prior. The model is for multitasking analysis of spatially resolved transcriptomics (SRT) data, including dimensionality reduction, visualization, clustering, batch integration, denoising, differential expression, spatial interpolation, and  resolution enhancement.<br/>
**spaPeakVAE** is a variant model of spaVAE, which uses a Bernoulli decoder to characterize spatial ATAC-seq binary data. The analytical tasks in spaVAE can also be fulfilled by spaPeakVAE for spatial ATAC-seq data.<br/>
**spaMultiVAE** characterizes spatial multi-omics data, which profiles gene expression and surface protein intensity simultaneously. Besides the analyses aforementioned, spaMultiVAE uses a NB mixture decoder to denoise backgrounds in proteins.<br/>
**spaLDVAE** and **spaPeakLDVAE** are spaVAE variants with a linear decoder, which also contains two hybrid latent embedding components, one follows GP prior and the other follows standard normal prior. The model can be used for detecting spatial variable genes and peaks. 

## Table of contents
- [Network diagram](#diagram)
- [Requirements](#requirements)
- [Folders](#folders)
- [Usage](#usage)
- [API documents](#api)
- [Parameters](#parameters)
- [Datasets](#datasets)
- [Reference](#reference)
- [Contact](#contact)

## <a name="diagram"></a>Network diagram

Diagram of spaVAE (**a**), spaPeakVAE (**a**), spaMultiVAE (**b**), spaLDVAE (**c**), and spaPeakLDVAE (**c**) networks:

<img src="https://github.com/ttgump/spaVAE/blob/main/network.svg" width="800" height="800">

## <a name="requirements"></a>Requirements

Python: 3.9.7<br/>
PyTorch: 1.11.0 (https://pytorch.org)<br/>
Scanpy: 1.9.1 (https://scanpy.readthedocs.io/en/stable)<br/>
Numpy: 1.21.5 (https://numpy.org)<br/>
Pandas: 1.4.2 (https://pandas.pydata.org)<br/>
h5py: 3.6.0 (https://pypi.org/project/h5py)<br/>

## <a name="folders"></a>Folders

**[Tutorial](https://github.com/ttgump/spaVAE/tree/main/Tutorial)**: we provide Jupyter notebooks demonstrating the functionalities of the spaVAE models.<br/>
**[src](https://github.com/ttgump/spaVAE/tree/main/src)**: source code of spaVAE models.<br/>

## <a name="usage"></a>Usage

For human DLPFC dataset:

```sh
python run_spaVAE.py --data_file HumanDLPFC_151673.h5 --inducing_point_steps 6
```

For integrating 4 human DLPFC samples:

```sh
python run_spaVAE_Batch.py --data_file 151673_151674_151675151676_samples_union.h5 --inducing_point_steps 6
```

For mouse hippocampus Slide-seq V2 dataset:

```sh
python run_spaVAE.py --data_file Mouse_hippocampus.h5 --grid_inducing_points False --inducing_point_nums 400 --loc_range 40
```

For spatial ATAC-seq dataset of mouse embryonic (E15.5) brain tissues in the MISAR-seq dataset:

```sh
python run_spaPeakVAE.py --data_file MISAR_seq_mouse_E15_brain_ATAC_data.h5 --inducing_point_steps 19
```

For spatial multi-omics Spatial-CITE-seq data:

```sh
python run_spaMultiVAE.py --data_file Multiomics_Spatial_Human_tonsil_SVG_data.h5 --inducing_point_steps 19
```

--data_file specifies the data file name, in the h5 file. For SRT data, spot-by-gene count matrix is stored in "X" and 2D location is stored in "pos". For spatial ATAC-seq data, "X" represents spot-by-peak count matrix. For spatial multi-omics data, "X_gene" represents spot-by-gene count matrix, and "X_protein" represents spot-by-protein count matrix.

## <a name="api"></a>API documents

[API documents](https://github.com/ttgump/spaVAE/wiki)

## <a name="parameters"></a>Parameters
**--data_file:** data file name.<br/>
**--select_genes:** number of selected genes for analysis, default = 0 means no filtering.  It will use the mean-variance relationship to select informative genes.<br/>
**--batch_size:** mini-batch size, default = "auto", which means if sample size <= 1024 then batch size = 128, if 1024 < sample size <= 2048 then batch size = 256, if sample size > 2048 then batch size = 512.<br/>
**--maxiter:** number of max training iterations, default = 5000.<br/>
**--train_size:** proportion of training set, others will be validating set, default = 0.95. In small datasets, e.g. there are only hundreds of spots, we recommend to set train_size to 1, and fix the maxiter to 1000.<br/>
**--patience:** patience of early stopping when using validating set, default = 200.<br/>
**--lr:** learning rate, default = 1e-3 for spaVAE and spaPeakVAE, and defualt = 5e-3 for spaMultiVAE.<br/>
**--weight_decay:** weight decay coefficient, default = 1e-6.<br/>
**--noise:** coefficient of random Gaussian noise for the encoder, default = 0.<br/>
**--dropoutE:** dropout probability for encoder, default = 0.<br/>
**--dropoutD:** dropout probability for decoder, default = 0.<br/>
**--encoder_layers:** hidden layer sizes of encoder, default = [128, 64].<br/>
**--GP_dim:** dimension of the latent Gaussian process embedding, default = 2 for spaVAE and spaMultiVAE, and default = 5 for spaPeakVAE.<br/>
**--Normal_dim:** dimension of the latent standard Gaussian embedding, default = 8 for spaVAE and spaPeakVAE, and 18 for spaMultiVAE.<br/>
**--decoder_layers:** hidden layer sizes of decoder, default = [128].<br/>
**--dynamicVAE:** whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value.<br/>
**--init_beta:** initial coefficient of the KL loss, default = 10.<br/>
**--min_beta:** minimal coefficient of the KL loss, default = 4.<br/>
**--max_beta:** maximal coefficient of the KL loss, default = 25. min_beta, max_beta, and KL_loss are used for dynamic VAE algorithm.<br/>
**--KL_loss:** desired KL_divergence value (GP and standard normal combined), default = 0.025.<br/>
**--num_samples:** number of samplings of the posterior distribution of latent embedding during training, default = 1.<br/>
**--fix_inducing_points:** fixed or trainable inducing points, default = True, which means inducing points are fixed.<br/>
**--grid_inducing_points:** whether to use 2D grid inducing points or k-means centroids of positions as inducing points, default = True. "True" for 2D grid, "False" for k-means centroids.<br/>
**--inducing_point_steps:** if using 2D grid inducing points, set the number of 2D grid steps, default = None. Needed when grid_inducing_points = True.<br/>
**--inducing_point_nums:** if using k-means centroids on positions, set the number of inducing points, default = None. Needed when grid_inducing_points = False.<br/>
**--fixed_gp_params:** kernel scale is fixed or not, default = False, which means kernel scale is trainable.<br/>
**--loc_range:** positional locations will be scaled to the specified range. For example, loc_range = 20 means x and y locations will be scaled to the range 0 to 20, default = 20. This value can be set larger if it isn't numerical stable during training.<br/>
**--kernel_scale:** initial kernel scale, default = 20.<br/>
**--model_file:** file name to save weights of the model, default = model.pt<br/>
**--final_latent_file:** file name to output final latent representations, default = final_latent.txt.<br/>
**--denoised_counts_file:** file name to output denoised counts, default = denoised_mean.txt.<br/>
**--device:** pytorch device, default = cuda.<br/>

## <a name="datasets"></a>Datasets

Datasets used in the study can be found

https://figshare.com/articles/dataset/Spatial_genomics_datasets/21623148

## <a name="reference"></a>Reference

## <a name="contact"></a>Contact

Tian Tian tt72@njit.edu
