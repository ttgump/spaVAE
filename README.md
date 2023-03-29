# spaVAE
SpaVAE, spaPeakVAE and spaMultiVAE are dependency-aware deep generative models for multitasking analysis of spatial genomics data. The spaVAE model (**a**) optimizes the parameters of a deep neural network to approximate the distributions that underlie the SRT data and introduces a Gaussian process (GP) prior to explicitly capture spatial correlations among spots. As a result, we can use spaVAE for various analyses, including dimensionality reduction, visualization, clustering, batch integration, denoising, differential expression, spatial imputation, and  resolution enhancement. SpaPeakVAE (**a**) is a variant model of spaVAE, which characterizes spatial ATAC-seq binary data. SpaMultiVAE (**b**) is based on spaVAE, which characterizes spatial multi-omics data that profiles gene expression and surface protein intensity simultaneously. SpaLDVAE (**c**) is spaVAE with a linear decoder, and can be used for detecting spatial variable genes and peaks. 

Diagram of spaVAE (**a**), spaPeakVAE (**a**), spaMultiVAE (**b**), and spaLDVAE (**c**) networks:
![alt text](https://github.com/ttgump/spaVAE/blob/main/network.png?raw=True)

**Requirements**

Python: 3.9.7<br/>
PyTorch: 1.11.0<br/>
Scanpy: 1.9.1<br/>
Numpy: 1.21.5<br/>
Scipy: 1.8.0<br/>
Pandas: 1.4.2<br/>
h5py: 3.6.0<br/>

**Usage**

For human DLPFC dataset:

```sh
python run_spaVAE.py --data_file HumanDLPFC_151673.h5 --noise 1 --inducing_point_steps 6
```
For mouse hippocampus Slide-seq V2 dataset:

```sh
python run_spaVAE.py --data_file Mouse_hippocampus.h5 --grid_inducing_points False --inducing_point_nums 300
```

For ATAC-seq dataset of mouse embryonic (E15.5) brain tissues in the MISAR-seq dataset:

```sh
python run_spaPeakVAE.py --data_file MISAR_seq_mouse_E15_brain_ATAC_data.h5 --inducing_point_steps 19
```

For spatial multi-omics DBiT-seq data:

```sh
python run_spaMultiVAE.py --data_file Multiomics_DBiT_seq_0713_data.sh --inducing_point_steps 15
```

--data_file specifies the data file name, in the h5 file, spot-by-gene count matrix is stored in "X" and 2D location is stored in "pos".

**Parameters**

--data_file: data file name.<br/>
--select_genes: number of selected genes for embedding analysis, default = 0 means no filtering.<br/>
--batch_size: batch size, default = 512.<br/>
--maxiter: number of max training iterations, default = 2000.<br/>
--lr: learning rate, default = 1e-3.<br/>
--weight_decay: weight decay coefficient, default = 1e-2.<br/>
--noise: coefficient of random Gaussian noise for the encoder, default = 0.<br/>
--dropoutE: dropout probability for encoder, default = 0.<br/>
--dropoutD: dropout probability for decoder, default = 0.<br/>
--encoder_layers: hidden layer sizes of encoder, default = [128, 64, 32].<br/>
--z_dim: size of bottleneck layer, default = 2.<br/>
--decoder_layers: hidden layer sizes of decoder, default = [32].<br/>
--beta: coefficient of the reconstruction loss, default = 20.<br/>
--num_samples: number of samplings of the posterior distribution of latent embedding, default = 1<br/>
--fix_inducing_points: fixed or trainable inducing points, default = True<br/>
--grid_inducing_points: whether to use 2D grid inducing points or k-means centroids on positions, default = True<br/>
--inducing_point_steps: if using 2D grid inducing points, set the number of 2D grid steps, default = None<br/>
--inducing_point_nums: if using k-means centroids on positions, set the number of inducing points, default = None<br/>
--fixed_gp_params: kernel scale is trainable or not, default = True<br/>
--loc_range: range of locations scaling to. For example, loc_range = 20 means x and y locations will be scaled to the range 0 to 20, default = 20.<br/>
--kernel_scale: initial kernel scale, default = 20.<br/>
--model_file: file name to save weights of the model, default = model.pt<br/>
--final_latent_file: file name to output final latent representations, default = final_latent.txt.<br/>
--denoised_counts_file: file name to output denoised counts, default = denoised_mean.txt.<br/>
--device: pytorch device, default = cuda.<br/>

**Datasets used in the study can be found**

https://figshare.com/articles/dataset/Spatial_genomics_datasets/21623148
