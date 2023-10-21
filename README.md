# CluSiam: Improving Representation Learning for Histopathologic Images with Clustering Constraints
Pytorch training code of the paper [Improving Representation Learning for Histopathologic Images with Cluster Constraints](https://arxiv.org/abs/2310.12334) 

## Contents

<!-- - [Visualization](#visualization) -->
- [Data preparing](#getting-started)
- [Unsupervised Pre-Training](#unsupervised-pre-training)
  - [CluSiam](#clusiam)
  - [CluBYOL](#clubyol)
- [Evaluation](#evaluation)
- [Repository Status](#repository-status)

## Visualization
<table>
  <tr>
    <td><img src=".github\tumor_076.png" alt="Image 1" width="400"/></td>
    <td><img src=".github\tumor_090.png" alt="Image 2" width="400"/></td>
  </tr>
  <tr>
    <td><img src=".github\tumor_085.png" alt="Image 3" width="400"/></td>
    <td><img src=".github\tumor_110.png" alt="Image 4" width="400"/></td>
  </tr>
</table>
In the visualizations provided, distinct color-filled regions depict different cluster assignments. The blue contours delineate the annotations made by human annotators.

While the clustering module, a by-product of representation learning, may not delineate positive regions with absolute precision, it's remarkable that the clustering module achieves this entirely through self-supervised means. Given this constraint, we believe the results are commendable. Furthermore, these cluster assignments can potentially be enhanced when enhanced with other methodologies.

## Data preparing
We processed the Camelyon16 dataset with all slides adjusted to a 20x magnification level for uniformity and ease of analysis. You can download the pre-processed patches from the link below:

[Download Processed Camelyon16 Dataset](https://www.dropbox.com/s/58j49j8vy2cwkpj/Camelyon_20xpatch.zip)

Please ensure you adhere to the Camelyon16 dataset's licensing and usage conditions.


## Self Supervised Pre-Training

During self supervised pre-training, we observed two kinds of mode collapse related to the cluster assigner:

1. **Dominant Dimension:** Here, the output of the cluster assigner is largely dominated by one dimension. Even when Gumbel noise is introduced, the final cluster assignment remains unaffected. Such a situation often arises when the weight distribution is like `1*invariance loss + 100*cluster loss`. By constraining the range of loss weights, this can be circumvented.

2. **Uniform Dimensions with Slight Variance:** In this scenario, the output from the cluster assigner is almost uniform across all dimensions. However, one dimension slightly exceeds the others. When Gumbel noise is added, the cluster assignment becomes random. Setting a small value for `alpha` can lead to this.

### CluSiam
For CluSiam training, we've set the default `alpha` to 0.5. This value was chosen because both the invariance loss and cluster loss are based on cosine similarities, although the cluster loss has a lower bound of `-1/(k-1)`. We later discovered that using a larger value here is also acceptable.

To initiate CluSiam unsupervised pre-training, execute:

```
python main.py \
--model_type clusiam \
--num_clusters 100 \
--feat_size 2048 \
--epochs 50 \
--alpha 0.5 \
--fix_pred_lr \
--batch_size 512 \
--save_path /path/to/save/model \
--log_dst /path/to/log/destination \
/path/to/your/dataset \
```


### CluBYOL
For CluBYOL training, we've chosen the default `alpha` to be 0.9. This decision stems from the fact that the invariance loss in CluBYOL is derived from the Euclidean distance, which spans a broader range than cosine similarity. We aim to rescale it to a range that aligns more closely with cosine similarity.

To start CluBYOL unsupervised pre-training, run:

```
python main.py \
--model_type clubyol \
--num_clusters 100 \
--feat_size 256 \
--epochs 50 \
--alpha 0.9 \
--batch_size 512 \
--save_path /path/to/save/model \
--log_dst /path/to/log/destination \
/path/to/your/dataset \
```

## Evaluation

The AUC values are calculated and reported using a methodology similar to [DSMIL](https://github.com/binli123/dsmil-wsi/blob/master/train_tcga.py), for class-wise prediction scores.

We compare our representation's performance on whole slide-level classification and patch-level top-1 KNN classification (non-parameter) with other methods.

### WSIs-level Performance with DSMIL
| Rep.       | Acc.  |AUC (Neg.)|AUC (Pos.)|
|------------|-------|----------|----------|
| Supervised | 0.651 |   0.635  |   0.635  |
| SimCLR     | 0.822 |   0.845  |   0.874  |
| SwAV       | 0.876 |   0.866  |   0.859  |
| PCL        | 0.488 |   0.535  |   0.496  |
| Barlow.    | 0.860 |   0.873  |   0.945  |
| BYOL       | 0.558 |   0.501  |   0.586  |
| CluBYOL    | 0.923 |   0.947  |   0.975  |
| SimSiam    | 0.721 |   0.656  |   0.680  |
| CluSiam    | 0.907 |   0.945  |   0.952  |

### Patch-level Performance with Top-1 KNN
| Rep.       |F1 Scores|
|------------|-------|
| SimCLR     | 0.656 |
| SwAV       | 0.615 |
| PCL        | 0.382 |
| Barlow.    | 0.681 |
| BYOL       | 0.441 |
| CluBYOL    | 0.768 |
| SimSiam    | 0.511 |
| CluSiam    | 0.730 |

## Citing Our Work

If our research proves beneficial or serves as an inspiration for your work, we kindly request that you reference our publication:

```bibtex
@inproceedings{wu2023improving,
  title={Improving Representation Learning for Histopathologic Images with Cluster Constraints},
  author={Wu, Weiyi and Gao, Chongyang and DiPalma, Joseph and Vosoughi, Soroush and Hassanpour, Saeed},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21404--21414},
  year={2023}
}
```


