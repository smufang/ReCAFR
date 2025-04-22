# A Contrastive Framework with User, Item and Review Alignment for Recommendation
 This is the PyTorch implementation by for ReCAFR model proposed in this [paper](https://dl.acm.org/doi/abs/10.1145/3701551.3703530):

 >*WSDM 2025*


## 📝 Environment

Then run the following commands to create a conda environment:

```bash
python=3.9
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install pyyaml tqdm
```


## Review-basedRecommendation Dataset

We utilized  public datasets Amazon-book

Each user and item has a text informations.


Dataset consists of a training set, a validation set, and a test set. During the training process, we utilize the validation set to determine when to stop the training in order to prevent overfitting.
```
- amazon(yelp/steam)
|--- trn_mat.pkl    # training set (sparse matrix)
|--- val_mat.pkl    # validation set (sparse matrix)
|--- tst_mat.pkl    # test set (sparse matrix)
|--- usr_emb_np.pkl # user text embeddings
|--- itm_emb_np.pkl # item text embeddings
```

### text Representation
- Each user and item has a semantic embedding encoded from its own profile using **Text Embedding Models**.
- The encoded semantic embeddings are stored in `usr_emb_np.pkl` and `itm_emb_np.pkl`.

We provide the **mapping dictionary** in JSON format in the `data/mapper` folder to map the `user/item ID` in our processed data to the `original identification` in original data (e.g., asin for items in Amazon-book).

## Examples to run the codes

The command to evaluate the backbone models and RLMRec is as follows. 
**(Constrastive Alignment)**:

    ```python encoder/train_encoder.py```

Supported models/datasets:

* model_name:`lightgcn`, 
* dataset: `amazon`



## 🌟 Citation
If you find this work is helpful to your research, please consider citing our paper:
```bibtex
@inproceedings{v2025contrastive,
  title={A contrastive framework with user, item and review alignment for recommendation},
  author={V. Dong, Hoang and Fang, Yuan and Lauw, Hady W},
  booktitle={Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining},
  pages={117--126},
  year={2025}
}
```

**Thanks for your interest in our work!**
