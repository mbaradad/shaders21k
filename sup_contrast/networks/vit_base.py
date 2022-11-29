"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel

from transformers import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEEmbeddings
from torch.nn import init

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class ModelWithHead(nn.Module):
    def __init__(self, model, cls_token_pos, embedding_dim, global_pool):
        super().__init__()

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = nn.LayerNorm(eps=1e-6, normalized_shape=embedding_dim)

        vit_config_emedding = ViTMAEConfig()
        vit_config_emedding.hidden_size = embedding_dim

        img_to_text_embedder = ViTMAEEmbeddings(vit_config_emedding)

        # enable optimization also for position embedings:
        img_to_text_embedder.position_embeddings.requires_grad = True
        # init positional embedding the same as BERT
        # which is the same as the default nn.Embeddings initialization, init.normal_(self.weight)
        #imshow(img_to_text_embedder.position_embeddings, title='position_embeddings_before_init')

        init.normal_(img_to_text_embedder.position_embeddings)
        #imshow(img_to_text_embedder.position_embeddings, title='position_embeddings_after_init')

        # also set the positional embeddings to 0 for the model,
        # so that when we forward input_embeds it doesn't add extra positional embeddings,
        # see transformers.models.roberta.modeling_roberta.py L129
        model.embeddings.position_embeddings.weight.data = model.embeddings.position_embeddings.weight.data * 0

        self.embeddings = img_to_text_embedder
        self.model = model
        self.cls_token_pos = cls_token_pos

    def init_weights(self):
        self.model.init_weights()

    def forward(self, imgs):
        input_embeds = self.embeddings(imgs)
        features = self.model(inputs_embeds=input_embeds[0])
        if self.global_pool:
            features = features['last_hidden_state'][:, 1:, :].mean(dim=1)  # global pool without cls token
            features = self.fc_norm(features)
        else:
            features = features['last_hidden_state'][:, self.cls_token_pos, :]
        prediction = self.head(features)
        return prediction


class SupConVitBase(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128, global_pool=False):
        super(SupConVitBase, self).__init__()

        self.global_pool = global_pool

        base_model = RobertaModel.from_pretrained('roberta-base')

        self.cls_token_pos = 0
        self.embedding_dim = 768

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = nn.LayerNorm(eps=1e-6, normalized_shape=self.embedding_dim)

        vit_config_emedding = ViTMAEConfig()
        vit_config_emedding.hidden_size = self.embedding_dim

        img_to_text_embedder = ViTMAEEmbeddings(vit_config_emedding)

        # enable optimization also for position embedings:
        img_to_text_embedder.position_embeddings.requires_grad = True
        # init positional embedding the same as BERT
        # which is the same as the default nn.Embeddings initialization, init.normal_(self.weight)
        #imshow(img_to_text_embedder.position_embeddings, title='position_embeddings_before_init')

        init.normal_(img_to_text_embedder.position_embeddings)
        #imshow(img_to_text_embedder.position_embeddings, title='position_embeddings_after_init')

        # also set the positional embeddings to 0 for the model,
        # so that when we forward input_embeds it doesn't add extra positional embeddings,
        # see transformers.models.roberta.modeling_roberta.py L129
        base_model.embeddings.position_embeddings.weight.data = base_model.embeddings.position_embeddings.weight.data * 0

        self.embeddings = img_to_text_embedder

        base_model.init_weights()

        self.encoder = base_model
        if head == 'linear':
            self.head = nn.Linear(self.embedding_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def get_features(self, x):
        upscaled_x = F.interpolate(x, (224,224))
        input_embeds = self.embeddings(upscaled_x)
        features = self.encoder(inputs_embeds=input_embeds[0])
        if self.global_pool:
            features = features['last_hidden_state'][:, 1:, :].mean(dim=1)  # global pool without cls token
            features = self.fc_norm(features)
        else:
            features = features['last_hidden_state'][:, self.cls_token_pos, :]
        return features

    def forward(self, x):
        features = self.get_features(x)

        feat = F.normalize(self.head(features), dim=1)
        return feat

class VitBaseLinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=10):
        super(VitBaseLinearClassifier, self).__init__()
        feat_dim = 768
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
