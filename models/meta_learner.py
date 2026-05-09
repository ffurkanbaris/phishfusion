import torch
import torch.nn as nn
from .anatomical import AnatomicalBranch
from .lexical import LexicalBranch
from .statistical import StatisticalBranch

class PhishFusion(nn.Module):
    def __init__(self):
        super(PhishFusion, self).__init__()
        
        # 1. Expert Expert Branches
        self.anatomical_branch = AnatomicalBranch()
        self.lexical_branch = LexicalBranch()
        self.statistical_branch = StatisticalBranch()
        
        # 2. Post-Branch Normalization
        # These ensure that the 32D outputs have mean=0 and var=1 
        # before entering the Cross-Modal Attention.
        self.norm_anatomical = nn.BatchNorm1d(32)
        self.norm_lexical = nn.BatchNorm1d(32)
        self.norm_statistical = nn.BatchNorm1d(32)
        
        # 3. Cross-Modal Attention Layer
        self.attention_layer = nn.TransformerEncoderLayer(
            d_model=32, 
            nhead=4,
            dim_feedforward=64, 
            dropout=0.1,
            batch_first=True
        )
        self.cross_modal_transformer = nn.TransformerEncoder(self.attention_layer, num_layers=1)

        # 4. Meta-Learner
        self.meta_learner = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    def forward(self, qr_features, url_tokens, stat_features):
        # Step 1: Generate Raw Branch Embeddings (32D)
        v_a = self.anatomical_branch(qr_features)     
        v_l = self.lexical_branch(url_tokens)           
        v_s = self.statistical_branch(stat_features) 

        # Step 2: Apply BatchNorm1d to align scales
        # This prevents any single branch from dominating the Fusion early on
        v_a = self.norm_anatomical(v_a)
        v_l = self.norm_lexical(v_l)
        v_s = self.norm_statistical(v_s)
        
        # Step 3: Prepare for Cross-Modal Attention (batch, 3, 32)
        combined = torch.stack([v_a, v_l, v_s], dim=1)
        
        # Step 4: Cross-Modal Contextualization
        v_attended = self.cross_modal_transformer(combined)
        
        # Step 5: Flatten and Decision
        v_fusion = v_attended.view(v_attended.size(0), -1)
        risk_score = self.meta_learner(v_fusion)
        
        return risk_score