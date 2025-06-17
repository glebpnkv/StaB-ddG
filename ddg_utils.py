import torch
from torch import nn
from training.model_utils import featurize
from dataclasses import dataclass

# TODO: switch model config to include this.
@dataclass
class ProtddGConfig:
    pmpnn: nn.Module
    fix_perm: bool = False
    fix_noise: bool = False
    decode_mut_last: bool = False
    device: str = 'cuda'

class ProtddG(nn.Module):
    def __init__(self, pmpnn, scale_binder=False, fix_perm=False, fix_noise=False, device='cuda'):
        super(ProtddG, self).__init__()
        self.pmpnn = pmpnn
        self.fix_perm = fix_perm
        self.fix_noise = fix_noise
        self.device = device
        self.use_beta = scale_binder

        self.beta = nn.Parameter(torch.tensor(1.0))

    def get_decoding_order(self, chain_M):
        """ Generate a random decoding order with the same shape as chain_M. """
        return torch.argsort(torch.abs(torch.randn(chain_M.shape, device=self.device)))
    
    def get_backbone_noise(self, X, noise_level=0.2):
        """ Generate random backbone noise. Defaults to 0.2A. """
        return noise_level * torch.randn_like(X, device=self.device)

    def get_wt_seq(self, domain):
        """ Returns the wild type sequence of a protein. """
        _, wt_seq, *_ = featurize([domain], self.device)
        return wt_seq
    
    def folding_dG(self, domain, seqs, decoding_order=None, backbone_noise=None):
        """ Predicts the folding stability (dG) for a list of sequences. """
        B = seqs.shape[0]

        X, _, mask, _, chain_M, residue_idx, _, chain_encoding_all = featurize([domain], self.device)
        X, S, mask = X.repeat(B, 1, 1, 1), seqs.to(self.device), mask.repeat(B, 1)
        chain_M = chain_M.repeat(B, 1)
        residue_idx, chain_encoding_all = residue_idx.repeat(B, 1), chain_encoding_all.repeat(B, 1)
        
        order = decoding_order.repeat(B, 1) if self.fix_perm else None
        backbone_noise = backbone_noise.repeat(B, 1, 1, 1) if self.fix_noise else None

        log_probs = self.pmpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, 
                              fix_order=order, fix_backbone_noise=backbone_noise)

        seq_oh = torch.nn.functional.one_hot(seqs, 21).to(self.device)
        dG = torch.sum((seq_oh) * log_probs, dim=(1, 2))

        return dG

    def folding_ddG(self, domain, mut_seqs, set_wt_seq=None):
        """ Predicts the folding ddG. """
        X, wt_seq, _, _, chain_M, _, _, _ = featurize([domain], self.device)

        if not set_wt_seq is None:
            wt_seq = set_wt_seq

        decoding_order = self.get_decoding_order(chain_M) if self.fix_perm else None
        backbone_noise = self.get_backbone_noise(X) if self.fix_noise else None

        wt_dG = self.folding_dG(domain, wt_seq, decoding_order=decoding_order, backbone_noise=backbone_noise)
        mut_dG = self.folding_dG(domain, mut_seqs, decoding_order=decoding_order, backbone_noise=backbone_noise)

        ddG = mut_dG - wt_dG

        return ddG
    
    def binding_ddG(self, complex, binder1, binder2, complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs):
        """ We calculate the binding ddG by decomposing it into three folding ddG terms, 
            corresponding to the entire complex and each individual binders. 
        """
        complex_ddG_fold = self.folding_ddG(complex, complex_mut_seqs)
        binder1_ddG_fold = self.folding_ddG(binder1, binder1_mut_seqs)
        binder2_ddG_fold = self.folding_ddG(binder2, binder2_mut_seqs)

        beta = self.beta if self.use_beta else 1.0
        
        ddG = complex_ddG_fold - (binder1_ddG_fold - binder2_ddG_fold) * beta
  
        return ddG

    def forward(self, complex, binder1, binder2, complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs):
        return self.binding_ddG(complex, binder1, binder2, complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs)

class LinearModel(nn.Module):
    def __init__(self, num_features):
        super(LinearModel, self).__init__()
        # Linear layer with num_features inputs and 1 output (with bias term)
        self.linear = nn.Linear(num_features, 1, bias=True)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def ddG_pred(model, domain, mut_seqs, fix_perm=False, fix_noise=False, decode_mut_last=False, batch_size=10000, wt_seq=None, device='cuda'):
    X, S, mask, _, chain_M, residue_idx, _, chain_encoding_all = featurize([domain], device)

    if not wt_seq is None:
        # The wildtype sequence of interest is different from the one in domain
        S = wt_seq
    
    mut_seq_list = torch.nn.functional.one_hot(mut_seqs, 21).to(device)
    wt_one_hot = torch.nn.functional.one_hot(S, 21)

    # fix a random decoding order for all sequences
    order = torch.argsort(torch.abs(torch.randn(chain_M.shape, device=device))) if fix_perm else None
    
    backbone_noise = 0.2 * torch.randn_like(X) if fix_noise else None
    wt_log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, fix_order=order, fix_backbone_noise=backbone_noise)
    wt_likelihood = torch.sum(wt_one_hot * wt_log_probs)

    N = mut_seqs.shape[0] # total number of mutants
    M = batch_size // S.shape[1] # number of mutants per batch
    pred = []
    
    for batch_idx in range(0, N, M):
        B = min(N - batch_idx, M)
        X_, S_, mask_ = X.repeat(B, 1, 1, 1), mut_seqs[batch_idx:batch_idx+B].to(device), mask.repeat(B, 1)
        chain_M_ = chain_M.repeat(B, 1)
        residue_idx_, chain_encoding_all_ = residue_idx.repeat(B, 1), chain_encoding_all.repeat(B, 1)
        order_ = None
        if decode_mut_last:
            chain_M_ = (S_ != mut_seqs[batch_idx:batch_idx+B].to(device)).float()
            order_ = torch.argsort((chain_M_+0.0001)*(torch.abs(torch.randn(chain_M_.shape, device=device))))
            wt_log_probs = model(X_, S.repeat(B, 1), mask_, chain_M_, residue_idx_, chain_encoding_all_, fix_order=order_,
                          fix_backbone_noise=backbone_noise.repeat(B, 1, 1, 1) if fix_noise else None)
            wt_likelihood = torch.sum(wt_one_hot * wt_log_probs, dim=(1, 2))
        else:
            chain_M_ = chain_M.repeat(B, 1)
            if fix_perm:
                order_ = order.repeat(B, 1)
        assert (order_ is None) or (order_.shape == chain_M_.shape)
        log_probs = model(X_, S_, mask_, chain_M_, residue_idx_, chain_encoding_all_, fix_order=order_,
                          fix_backbone_noise=backbone_noise.repeat(B, 1, 1, 1) if fix_noise else None)
        pred.append(torch.sum((mut_seq_list[batch_idx:batch_idx+B, :, :]) * log_probs, dim=(1, 2)) - wt_likelihood)

    pred = torch.cat(pred)
    return pred, log_probs, S

def dG_pred(model, domain, mut_seqs, fix_perm=False, fix_noise=False, decode_mut_last=False, wt_seq=None, batch_size=10000, device='cuda'):
    X, S, mask, _, chain_M, residue_idx, _, chain_encoding_all = featurize([domain], device)

    if not wt_seq is None:
        # The wildtype sequence of interest is different from the one in domain
        S = wt_seq
    
    mut_seq_list = torch.nn.functional.one_hot(mut_seqs, 21).to(device)

    # fix a random decoding order for all sequences
    order = torch.argsort(torch.abs(torch.randn(chain_M.shape, device=device))) if fix_perm else None
    
    backbone_noise = 0.2 * torch.randn_like(X) if fix_noise else None

    N = mut_seqs.shape[0] # total number of mutants
    M = batch_size // S.shape[1] # number of mutants per batch
    pred = []
    
    for batch_idx in range(0, N, M):
        B = min(N - batch_idx, M)
        X_, S_, mask_ = X.repeat(B, 1, 1, 1), mut_seqs[batch_idx:batch_idx+B].to(device), mask.repeat(B, 1)
        chain_M_ = chain_M.repeat(B, 1)
        residue_idx_, chain_encoding_all_ = residue_idx.repeat(B, 1), chain_encoding_all.repeat(B, 1)
        order_ = None
        if decode_mut_last:
            chain_M_ = (S_ != mut_seqs[batch_idx:batch_idx+B].to(device)).float()
            order_ = torch.argsort((chain_M_+0.0001)*(torch.abs(torch.randn(chain_M_.shape, device=device))))
        else:
            chain_M_ = chain_M.repeat(B, 1)
            if fix_perm:
                order_ = order.repeat(B, 1)
        assert (order_ is None) or (order_.shape == chain_M_.shape)
        log_probs = model(X_, S_, mask_, chain_M_, residue_idx_, chain_encoding_all_, fix_order=order_,
                          fix_backbone_noise=backbone_noise.repeat(B, 1, 1, 1) if fix_noise else None)
        pred.append(torch.sum((mut_seq_list[batch_idx:batch_idx+B, :, :]) * log_probs, dim=(1, 2)))

    pred = torch.cat(pred)
    return pred, log_probs, S