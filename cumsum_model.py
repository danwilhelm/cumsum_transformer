import numpy as np

class CumsumModel:

    # avoids numeric precision errors
    # e.g. see https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
    @staticmethod
    def log_softmax(x):
        c = x.max()
        logsumexp = np.log(np.exp(x - c).sum(axis=-1, keepdims=True))
        return x - c - logsumexp

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True))


    def __init__(self, state_dict=None, toks_in=None):
        if state_dict is None:
            self.load_minimal_transformer()
        else:
            self.load_state_dict(state_dict)

        if toks_in is not None:
            self.run(toks_in)


    def load_state_dict(self, state_dict):
        self.seq_len = 20; self.max_value = 5
        self.n_layers = 1; self.n_heads = 1
        self.d_embd = 24; self.d_head = 12; self.d_mlp = 8
        self.mlp = True
        self.embed_labels = list(range(self.max_value+1)) + list(range(-self.max_value,0))
        self.n_vocab = len(self.embed_labels)

        self.embeds = state_dict['embed.W_E'].cpu().numpy()           # (11, 24)  <= order: [0, +1, +2, +3, +4, +5, -5, -4, -3, -2, -1]  very literal, direct indexing
        self.pos_embeds = state_dict['pos_embed.W_pos'].cpu().numpy() # (20, 24)  <= max seq len of 20
        self.attn_wq = state_dict['blocks.0.attn.W_Q'].cpu().numpy()[0]; self.attn_bq = state_dict['blocks.0.attn.b_Q'].cpu().numpy()[0]  # (24, 12)    (12,)
        self.attn_wk = state_dict['blocks.0.attn.W_K'].cpu().numpy()[0]; self.attn_bk = state_dict['blocks.0.attn.b_K'].cpu().numpy()[0]  # (24, 12)    (12,)
        self.attn_wv = state_dict['blocks.0.attn.W_V'].cpu().numpy()[0]; self.attn_bv = state_dict['blocks.0.attn.b_V'].cpu().numpy()[0]  # (24, 12)    (12,)
        self.attn_wo = state_dict['blocks.0.attn.W_O'].cpu().numpy()[0]; self.attn_bo = state_dict['blocks.0.attn.b_O'].cpu().numpy()     # (12, 24)    (24,)

        self.attn_mask = state_dict['blocks.0.attn.mask'].cpu().numpy()     # (20, 20)    [[True, False, ...], [True, True, False, ...], ..., [True, ...]]
        self.attn_ignore = state_dict['blocks.0.attn.IGNORE'].cpu().numpy() # [-100000.]

        self.mlp_win = state_dict['blocks.0.mlp.W_in'].cpu().numpy(); self.mlp_bin = state_dict['blocks.0.mlp.b_in'].cpu().numpy()      # (24, 8)   (8,)
        self.mlp_wout = state_dict['blocks.0.mlp.W_out'].cpu().numpy(); self.mlp_bout = state_dict['blocks.0.mlp.b_out'].cpu().numpy()  # (8, 24)   (24,)

        self.unembed_w = state_dict['unembed.W_U'].cpu().numpy(); self.unembed_b = state_dict['unembed.b_U'].cpu().numpy()              # (24, 3)   (3,)

        # MHA equivalent coefs
        self.mha_coefs = (self.attn_wv @ self.attn_wo).T                # rows=output pos (as a func of input pos)  (24,24) 
        self.mha_biases = self.attn_bv @ self.attn_wo + self.attn_bo    # (24,)

        # FFN equivalent coefs (assuming no RELU)
        self.ffn_coefs = (self.mlp_win @ self.mlp_wout).T               # rows=output pos  (24,24)
        self.ffn_biases = self.mlp_bin @ self.mlp_wout + self.mlp_bout  # (24,)


    def load_minimal_transformer(self):
        attn_value = 9.5367431640625e-7         # exactly-stored 1/2^20
        softmax_ignore_value = -100000.

        self.seq_len = 20; self.max_value = 5
        self.n_layers = 1; self.n_heads = 1
        self.d_embd = 2; self.d_head = 1; self.d_mlp = 0
        self.mlp = False; self.n_labels = 3
        self.embed_labels = np.array(list(range(self.max_value+1)) + list(range(-self.max_value,0)))
        self.n_vocab = len(self.embed_labels)

        # order: [0, +1, +2, +3, +4, +5, -5, -4, -3, -2, -1]  very literal, direct indexing
        self.embeds = np.array([self.embed_labels, -self.embed_labels], dtype=np.float64).T / 100.             # (11,2)
        self.pos_embeds = np.zeros((self.seq_len, self.d_embd))                                                # (20,2)
        self.attn_wq = np.full((self.d_embd,self.d_head), attn_value); self.attn_bq = np.zeros(self.d_head)    # (2,1)  (1,)
        self.attn_wk = np.full((self.d_embd,self.d_head), attn_value); self.attn_bk = np.zeros(self.d_head)    # (2,1)  (1,)
        self.attn_wv = np.array([[1.], [0.]]); self.attn_bv = np.zeros(self.d_head)                            # (2,1)  (1,)
        self.attn_wo = np.array([[250., -250.]]); self.attn_bo = np.array([0.3, 0.3])                          # (1,2)  (2,)
        
        self.attn_mask = np.tril(np.full((self.seq_len, self.seq_len), True), k=0)
        self.attn_ignore = softmax_ignore_value

        self.unembed_w = np.array([[-10.0, 2.0, 10.0], [10.0, 2.0, -10.]]); self.unembed_b = np.array([0.2])   # (2,3) (1,)
        
        # MHA equivalent coefs
        self.mha_coefs = (self.attn_wv @ self.attn_wo).T                # rows=output pos (as a func of input pos)  (24,24) 
        self.mha_biases = self.attn_bv @ self.attn_wo + self.attn_bo    # (24,)


    def run(self, toks_in):
        self.toks_in = np.array(toks_in) - self.max_value
        self.n_toks_in = len(self.toks_in)
        self.embeds_in = self.embeds[self.toks_in]
        self.pos_embeds_in = self.pos_embeds[:self.n_toks_in]
        self.stream_in = self.embeds_in + self.pos_embeds_in

        # ATTENTION
        self.attn_in = self.stream_in                               # (toks,24)
        self.attn_q = self.attn_in @ self.attn_wq + self.attn_bq    # (toks,24) @ (24,12) => (toks,12)
        self.attn_k = self.attn_in @ self.attn_wk + self.attn_bk    # (toks,24) @ (24,12) => (toks,12)
        self.attn_v = self.attn_in @ self.attn_wv + self.attn_bv    # (toks,24) @ (24,12) => (toks,12)

        # [(toks,12) @ (12,toks) => (toks,toks) + (toks,toks)] @ (toks,12) => (toks,12)
        self.attn_scores = np.where(self.attn_mask[:self.n_toks_in, :self.n_toks_in],
                                    self.attn_q @ self.attn_k.T / np.sqrt(self.d_head),
                                    self.attn_ignore)
        self.attn_pattern = self.softmax(self.attn_scores)
        self.attn = self.attn_pattern @ self.attn_v

        self.attn_out = self.attn @ self.attn_wo + self.attn_bo              # (toks,12) @ (12,24) + (,12) => (toks,12)

        # MLP
        self.mlp_in = self.stream_in + self.attn_out
        self.mlp_out = 0.0
        if self.mlp:
            self.hidden = self.mlp_in @ self.mlp_win + self.mlp_bin          # (toks,24) @ (24,8) + (8,) => (toks,8)
            self.hidden_out = np.maximum(self.hidden, 0)                     # relu => (toks,8)
            self.mlp_out = self.hidden_out @ self.mlp_wout + self.mlp_bout   # (toks,8) @ (8,24) + (24,) => (toks,24)            

        # UNEMBED
        self.stream_out = self.mlp_in + self.mlp_out
        self.logits = self.stream_out @ self.unembed_w + self.unembed_b      # (toks,24) @ (24,3) + (3,) => (3,)
        self.logprobs = self.log_softmax(self.logits)
        self.probs = self.softmax(self.logprobs)
        self.labels = np.argmax(self.probs, axis=1)
        
        self.probs_correct = np.max(self.probs, axis=1)         # np.choose(self.labels, self.probs.T)
        self.logprobs_correct = np.max(self.logprobs, axis=1)   # np.choose(self.labels, self.logprobs.T)

        return self.probs
