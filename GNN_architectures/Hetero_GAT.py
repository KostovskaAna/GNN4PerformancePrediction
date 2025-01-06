import dgl.data
import dgl.nn as dglnn
import torch.nn as nn
import torch
import torch.nn.functional as F

class HeteroGAT(nn.Module):
    def __init__(
        self, 
        n_layers, 
        n_inp_dict,  # Dictionary: {node_type: input_dim} 
        n_hid,
        n_out,
        rel_names,
        num_heads=4,  # Number of attention heads
        use_batch_norm=False,  # New parameter to toggle batch normalization
        dropout = 0.2
    ):
        super().__init__()
        self.n_layers = n_layers
        self.use_batch_norm = use_batch_norm  # Store the flag for batch norm
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.adapt_ws = nn.ModuleDict()
        self.bn_adapt = nn.ModuleDict() if use_batch_norm else None
        self.bn_layers = nn.ModuleList() if use_batch_norm else None

        for ntype, inp_dim in n_inp_dict.items():
            self.adapt_ws[ntype] = nn.Linear(inp_dim, n_hid)
            if use_batch_norm:
                self.bn_adapt[ntype] = nn.BatchNorm1d(n_hid)

        # Initialize the first layer
        self.layers.append(
            dglnn.HeteroGraphConv(
                {rel: dglnn.GATConv(n_hid, 
                                    n_hid, 
                                    num_heads=num_heads, 
                                    feat_drop=0.05,
                                    attn_drop=dropout,
                                    activation=None) 
                    for rel in rel_names
                },
                aggregate='sum')
        )
        if use_batch_norm:
            self.bn_layers.append(nn.BatchNorm1d(n_hid * num_heads))

        # Initialize the rest of the layers
        for _ in range(n_layers-1):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {rel: dglnn.GATConv(n_hid*num_heads,
                                        n_hid, 
                                        num_heads=num_heads, 
                                        feat_drop=0.05,
                                        attn_drop=dropout,
                                        residual=True) 
                        for rel in rel_names
                    },
                    aggregate='sum')
            )
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(n_hid * num_heads, track_running_stats=False))
        
        # Add two MLP layers before the final output layer
        self.mlp_layers = nn.Sequential(
            nn.Linear(n_hid * num_heads, n_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.out = nn.Linear(n_hid, n_out)  # Final output layer

    def forward(self, graph, feat, out_key='performance', eweight=None):
        h = {}
        for ntype in graph.ntypes:
            h[ntype] = self.adapt_ws[ntype](feat[ntype])
            if self.use_batch_norm:
                h[ntype] = self.bn_adapt[ntype](h[ntype])
            h[ntype] = F.gelu(h[ntype])
        
        for i in range(self.n_layers):
            h = self.layers[i](graph, h)
            h = {k: v.flatten(1) for k, v in h.items()}
            if self.use_batch_norm:
                h = {k: self.bn_layers[i](v) for k, v in h.items()}
            h = {k: F.gelu(v) for k, v in h.items()}
            h = {k: self.dropout(v) for k, v in h.items()}
        
        # Apply the MLP layers before the final output
        h = {k: self.mlp_layers(v) for k, v in h.items()}

        if out_key == 'performance':
            return self.out(h['performance'])
        else:
            return h

    def predict(self, embeddings, out_key='performance'):
        return self.out(embeddings[out_key])
    

    def predict_no_perf(self, embeddings):
        # Extract embeddings for problem and algo nodes
        problem_embeddings = embeddings['problem']  # Shape: (72, 128)
        algo_embeddings = embeddings['algo']        # Shape: (324, 128)
        
        # Create the Cartesian product
        algo_embeddings_expanded = algo_embeddings.unsqueeze(1).repeat(1, problem_embeddings.shape[0], 1)  # Shape: (324, 72, 128)
        problem_embeddings_expanded = problem_embeddings.unsqueeze(0).repeat(algo_embeddings.shape[0], 1, 1)  # Shape: (324, 72, 128)
        
        # Concatenate along the last dimension
        concatenated_embeddings = torch.cat((algo_embeddings_expanded, problem_embeddings_expanded), dim=2)  # Shape: (324, 72, 256)
        
        # Reshape to a 2D tensor where each row is an algo-problem pair
        concatenated_embeddings = concatenated_embeddings.view(-1, concatenated_embeddings.shape[-1])  # Shape: (324*72, 256)
        
        # Pass the concatenated embeddings through the MLP layers
        concatenated_embeddings = self.mlp_layers(concatenated_embeddings)
        
        # Pass the embeddings through the output layer
        output = self.out(concatenated_embeddings)  # Shape: (324*72, output_dim)
        
        return output
