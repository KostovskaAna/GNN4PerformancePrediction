import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class HeteroGraphSage(nn.Module):
    def __init__(
        self, 
        n_layers, 
        n_inp_dict,  # Now it's a dictionary: {node_type: input_dim} 
        n_hid,
        n_out,
        rel_names,
        use_batch_norm=False,  # New parameter to toggle batch normalization
        dropout = 0.2,
        feat_drop = 0.05
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm  # Store the flag for batch norm
        self.layers = nn.ModuleList()
        self.adapt_ws = nn.ModuleDict()  # Changed to ModuleDict
        self.bn_adapt = nn.ModuleDict() if use_batch_norm else None
        self.bn_layers = nn.ModuleList() if use_batch_norm else None


        for ntype, inp_dim in n_inp_dict.items():  # Loop over each node type and its input dimension
            self.adapt_ws[ntype] = nn.Linear(inp_dim, n_hid)  # Create a Linear layer for this node type
            if use_batch_norm:
                self.bn_adapt[ntype] = nn.BatchNorm1d(n_hid)

        for _ in range(n_layers):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    { rel: dglnn.SAGEConv(n_hid, n_hid, 'mean', feat_drop=feat_drop) for rel in rel_names}, aggregate='sum')
                    # { rel: dglnn.SAGEConv(n_hid, n_hid, 'mean') for rel in rel_names}, aggregate='sum')
            )
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(n_hid, track_running_stats=False))  # Batch normalization for layer outputs
            
        # self.penultimate = nn.Linear(n_hid, n_hid) 
        self.out = nn.Linear(n_hid, n_out)
        

    def forward(self, graph, feat, out_key='performance', eweight=None):
        # inputs are features of nodes
        # h = self.conv1(graph, feat)
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = {k: self.dropout(v) for k, v in h.items()}
 
        with graph.local_scope():
            h = {}
            for ntype in graph.ntypes:
                h[ntype] = self.adapt_ws[ntype](feat[ntype])
                if self.use_batch_norm and h[ntype].size(0):
                    h[ntype] = self.bn_adapt[ntype](h[ntype])
                h[ntype] = F.gelu(h[ntype])

            for i in range(self.n_layers):
                if eweight:
                    # Pass edge weights to each layer if available
                    h = self.layers[i](graph, h, mod_kwargs={rel: {'edge_weight': eweight[rel]} for rel in eweight})
                else:
                    h = self.layers[i](graph, h)
                # if self.use_batch_norm and h.size(0) > 1:
                #     h = {k: v.flatten(1) for k, v in h.items()}
                #     h = {k: self.bn_layers[i](v) for k, v in h.items()}
                if self.use_batch_norm:
                    h = {k: v.flatten(1) for k, v in h.items()}
                    h = {k: self.bn_layers[i](v) if v.size(0) > 1 else v for k, v in h.items()}
                
                h = {k: F.gelu(v) for k, v in h.items()}
                h = {k: self.dropout(v) for k, v in h.items()}
            # Use the penultimate layer with a sigmoid activation
            # h = {k: F.gelu(self.penultimate(v)) for k, v in h.items()}
            # h = {k: self.dropout(v) for k, v in h.items()}
        
            if out_key == 'performance':
                return self.out(h['performance'])
            else:
                return h


    def predict(self, embeddings, out_key='performance'):
        # Predict the final output using the embeddings from the forward pass
        return self.out(embeddings[out_key])
        # output = F.relu(self.out(embeddings[out_key])) - 8
        # return output