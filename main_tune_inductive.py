import torch
import pandas as pd
import os
import sys
from GNN_architectures.Hetero_GAT import HeteroGAT
from GNN_architectures.Hetero_GATv2 import HeteroGATv2
from GNN_architectures.Hetero_APPNP import HeteroAPPNP
from GNN_architectures.Hetero_GT import HGT
from GNN_architectures.Hetero_GraphSage import HeteroGraphSage
from GNN_architectures.Hetero_GIN import HeteroGIN
from utils_inductive import set_random_seed, read_input_data, prepare_splits, create_train_graph, create_test_graph, train_inductive, train_transductive, inference, create_splits_ecj


def train_predict(GNNalgo, algo, dim, budget, seed, whlf, n_heads, n_hid, n_layers, dropout, mode='inductive'):
    loc = '.'
    out_loc = f'../results_tune_{mode}'
    set_random_seed(seed)
    wpc = False # with_problem_class
    wad = True # with_algo_details
    normalize_flag = True
    n_epochs = 200
    learning_rate = 0.01
    emb_dim = 24
    weight_decay = 0.0001
    patience = 20
    factor = 0.5
    use_batch_norm = True
    use_layer_norm = True
    loss_fn_str = 'l1'
    perf_data_str = ''
    feat_drop = 0.05

    # prepare data
    problems_data, perf_data, algo_data = read_input_data(dim, budget, algo, loc, perf_data_str)

    all_predictions = []

    for outer_fold in range(1,6):
        train_mask, val_mask, test_mask, val_iid = create_splits_ecj(problems_data['label'], outer_fold)
        problems_train, problems_val, problems_test, perf_train, perf_val, perf_test, ela_train, ela_val, ela_test = prepare_splits(problems_data, perf_data, train_mask, val_mask, test_mask, normalize_flag)
        graph, y_true_train, node_dict, edge_dict, in_features = create_train_graph(problems_train, perf_train, algo_data, ela_train, emb_dim, wpc, wad, whlf)
        
        if mode == 'inductive':
            graph_val, y_true_val = create_test_graph(graph.clone(), perf_val, wpc, whlf, list(problems_val['group']), edge_dict, ela_val, emb_dim)  
            hl_feat = list(problems_test['group'])
            graph_test, y_true_test = create_test_graph(graph.clone(), perf_test, wpc, whlf, hl_feat, edge_dict, ela_test, emb_dim)    
        else:
            graph, y_true_val = create_test_graph(graph.clone(), perf_val, wpc, whlf, list(problems_val['group']), edge_dict, ela_val, emb_dim)  
            hl_feat = list(problems_test['group'])
            graph, y_true_test = create_test_graph(graph.clone(), perf_test, wpc, whlf, hl_feat, edge_dict, ela_test, emb_dim)   
        

        # create folders if they don't exist
        os.makedirs(f'{out_loc}/models/', exist_ok=True)
        os.makedirs( f'{out_loc}/performance/', exist_ok=True)
        os.makedirs( f'{out_loc}/predictions/', exist_ok=True)


        
        
        model_loc_save = f'{out_loc}/models/{perf_data_str}{loss_fn_str}_model_state_dict_{GNNalgo}_{algo}_{dim}_{budget}_{seed}_{whlf}_{n_heads}_{n_hid}_{n_layers}_{dropout}_BLN_{use_batch_norm or use_layer_norm}_{feat_drop}_{emb_dim}.pth'
        if GNNalgo == 'Hetero_GAT':
            model = HeteroGAT(n_layers, in_features, n_hid, 1, graph.etypes, n_heads, use_batch_norm, dropout)
            # model.reset_parameters()
        elif GNNalgo == 'Hetero_GraphSage':
            model = HeteroGraphSage(n_layers, in_features, n_hid, 1, graph.etypes, use_batch_norm, dropout, feat_drop)
            # model.reset_parameters()

        

        
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, total_steps=n_epochs, max_lr=learning_rate)
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=patience, factor=factor, verbose=False, min_lr=1e-5)
        if mode == 'inductive':
            # train
            trained_model, saved_epoch, saved_lr, l1_train, mse_train, l1_val, mse_val, r2_train, r2_val, training_time = train_inductive(graph, graph_val, model, opt, scheduler, n_epochs, y_true_train, y_true_val, model_loc_save, seed, loss_fn_str, return_predictions=False)
            #test   
            y_pred_test, l1_test, mse_test, r2_test, inference_time = inference(model, model_loc_save, graph_test, y_true_test, seed)
        else: 
            # train
            trained_model, saved_epoch, saved_lr, l1_train, mse_train, l1_val, mse_val, r2_train, r2_val, training_time = train_transductive(graph, model, opt, scheduler, n_epochs, y_true_train, y_true_val, model_loc_save, seed, loss_fn_str, return_predictions=False)
            #test   
            y_pred_test, l1_test, mse_test, r2_test, inference_time = inference(model, model_loc_save, graph, y_true_test, seed)

        
            

    
        # Store predictions for this fold
        all_predictions.append({
            "outer_fold": outer_fold,
            "val_iid": val_iid,
            "y_pred_test": y_pred_test.cpu().numpy(),  # Convert to numpy for easier saving
            "y_true_test": y_true_test.cpu().numpy(),  # Convert to numpy for easier saving
            "l1_test": l1_test,
            "mse_test": mse_test.item(),
            "r2_test": r2_test,
            "training_time": training_time,
            "inference_time": inference_time
        })


        # save results
        val_table = []
        val_table.append([outer_fold, val_iid, seed, saved_epoch, saved_lr, n_hid, n_heads, n_layers, patience, factor, dropout, feat_drop, emb_dim, l1_train.item(), mse_train.item(), r2_train, l1_val.item(), mse_val.item(), r2_val, l1_test.item(), mse_test.item(), r2_test])
        # print(f"\t\t\tModel saved at epoch {saved_epoch}, loss: {mse_val.item()}, r2: {r2_val}")
        val_columns=['outer_fold', 'val_iid', 'seed', 'saved_epoch', 'saved_lr', 'n_hid', 'n_heads', 'n_layers', 'patience', 'factor', 'dropout', 'feat_drop', 'emb_dim', 'l1_train', 'mse_train', 'r2_train', 'l1_val', 'mse_val', 'r2_val', 'l1_test', 'mse_test', 'r2_test']
        val_table_df = pd.DataFrame(val_table, columns=val_columns)
        if not os.path.exists(f'{out_loc}/performance/{perf_data_str}{loss_fn_str}_{GNNalgo}_val_test_table_{algo}_{dim}_{budget}_{mode}_whlf_{whlf}_seed_{seed}_BLN_{use_batch_norm or use_layer_norm}.csv'):
            val_table_df.to_csv(f'{out_loc}/performance/{perf_data_str}{loss_fn_str}_{GNNalgo}_val_test_table_{algo}_{dim}_{budget}_{mode}_whlf_{whlf}_seed_{seed}_BLN_{use_batch_norm or use_layer_norm}.csv', index=False, mode='w', header=True)
        else:
            val_table_df.to_csv(f'{out_loc}/performance/{perf_data_str}{loss_fn_str}_{GNNalgo}_val_test_table_{algo}_{dim}_{budget}_{mode}_whlf_{whlf}_seed_{seed}_BLN_{use_batch_norm or use_layer_norm}.csv', index=False, mode='a', header=False)
        
        print(f"\tevaluate hyperparameters on outer fold: {outer_fold}, seed: {seed}: n_heads = {n_heads}, hid = {n_hid}, n_layers = {n_layers}, dropout = {dropout} Train r2:  {r2_train}, Val r2: {r2_val}, Test r2: {r2_test}")

    # Save all predictions after all outer folds
    predictions_save_path = f'{out_loc}/predictions/{perf_data_str}{loss_fn_str}_{GNNalgo}_{algo}_{dim}_{budget}_{mode}_{whlf}_{seed}_BLN_{use_batch_norm or use_layer_norm}_{n_heads}_{n_hid}_{n_layers}_{dropout}_{feat_drop}_{emb_dim}.pkl'
    with open(predictions_save_path, 'wb') as f:
        torch.save(all_predictions, f)
    # print(f"Predictions saved to {predictions_save_path}")

if __name__ == "__main__":
    GNNalgo = sys.argv[1]
    algo = sys.argv[2]
    dim = int(sys.argv[3])
    budget = int(sys.argv[4])
    whlf = sys.argv[5].lower() 
    whlf = True if whlf == 'true' else False
    n_heads = int(sys.argv[6])
    n_hid = int(sys.argv[7])
    n_layers = int(sys.argv[8])
    dropout = float(sys.argv[9])
    seed = int(sys.argv[10])
    mode = sys.argv[11]

    
    train_predict(GNNalgo, algo, dim, budget, seed, whlf, n_heads, n_hid, n_layers, dropout, mode)
        