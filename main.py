import torch
import pandas as pd
import os
import sys
from GNN_architectures.Hetero_GraphSage import HeteroGraphSage
from utils import (
    set_random_seed, read_input_data, prepare_splits, create_train_graph,
    create_test_graph, train_inductive, inference, create_data_splits
)

def train_predict(algo, dim, budget, seed, n_hid, n_layers, dropout):
    """
    Train and evaluate a Heterogeneous GraphSAGE model for predicting performance metrics.
    
    Args:
        algo (str): Algorithm name.
        dim (int): Dimension of the problem space.
        budget (int): Computational budget of function evaluations.
        seed (int): Random seed for reproducibility.
        n_hid (int): Number of hidden units in each layer.
        n_layers (int): Number of layers in the GNN model.
        dropout (float): Dropout rate for regularization.
    """
    loc = '.'  # Data location
    out_loc = '../results'  # Output directory for storing models and results
    set_random_seed(seed)  # Ensure reproducibility
    
    
    # Hyperparameters
    n_epochs = 200
    learning_rate = 0.01
    emb_dim = 24
    weight_decay = 0.0001
    patience = 20  # Early stopping patience
    factor = 0.5  # Learning rate reduction factor

    # Read input data
    problems_data, perf_data, algo_data = read_input_data(dim, budget, algo, loc)
    all_predictions = []  # Store predictions across folds
    
    for outer_fold in range(1, 6):
        # Create train-validation-test splits
        train_mask, val_mask, test_mask, val_iid = create_data_splits(problems_data['label'], outer_fold)
        problems_train, problems_val, problems_test, perf_train, perf_val, perf_test, ela_train, ela_val, ela_test = prepare_splits(
            problems_data, perf_data, train_mask, val_mask, test_mask
        )
        
        # Create train graph
        graph, y_true_train, node_dict, edge_dict, in_features = create_train_graph(
            problems_train, perf_train, algo_data, ela_train, emb_dim
        )
        
        # Create validation and test graphs
        graph_val, y_true_val = create_test_graph(graph.clone(), perf_val, list(problems_val['group']), edge_dict, ela_val, emb_dim)
        graph_test, y_true_test = create_test_graph(graph.clone(), perf_test, list(problems_test['group']), edge_dict, ela_test, emb_dim)
        
        # Ensure output directories exist
        os.makedirs(f'{out_loc}/models/', exist_ok=True)
        os.makedirs(f'{out_loc}/performance/', exist_ok=True)
        os.makedirs(f'{out_loc}/predictions/', exist_ok=True)
        
        # Define model save path
        model_loc_save = f'{out_loc}/models/model_state_dict_{algo}_{dim}_{budget}_{seed}_{n_hid}_{n_layers}_{dropout}_{emb_dim}.pth'
        
        # Initialize model
        model = HeteroGraphSage(n_layers, in_features, n_hid, 1, graph.etypes, dropout)
        
        # Define optimizer and scheduler
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=patience, factor=factor, verbose=False, min_lr=1e-5)
        
        # Train model
        trained_model, saved_epoch, saved_lr, l1_train, mse_train, l1_val, mse_val, r2_train, r2_val, training_time = train_inductive(
            graph, graph_val, model, opt, scheduler, n_epochs, y_true_train, y_true_val, model_loc_save, seed, return_predictions=False
        )
        
        # Perform inference on test data
        y_pred_test, l1_test, mse_test, r2_test, inference_time = inference(model, model_loc_save, graph_test, y_true_test, seed)
        
        # Store predictions for this fold
        all_predictions.append({
            "outer_fold": outer_fold,
            "val_iid": val_iid,
            "y_pred_test": y_pred_test.cpu().numpy(),  # Convert tensors to numpy arrays
            "y_true_test": y_true_test.cpu().numpy(),
            "l1_test": l1_test,
            "mse_test": mse_test.item(),
            "r2_test": r2_test,
            "training_time": training_time,
            "inference_time": inference_time
        })
        
        # Save validation and test performance metrics
        val_table = [[outer_fold, val_iid, seed, saved_epoch, saved_lr, n_hid, n_layers, patience, factor, dropout, emb_dim,
                      l1_train.item(), mse_train.item(), r2_train, l1_val.item(), mse_val.item(), r2_val, l1_test.item(), mse_test.item(), r2_test]]
        val_columns = ['outer_fold', 'val_iid', 'seed', 'saved_epoch', 'saved_lr', 'n_hid', 'n_layers', 'patience', 'factor', 'dropout', 'emb_dim',
                       'l1_train', 'mse_train', 'r2_train', 'l1_val', 'mse_val', 'r2_val', 'l1_test', 'mse_test', 'r2_test']
        val_table_df = pd.DataFrame(val_table, columns=val_columns)
        val_file_path = f'{out_loc}/performance/val_test_table_{algo}_{dim}_{budget}_seed_{seed}.csv'
        
        # Save results to CSV
        if not os.path.exists(val_file_path):
            val_table_df.to_csv(val_file_path, index=False, mode='w', header=True)
        else:
            val_table_df.to_csv(val_file_path, index=False, mode='a', header=False)
        
        print(f"Evaluate hyperparameters on outer fold: {outer_fold}, seed: {seed}, hid = {n_hid}, n_layers = {n_layers}, dropout = {dropout} \n"
              f"Train R²: {r2_train}, Val R²: {r2_val}, Test R²: {r2_test}")
    
    # Save all predictions after all folds
    predictions_save_path = f'{out_loc}/predictions/{algo}_{dim}_{budget}_{seed}_{n_hid}_{n_layers}_{dropout}_{emb_dim}.pkl'
    with open(predictions_save_path, 'wb') as f:
        torch.save(all_predictions, f)
    
if __name__ == "__main__":

    algo = sys.argv[2]
    dim = int(sys.argv[3])
    budget = int(sys.argv[4])
    n_hid = int(sys.argv[7])
    n_layers = int(sys.argv[8])
    dropout = float(sys.argv[9])
    seed = int(sys.argv[10])
    
    train_predict(algo, dim, budget, seed, n_hid, n_layers, dropout)
