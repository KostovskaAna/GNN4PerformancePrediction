import ast
import dgl
import torch
import time
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.nn import L1Loss
from torch.nn import MSELoss

def get_encoded_problem_instances(list, offset=0):
    problem_dict = {}
    encoded_problems = []
    for problem in list:
        if problem not in problem_dict:
            problem_dict[problem] = len(problem_dict)+offset
        encoded_problems.append(problem_dict[problem])
    return encoded_problems, problem_dict

def get_problem_instances(dict):
    return [dict[key] for key in dict]

def get_problem_classes(dict):
    return [int(key.split('_')[0])-1 for key in dict]

def get_algo_parameters(algo_nodes_data, n_diff_options_per_module):
    algo_parameter_dst_array = []
    algo_feat = algo_nodes_data['feat'].apply(lambda x: ast.literal_eval(x))
    for algo_row_feat in algo_feat:
        for i, el in enumerate(algo_row_feat):
            offset = sum([n_diff_options_per_module[j] for j in range(i)])
            algo_parameter_dst_array.append(offset + el-1)
    return algo_parameter_dst_array

def create_in_features_node_dict(hetero_graph):
    in_features_dict_node = {}
    for node in hetero_graph.ntypes:
        in_features_dict_node[node] = hetero_graph.nodes[node].data['feat'].shape[1]
    return in_features_dict_node

def create_graph_edges_train(performance_edges_data_train, algo_nodes_data, n_algos, n_parameter_per_conf, add_reverse_edges=True):
    n_performance_records = performance_edges_data_train.shape[0]
    algo_performance_src_array = performance_edges_data_train['src_id'].to_numpy()
    algo_performance_dst_array = list(range(n_performance_records))
    performance_problem_src_array = list(range(n_performance_records))
    performance_problem_dst_array, problem_dict = get_encoded_problem_instances(performance_edges_data_train['dst_id'].to_numpy())

    
    graph_data = {
        ('algo', 'algo-performance', 'performance'): (algo_performance_src_array, algo_performance_dst_array),
        ('performance', 'performance-problem', 'problem'): (performance_problem_src_array, performance_problem_dst_array)
        
    }    
    if add_reverse_edges:
        graph_data[('performance', 'algo-performance-reverse', 'algo')]= (algo_performance_dst_array, algo_performance_src_array)
        graph_data[('problem', 'problem-performance', 'performance')]= (performance_problem_dst_array, performance_problem_src_array)

    
       
    algo_parameter_src_array = [algo_id for algo_id in range(n_algos) for i in range(n_parameter_per_conf)]
    n_diff_options_per_module = [2,3,3,3,3,2] if (n_algos == 324) else [3, 4, 2, 2, 2, 3, 2] 
    algo_parameter_dst_array = get_algo_parameters(algo_nodes_data, n_diff_options_per_module)
    parameter_parameter_class_src_array = [i for i in range(sum(n_diff_options_per_module))]
    parameter_parameter_class_dsc_array = [i for i, n_option_per_class in enumerate(n_diff_options_per_module) for _ in range(n_option_per_class)]
    graph_data[('algo', 'has-algo-parameter', 'parameter')] = (algo_parameter_src_array, algo_parameter_dst_array)
    graph_data[('parameter', 'is-parameter-class', 'parameter-class')] = (parameter_parameter_class_src_array, parameter_parameter_class_dsc_array)
    if add_reverse_edges:
        graph_data[('parameter', 'has-algo-parameter-reverse', 'algo')] = (algo_parameter_dst_array, algo_parameter_src_array)
        graph_data[('parameter-class', 'is-parameter-class-reverse', 'parameter')] = (parameter_parameter_class_dsc_array, parameter_parameter_class_src_array)
    if (n_algos == 324): 
        # modCMA
        # parameter classes -> 0: elitist, 1: mirrored, 2: base_sampler, 3: weights_option, 4: local_restart, 5: step_size_adaptation
        # elitism -> modCMA-ES_selection (2)                      0-2
        # mirrored -> modCMA-ES_mutation (1)                      1-1
        # base_sampler -> modCMA-ES_mutation (1)                  2-1
        # weights_option -> modCMA-ES_recombination (3)           3-3
        # local_restart -> modCMA-ES_initialization (0)           4-0
        # step_size_adaptation -> modCMA-ES_parameter_update (4)  5-4
        graph_data[('parameter-class', 'is-class-parameter-of', 'algo-execution-part')] = ([0,1,2,3,4,5], [2,1,1,3,0,4])
        # graph_data[('algo-execution-part', 'is-part-of', 'algo-execution')] = ([0,1,2,3,4], [0,0,0,0,0])
        if add_reverse_edges:
                graph_data[('algo-execution-part', 'is-class-parameter-of-reverse', 'parameter-class')] = ([2,1,1,3,0,4], [0,1,2,3,4,5])
            #  graph_data[('algo-execution', 'has-part', 'algo-execution-part')] = ([0,0,0,0,0], [0,1,2,3,4])
    else:
        # modDE
        # parameter classes -> 0: mutation_base, 1: mutation_reference, 2: mutation_n_comps, 3: use_archive, 4: crossover, 5: adaptation_method, 6: lpsr
        # mutation_base -> modDE_mutation (1)                     0-1  
        # mutation_reference -> modDE_mutation (1)                1-1
        # mutation_n_comps -> modDE_mutation (1)                  2-1
        # use_archive -> modDE_mutation (1)                       3-1
        # crossover -> modDE_recombination (2)                    4-2
        # adaptation_method -> modDE_parameter_update (3)         5-3
        # lpsr -> modDE_initialization (0)                        6-0
        graph_data[('parameter-class', 'is-class-parameter-of', 'algo-execution-part')] = ([0,1,2,3,4,5,6], [1,1,1,1,2,3,0])
        # graph_data[('algo-execution-part', 'is-part-of', 'algo-execution')] = ([0,1,2,3], [0,0,0,0])
        if add_reverse_edges:
            graph_data[('algo-execution-part', 'is-class-parameter-of-reverse', 'parameter-class')] = ([1,1,1,1,2,3,0], [0,1,2,3,4,5,6])
            # graph_data[('algo-execution', 'has-part', 'algo-execution-part')] = ([0,0,0,0], [0,1,2,3])

            
    hetero_graph= dgl.heterograph(graph_data)
    return hetero_graph

def create_train_graph(problem_nodes_data, performance_edges_data, algo_nodes_data, ela_df, embedding_dim):
    n_algos = algo_nodes_data.shape[0]
    total_params = 16 if n_algos == 324 else 18 
    n_exicution_parts = 5 if n_algos == 324 else 4
    algo_feat = algo_nodes_data['feat'].apply(lambda x: ast.literal_eval(x))
    n_parameter_per_conf = len(algo_feat[0])
    ground_truth = torch.tensor(performance_edges_data['score'].tolist()).to(torch.float32).unsqueeze(1)
    hetero_graph = create_graph_edges_train(performance_edges_data, algo_nodes_data, n_algos, n_parameter_per_conf)
    

    ela_instance_tensor = torch.tensor(ela_df.values).to(torch.float32)
    hetero_graph.nodes['problem'].data['feat'] = nn.Parameter(ela_instance_tensor, requires_grad=False)
    
    hetero_graph.nodes['performance'].data['feat'] = nn.init.kaiming_uniform_(nn.Parameter(torch.randn(ground_truth.shape[0], embedding_dim), requires_grad=False))


    hetero_graph.nodes['parameter-class'].data['feat'] = nn.init.kaiming_uniform_(nn.Parameter(torch.randn(n_parameter_per_conf, embedding_dim), requires_grad=False))
    hetero_graph.nodes['algo-execution-part'].data['feat'] = nn.init.kaiming_uniform_(nn.Parameter(torch.randn(n_exicution_parts, embedding_dim), requires_grad=False))
    hetero_graph.nodes['algo'].data['feat'] = nn.init.kaiming_uniform_(nn.Parameter(torch.randn(n_algos, embedding_dim), requires_grad=False))
    hetero_graph.nodes['parameter'].data['feat'] = nn.init.kaiming_uniform_(nn.Parameter(torch.randn(total_params, embedding_dim), requires_grad=False))

   
    node_dict = {}
    edge_dict = {}
    for ntype in hetero_graph.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hetero_graph.etypes:
        edge_dict[etype] = len(edge_dict)
        hetero_graph.edges[etype].data["id"] = (torch.ones(hetero_graph.num_edges(etype), dtype=torch.long) * edge_dict[etype])

    in_features_dict_node = create_in_features_node_dict(hetero_graph)

    return hetero_graph, ground_truth, node_dict, edge_dict, in_features_dict_node

def train_inductive(hetero_graph, hetero_graph_val, model, optimizer, scheduler, n_epochs, ground_truth_train, ground_truth_val, model_loc_save, seed, return_predictions= False, clip_grad_norm=0.5):
    set_random_seed(seed)
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_train_r2 = -1
    best_val_r2 = -1
    best_train_mse=-1
    best_val_mse=-1
    best_train_l1=-1
    best_val_l1=-1
    best_epoch = -1
    best_pred = None
    save_lr=0

    start_time = time.time()  # Start timer
    for epoch in range(n_epochs):
        model.train()
        # forward propagation by using all nodes and extracting the performance predictions
        embeddings = model(hetero_graph, hetero_graph.ndata['feat'], out_key='none')
        predictions = model.predict(embeddings)
        
        loss_fn = L1Loss()
        mse_fn = MSELoss()
        l1_fn = L1Loss()
        
        train_loss = loss_fn(predictions, ground_truth_train) 
        train_r2_score = r2_score(ground_truth_train.detach().numpy(), predictions.detach().numpy())
        train_mse = mse_fn(predictions, ground_truth_train)
        train_l1 = l1_fn(predictions, ground_truth_train)

        model.eval()
        with torch.no_grad():
            embeddings_val = model(hetero_graph_val, hetero_graph_val.ndata['feat'], out_key='none')
            predictions_val = model.predict(embeddings_val)

        val_mask = [False if i < predictions.shape[0] else True for i in range(predictions_val.shape[0])]
        predictions_val = predictions_val[val_mask]
        model.train()
        

        val_loss = loss_fn(predictions_val, ground_truth_val) 
        val_r2_score = r2_score(ground_truth_val.detach().numpy(), predictions_val.detach().numpy())
        val_mse = mse_fn(predictions_val, ground_truth_val)
        val_l1 = l1_fn(predictions_val, ground_truth_val)
        optimizer.zero_grad()
        train_loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        scheduler.step(val_loss)
        print(f"Epoch {epoch}, Train R2 Score: {train_r2_score}, Val R2 Score: {val_r2_score}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_loc_save)
            best_pred = predictions_val
            best_train_r2 = train_r2_score
            best_val_r2 = val_r2_score
            best_train_l1 = train_l1
            best_val_l1 = val_l1
            best_train_mse = train_mse
            best_val_mse = val_mse
            training_time = (time.time() - start_time) * 1000 
            save_lr = optimizer.param_groups[0]['lr']

    if return_predictions:
        return model, best_epoch, save_lr, best_train_l1, best_train_mse, best_val_l1, best_val_mse, best_train_r2, best_val_r2, training_time, best_pred
    else:
        return model, best_epoch, save_lr, best_train_l1, best_train_mse, best_val_l1, best_val_mse, best_train_r2, best_val_r2, training_time
    

def inference(model, model_loc_save, graph_test, y_true_test, seed):
    set_random_seed(seed)
    model.load_state_dict(torch.load(model_loc_save))
    model.eval()

    start_time = time.time()  # Start timer
    with torch.no_grad():
        # y_pred_test = model(graph_test, graph_test.ndata['feat'], 'performance')
        embeddings_test = model(graph_test, graph_test.ndata['feat'], out_key='none')
        # print(embeddings_test)
        y_pred_test = model.predict(embeddings_test)

    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    num_train_examples =  len(y_pred_test) -  len(y_true_test)
    # print(num_train_examples)
    y_pred_test = y_pred_test[num_train_examples:] 
    # test_loss = F.mse_loss(y_pred_test, y_true_test) 
    l1_loss_fn = L1Loss()
    mse_loss_fn = MSELoss()
    l1_test_loss = l1_loss_fn(y_pred_test, y_true_test) 
    mse_test_loss = mse_loss_fn(y_pred_test, y_true_test) 
    test_r2_score = r2_score(y_true_test.detach().numpy(), y_pred_test.detach().numpy())
    return y_pred_test, l1_test_loss, mse_test_loss, test_r2_score, inference_time



def create_test_graph_edges(og_hetero_graph, performance_edges_data_test, with_problem_class, with_high_level_features, high_level_features):
    perf_offset = og_hetero_graph.number_of_nodes('performance')
    problem_offset = og_hetero_graph.number_of_nodes('problem')
    n_performance_records = performance_edges_data_test.shape[0]
    algo_performance_src_array = performance_edges_data_test['src_id'].to_numpy()
    algo_performance_dst_array = [i+perf_offset for i in range(n_performance_records)]
    performance_problem_src_array = [i+perf_offset for i in range(n_performance_records)]
    performance_problem_dst_array, problem_dict = get_encoded_problem_instances(performance_edges_data_test['dst_id'].to_numpy(), problem_offset)

    graph_data = {
        ('algo', 'algo-performance', 'performance'): (algo_performance_src_array, algo_performance_dst_array),
        ('performance', 'algo-performance-reverse', 'algo'): (algo_performance_dst_array, algo_performance_src_array),
        ('performance', 'performance-problem', 'problem'): (performance_problem_src_array, performance_problem_dst_array),
        ('problem', 'problem-performance', 'performance'): (performance_problem_dst_array, performance_problem_src_array),
       
    }    


    return graph_data

def create_test_graph(hetero_graph, performance_edges_data_test, edge_dict, ela_df, embedding_dim):
    ela_instance_tensor = torch.tensor(ela_df.values).to(torch.float32)

    ground_truth_test = torch.tensor(performance_edges_data_test['score'].tolist()).to(torch.float32).unsqueeze(1)
    performance_arr_test_torch = nn.init.kaiming_uniform_(nn.Parameter(torch.randn(ground_truth_test.shape[0], embedding_dim), requires_grad=False))
    merged_problem_feat_test = torch.cat((hetero_graph.nodes['problem'].data['feat'], ela_instance_tensor), 0)
    merged_problem_feat_test = nn.Parameter(merged_problem_feat_test, requires_grad=False)
    merged_performance_feat_test = torch.cat((hetero_graph.nodes['performance'].data['feat'], performance_arr_test_torch), 0)
    merged_performance_feat_test = nn.Parameter(merged_performance_feat_test, requires_grad=False)
    edge_ids = {}
    for etype in hetero_graph.etypes:
        edge_ids[etype] = hetero_graph.edges[etype].data['id']

    test_edges = create_test_graph_edges(hetero_graph, performance_edges_data_test)

    # add new nodes
    num_new_problems = ela_df.shape[0]
    num_new_performance = performance_edges_data_test.shape[0]
    hetero_graph.add_nodes(num_new_problems, ntype='problem')
    hetero_graph.add_nodes(num_new_performance, ntype='performance')
    # add new edges
    for edge_type in test_edges:
        arr_src, arr_dst = test_edges[edge_type]
        hetero_graph.add_edges(arr_src, arr_dst, etype=edge_type[1])
        tt = ( torch.ones(len(arr_src), dtype=torch.long) * edge_dict[edge_type[1]])
        edge_ids[edge_type[1]] = torch.cat((edge_ids[edge_type[1]], tt), 0)
        hetero_graph.edges[edge_type[1]].data["id"] = edge_ids[edge_type[1]]

    hetero_graph.nodes['problem'].data['feat'] = merged_problem_feat_test
    hetero_graph.nodes['performance'].data['feat'] = merged_performance_feat_test
    return hetero_graph, ground_truth_test


def prepare_splits(problems_data, performance_data, train_mask, val_mask, test_mask, normalize_flag=True):
    problems_train_f = problems_data[train_mask]
    problems_val_f = problems_data[val_mask]
    problems_test_f = problems_data[test_mask]
    
    problem_instances_train_f = problems_train_f['label'].values
    problem_instances_val_f = problems_val_f['label'].values
    problem_instances_test_f = problems_test_f['label'].values

    performance_train_f = performance_data[performance_data['dst_id'].isin(problem_instances_train_f)]
    performance_val_f = performance_data[performance_data['dst_id'].isin(problem_instances_val_f)]
    performance_test_f = performance_data[performance_data['dst_id'].isin(problem_instances_test_f)]

    ela_train_f = problems_train_f['ela'].apply(ast.literal_eval).apply(pd.Series)
    ela_val_f = problems_val_f['ela'].apply(ast.literal_eval).apply(pd.Series)
    ela_test_f = problems_test_f['ela'].apply(ast.literal_eval).apply(pd.Series)
    if normalize_flag:
        scaler = MinMaxScaler()
        ela_train_f = pd.DataFrame(scaler.fit_transform(ela_train_f), index=ela_train_f.index)
        ela_val_f = pd.DataFrame(scaler.transform(ela_val_f), index=ela_val_f.index)
        ela_test_f = pd.DataFrame(scaler.transform(ela_test_f), index=ela_test_f.index)
    return problems_train_f, problems_val_f, problems_test_f, performance_train_f, performance_val_f, performance_test_f, ela_train_f, ela_val_f, ela_test_f

def read_input_data(dim, budget, algo, loc, perf_data_str):
    problem_nodes_data = pd.read_csv(f'{loc}/../data/graph_data/problem/problem_instance_{dim}D.csv')
    performance_edges_data = pd.read_csv(f'{loc}/../data/graph_data/performance/{perf_data_str}performance_edge_{algo}_{dim}_{budget}.csv')
    algo_nodes_data = pd.read_csv(f'{loc}/../data/graph_data/algo/algo_instance_{algo}.csv')
    return problem_nodes_data, performance_edges_data, algo_nodes_data

def set_random_seed(seed):
    torch.set_num_threads(1)
    """Set the seed for reproducibility."""
    random.seed(seed)             # Python random module.
    np.random.seed(seed)          # Numpy module.
    torch.manual_seed(seed)       # CPU tensors.





def create_data_splits(problem_instances, outer_fold):
    index = problem_instances.index.tolist()
    problem_instances = problem_instances.to_numpy()
    train_mask = np.zeros(len(problem_instances), dtype=bool)
    val_mask = np.zeros(len(problem_instances), dtype=bool)
    test_mask = np.zeros(len(problem_instances), dtype=bool)

    test_iid = outer_fold
    range_without_test_iid = [i for i in range(1, 6) if i != test_iid]
    val_iid = np.random.choice(range_without_test_iid, 1)[0]

    for i in range(1, 25):
        for j in range(1, 6):
            idx = np.where(problem_instances == f'{i}_{j}')
            if j == val_iid:
                val_mask[idx] = True
            elif j == test_iid:
                test_mask[idx] = True
            else:
                train_mask[idx] = True
    return train_mask, val_mask, test_mask, val_iid


def get_best_parameters(seed, perf_file):
    df = pd.read_csv(perf_file)
    df = df[df['seed'] == seed]
    max_r2_test = df['r2_test'].max()
    df = df[df['r2_test'] == max_r2_test]
    best_row = df.iloc[0]
    return int(best_row['n_hid']), int(best_row['n_heads']), int(best_row['n_layers']), int(best_row['patience']), best_row['factor'], best_row['dropout']
