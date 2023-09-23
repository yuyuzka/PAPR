from LBSNData import LBSNData
from near_location_query import Loc_Query_System
from utils import *
from loss_fn import *
from neg_sampler import *
from model import STiSAN
from trainer import *


if __name__ == "__main__":
    # Setting paths
    path_prefix = '..//STiSAN//TKY/'
    data_name = 'TKY'
    print("Dataset: ", data_name)
    finished_epoch=0
    raw_data_path = path_prefix + data_name + '.txt'
    clean_data_path = path_prefix + data_name + '.data'
    loc_query_path = path_prefix + data_name + '_loc_query.pkl'
    matrix_path = path_prefix + data_name + '_st_matrix.data'
    log_path = path_prefix + data_name + '_PAPR_log.txt'
    result_path = path_prefix + data_name + '_PAPR_result.txt'
    model_path = path_prefix+data_name+"model"
    load_path=path_prefix+data_name+"dropout_model"+str(finished_epoch-1)+".pth"

    # Data Process details
    min_loc_freq = 10
    min_user_freq = 20
    map_level = 12
    n_nearest = 2000
    max_len = 100

    if os.path.exists(clean_data_path):
        dataset = unserialize(clean_data_path)
    else:
        dataset = LBSNData(data_name, raw_data_path, min_loc_freq, min_user_freq, map_level)
        serialize(dataset, clean_data_path)
    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#POIs:", dataset.n_loc - 1)
    print("#median seq len:", np.median(np.array(length)))

    # Searching nearest POIs
    quadkey_processor = dataset.QUADKEY
    loc2quadkey = dataset.loc2quadkey
    loc_query_sys = Loc_Query_System()
    if os.path.exists(loc_query_path):
        loc_query_sys.load(loc_query_path)
    else:
        loc_query_sys.build_tree(dataset)
        loc_query_sys.prefetch_n_nearest_locs(n_nearest)
        loc_query_sys.save(loc_query_path)
        loc_query_sys.load(loc_query_path)

    # Building Spatial-Temporal Relation Matrix
    if os.path.exists(matrix_path):
        st_matrix = unserialize(matrix_path)
    else:
        dataset.spatial_temporal_matrix_building(matrix_path)
        st_matrix = unserialize(matrix_path)
    print("Data Partition...")
    train_data, eval_data = dataset.data_partition(max_len, st_matrix)

    # Setting training details
    device = 'cpu'
    num_workers = 0
    n_nearest_locs = 2000
    num_epoch = 20
    train_bsz = 32
    eval_bsz = 32
    train_num_neg = 15
    eval_num_neg = 100
    user_visited_locs = get_visited_locs(dataset)
    loss_fn = WeightedBCELoss(temperature=100.0)
    dis_loss=distance_loss(dis_weight=0.001)
    train_sampler = KNNSampler(loc_query_sys, n_nearest_locs, user_visited_locs, 'training', True)
    eval_sampler = KNNSampler(loc_query_sys, n_nearest_locs, user_visited_locs, 'evaluating', True)

    # Model details
    model = STiSAN(dataset.n_loc+1,
                   dataset.n_quadkey+1,
                   features=128,
                   exp_factor=1,
                   k_t=10,
                   k_g=15,
                   depth=4,
                   dropout=0.7)
    model.to(device)

    # Starting training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    train(model,
          max_len,
          train_data,
          train_sampler,
          train_bsz,
          train_num_neg,
          num_epoch,
          quadkey_processor,
          loc2quadkey,
          eval_data,
          eval_sampler,
          eval_bsz,
          eval_num_neg,
          optimizer,
          loss_fn,
          device,
          num_workers,
          log_path,
          result_path,
          dataset.idx2gps,
          model_path,
          finished_epoch,
          dis_loss)