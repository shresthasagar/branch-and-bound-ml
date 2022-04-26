import os

ROOT = '/scratch/sagar/Projects/combopt/branch-and-bound-ml'

# TASK = 'single_group_as_bm' # one of 'antenna_selection', 'single_cast_beamforming'
TASK = 'antenna_selection'
# TASK = 'robust_beamforming'
DEBUG = False

DEVICE = 'cuda'

DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER = 30
DAGGER_NUM_VALID_EXAMPLES_PER_ITER = 30
DAGGER_NUM_ITER = 20
BB_MAX_STEPS = 10000

REUSE_DATASET = True

if TASK =='antenna_selection':
    ANTENNA_NFEATS = 4
    EDGE_NFEATS = 9
    VAR_NFEATS = 8
    NODE_DEPTH_INDEX = 5

    IN_FEATURES = 219

    DATA_PATH = os.path.join(ROOT, 'antenna_selection/data/data_multiprocess')
    MODEL_PATH = os.path.join(ROOT, 'antenna_selection/trained_models')
    RESULT_PATH = os.path.join(ROOT, 'antenna_selection/data')

    LOAD_MODEL = False
    LOAD_MODEL_PATH = os.path.join(ROOT, 'antenna_selection/trained_models/gnn2_iter_3')

    CLASS_IMBALANCE_WT = 11 # for 8,3,5 (N,M,L) use 11, for 12,6,5 max_ant use 11
    ETA_EXP = 1.0
    LAMBDA_ETA = 1e-5 # for 8,3,5 use 1e-4, for 12, 6, 8 use 1e-5

elif TASK =='robust_beamforming':
    ANTENNA_NFEATS = 4
    EDGE_NFEATS = 9
    VAR_NFEATS = 8
    NODE_DEPTH_INDEX = 5

    IN_FEATURES = 219

    DATA_PATH = os.path.join(ROOT, 'robust_beamforming/data/dagger')
    MODEL_PATH = os.path.join(ROOT, 'robust_beamforming/trained_models')
    RESULT_PATH = os.path.join(ROOT, 'robust_beamforming/data/')
    VALIDATION_PATH = os.path.join(ROOT, 'robust_beamforming/validation_set')
    
    LOAD_MODEL = False
    LOAD_MODEL_PATH = os.path.join(ROOT, 'robust_beamforming/trained_models/gnn2_iter_3')

    CLASS_IMBALANCE_WT = 11
    ETA_EXP = 1.0
    LAMBDA_ETA = 0 # for 8,3,5 use 1e-4, for 12, 6, 8 use 1e-5
        
elif TASK =='single_group_as_bm':
    ANTENNA_NFEATS = 13
    EDGE_NFEATS = 3
    VAR_NFEATS = 10
    NODE_DEPTH_INDEX = 9

    IN_FEATURES = 219

    DATA_PATH = os.path.join(ROOT, 'single_group_as_bm/data/dagger')
    MODEL_PATH = os.path.join(ROOT, 'single_group_as_bm/trained_models')
    RESULT_PATH = os.path.join(ROOT, 'single_group_as_bm/data/')

    LOAD_MODEL = False
    LOAD_MODEL_PATH =  os.path.join(ROOT, 'single_group_as_bm/trained_models/gnn2_iter_3')

    CLASS_IMBALANCE_WT = 24
    ETA_EXP = 1.0
    LAMBDA_ETA = 0 # for 8,3,5 use 1e-4, for 12, 6, 8 use 1e-5
    
elif TASK == 'single_cast_beamforming':
    ANTENNA_NFEATS = 9
    EDGE_NFEATS = 3
    VAR_NFEATS = 10
    NODE_DEPTH_INDEX = 9


    IN_FEATURES = 219

    DATA_PATH = os.path.join(ROOT, 'single_beamforming/data')
    MODEL_PATH = os.path.join(ROOT, 'single_beamforming/trained_models/gnn1.model')


