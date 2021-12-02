TASK = 'antenna_selection' # one of 'antenna_selection', 'single_cast_beamforming'

if TASK =='antenna_selection':
    ANTENNA_NFEATS = 3
    EDGE_NFEATS = 9
    VAR_NFEATS = 6

    IN_FEATURES = 219


elif TASK == 'single_cast_beamforming':
    ANTENNA_NFEATS = 9
    EDGE_NFEATS = 3
    VAR_NFEATS = 10

    IN_FEATURES = 219

