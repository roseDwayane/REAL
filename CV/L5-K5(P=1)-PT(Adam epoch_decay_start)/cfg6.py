
base_path = '../../dataset/fif'

features_single_path = '../../dataset/fif/features_single'
features_double_path = '../../dataset/fif/features_double'

features_single_valid_path = '../../dataset/fif/features_single_valid'
features_single_valid_all_path = '../../dataset/fif/features_single_valid_all'

epochs_path = '../../dataset/fif/epochs'
epochs_lpf_path = '../../dataset/fif/epochs_lpf'

class_names = ['CTL', 'CM']

class_paths = {'CTL': 'CTL',
               'FM': 'FM',
               'CM': 'CM',
               'CMFM': 'CMFM'}

class_labels = {class_name: idx for idx, class_name in enumerate(class_names)}

label_map_class = {idx: class_name for idx, class_name in enumerate(class_names)}

file_name_prefix = 'EC'

# 各种别名
sub_names = class_names
sub_paths = class_paths
sub_labels = class_labels

sub_name = sub_names
sub_path = sub_paths
sub_label = sub_labels
