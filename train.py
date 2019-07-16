from data_processor import *
from model import *


# Data preparation
processed_data = formate_data()
class_to_ind, ind_to_class = create_dictionaries(processed_data)

# Parameters
n_data = len(processed_data)
seq_len = 100
n_batches = int((n_data-1)/seq_len) # to make sure labels do not fall out of range

# Creating input sequences and labels
create_labeled_data(processed_data, class_to_ind, seq_len, n_batches)