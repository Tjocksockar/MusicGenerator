from data_processor import *
from model import *

# Parameters
seq_len = 100

# Data preparation
processed_data = formate_data()
class_to_ind, ind_to_class = create_dictionaries(processed_data)
