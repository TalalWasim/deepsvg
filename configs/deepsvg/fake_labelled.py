from .defaults_gest import *

# max_num_groups = 6
# max_seq_len = 77
# max_total_len = 82

max_num_groups = 8
max_seq_len = 34
max_total_len = 48

class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = True
        self.use_vae = False

        self.max_num_groups = max_num_groups
        self.max_seq_len = max_seq_len
        self.max_total_len = max_total_len
        
        self.dim_z = 128


class Config(Config):
    def __init__(self, num_gpus=1):
        super().__init__(num_gpus=num_gpus)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_uni = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67,
                           68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                           81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100,
                           101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                           113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

        self.learning_rate = 2e-4 * num_gpus
        self.batch_size = 64 * num_gpus

        self.val_every = 500
        
        self.dataloader_module = "deepsvg.svg_dataset"
        self.data_dir = "./dataset/data_fake/data/"
        self.meta_filepath = "./dataset/data_fake/svg_meta_labelled_2.csv"
#         self.data_dir = "./dataset/data_bast/data_quant_0.85/"
#         self.meta_filepath = "./dataset/data_bast/svg_meta_quant_0.85.csv"
        
        self.max_num_groups = max_num_groups
        self.max_seq_len = max_seq_len
        self.max_total_len = max_total_len
