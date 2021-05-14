from .default_gest import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False
        self.max_num_groups = 8
        self.max_total_len = 78
        self.max_seq_len = 62


class Config(Config):
    def __init__(self, num_gpus=1):
        super().__init__(num_gpus=num_gpus)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        self.learning_rate = 1e-3 * num_gpus
        self.batch_size = 128 * num_gpus

        self.val_every = 500
        
        self.dataloader_module = "deepsvg.svg_dataset"
        self.collate_fn = None
        self.data_dir = "./dataset/data_fake/data/"
        self.meta_filepath = "./dataset/data_fake/svg_meta.csv"
#         self.data_dir = "./dataset/data_bast/data_quant_0.85/"
#         self.meta_filepath = "./dataset/data_bast/svg_meta_quant_0.85.csv"
        
        self.max_num_groups = 8
        self.max_total_len = 78
        self.max_seq_len = 62
