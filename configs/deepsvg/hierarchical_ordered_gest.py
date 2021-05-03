from .default_icons import *


class ModelConfig(Hierarchical):
    def __init__(self):
        super().__init__()

        self.label_condition = False
        self.use_vae = False


class Config(Config):
    def __init__(self, num_gpus=2):
        super().__init__(num_gpus=num_gpus)

        self.model_cfg = ModelConfig()
        self.model_args = self.model_cfg.get_model_args()

        self.filter_category = None

        self.learning_rate = 1e-3 * num_gpus
        self.batch_size = 120 * num_gpus

        self.val_every = 200
        
        self.dataloader_module = "deepsvg.svg_dataset"
        self.collate_fn = None
        self.data_dir = "./dataset/data_quant_0.95/"
        self.meta_filepath = "./dataset/svg_meta_quant_0.95.csv"
