import os

class setupFolders:
    def __init__(self, wfs):
        main_fold           = "./dataset/"
        main_result_path    = "./training_results/"
        sub_fold            = f"Dr0{wfs.Dr0_dataset[0]}--{wfs.Dr0_dataset[1]}/S{wfs.samp}_R{wfs.nPxPup}_Z{wfs.zModes[0]}-{wfs.zModes[1]}_D{wfs.D}"
        self.train_fold     = f"{main_fold}{sub_fold}/train"
        self.val_fold       = f"{main_fold}{sub_fold}/val"       
        self.model_path     = f"{main_result_path}{wfs.experimentName}/checkpoint/"
        self.de_path        = f"{main_result_path}{wfs.experimentName}/DE/"
        self.result_path    = f"{main_result_path}{wfs.experimentName}/validation_results/"
        self.log_path       = f"{main_result_path}{wfs.experimentName}/"
        self.tb_path    = f"{main_result_path}/tensorboard_summary/" + wfs.experimentName
            

        # Create directories if they don't exist
        os.makedirs(self.train_fold, exist_ok=True)
        os.makedirs(self.val_fold, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.de_path, exist_ok=True)
        os.makedirs(self.tb_path, exist_ok=True)