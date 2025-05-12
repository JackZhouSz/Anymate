class AnymateArgs:
    def __init__(self):
        self.checkpoint_joint = "Anymate/checkpoints/joint/bert-transformer_latent-train-8gpu-finetune.pth.tar"
        self.checkpoint_conn = "Anymate/checkpoints/conn/bert-attendjoints_con_combine-train-8gpu-finetune.pth.tar"
        self.checkpoint_skin = "Anymate/checkpoints/skin/bert-attendjoints_combine-train-8gpu-finetune.pth.tar"

        self.device = "cuda"
        self.num_joints = 96


class UIArgs:
    def __init__(self):
        self.checkpoint_joint = "Anymate/checkpoints/joint/bert-transformer_latent-train-8gpu-finetune.pth.tar"
        self.checkpoint_conn = "Anymate/checkpoints/conn/bert-attendjoints_con_combine-train-8gpu-finetune.pth.tar"
        self.checkpoint_skin = "Anymate/checkpoints/skin/bert-attendjoints_combine-train-8gpu-finetune.pth.tar"

ui_args = UIArgs()
anymate_args = AnymateArgs()