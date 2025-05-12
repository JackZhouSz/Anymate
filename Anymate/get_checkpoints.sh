cd Anymate/checkpoints
mkdir joint
cd joint

echo "Downloading joint checkpoints..."
wget "https://huggingface.co/yfdeng/Anymate/resolve/main/checkpoints/joint/bert-transformer_latent-train-8gpu-finetune.pth.tar?download=true" -O bert-transformer_latent-train-8gpu-finetune.pth.tar

cd ..
mkdir conn
cd conn

echo "Downloading conn checkpoints..."
wget "https://huggingface.co/yfdeng/Anymate/resolve/main/checkpoints/conn/bert-attendjoints_con_combine-train-8gpu-finetune.pth.tar?download=true" -O bert-attendjoints_con_combine-train-8gpu-finetune.pth.tar

cd ..
mkdir skin
cd skin

echo "Downloading skin checkpoints..."
wget "https://huggingface.co/yfdeng/Anymate/resolve/main/checkpoints/skin/bert-attendjoints_combine-train-8gpu-finetune.pth.tar?download=true" -O bert-attendjoints_combine-train-8gpu-finetune.pth.tar

echo "Finished downloading checkpoints!"
