import os
import shutil
import argparse
import torch
import torch.multiprocessing as mp
from Anymate.utils.train_utils import train_model
import yaml
from Anymate.dataset import AnymateDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyG DGCNN')
    parser.add_argument('--config', type=str, default='joints', help='load decoder')
    parser.add_argument('--split', action='store_true', help='use split dataset')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print('world_size', world_size)
    
    #load config file
    config_folder = './Anymate/configs'
    assert os.path.exists(os.path.join(config_folder, args.config+'.yaml')), f"Config file {os.path.join(config_folder, args.config+'.yaml')} not found"
    with open(os.path.join(config_folder, args.config+'.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in config['args'].items():
        setattr(args, key, value)
    setattr(args, 'decoder', config['model']['decoder'])
    args.logdir = os.path.join(args.logdir, args.mode + '-' + config['model']['encoder']+ '-' + config['model']['decoder'])
    args.checkpoint = os.path.join(args.checkpoint,  args.mode + '-' + config['model']['encoder']+ '-' + config['model']['decoder'])
    print(args)
    
    # create checkpoint dir and log dir
    if not os.path.isdir(args.checkpoint):
        print("Create new checkpoint folder " + args.checkpoint)
        os.makedirs(args.checkpoint, exist_ok=True)
    if not args.resume:
        if os.path.isdir(args.logdir):
            shutil.rmtree(args.logdir)
            os.makedirs(args.logdir, exist_ok=True)
        else:
            os.makedirs(args.logdir, exist_ok=True)
    global train_dataset

    if not args.split:
        # create a shared memory dataset dictionary
        train_dataset = AnymateDataset(name=args.trainset, root=args.root)
        train_dataset.data_list = [data for data in train_dataset.data_list if data['vox'].shape[0] != 0]
        print('train_dataset', len(train_dataset.data_list))
        import multiprocessing
        manager = multiprocessing.Manager()
        shared_dict = manager.dict()
        shared_dict['train_dataset'] = train_dataset
    else:
        shared_dict = None

    # Try different ports until we find a free one
    port = 12355
    while port < 65535:  # Max port number
        try:
            mp.spawn(train_model, args=(world_size, config, args, shared_dict, port), nprocs=world_size)
            break
        except Exception as e:
            if "address already in use" in str(e).lower():
                print(f"Port {port} is already in use, trying next port")
                port += 1
            else:
                print(f"Error starting training on port {port}: {e}")
                raise e
    print(f"Successfully started training on port {port}")