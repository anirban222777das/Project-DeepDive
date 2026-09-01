import torch

checkpoint_path = 'models/checkpoints_drqn/drqn_step_1000000.pt'
export_path = 'models/drqn_final.pt'

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
lightweight_checkpoint = {
    'online_network_state_dict': checkpoint['online_network_state_dict'],
    'step': checkpoint['step'],
    'episode': checkpoint['episode']
}

torch.save(lightweight_checkpoint, export_path)
print("Exported lightweight model!")
