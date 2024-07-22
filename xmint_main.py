import torch
from xmint_model import xMINT
from xmint_dataset import xMINTDataset
from torch.utils.data import DataLoader
from xmint_utils import train, validate_and_save

torch.manual_seed(42)

batch_size = 4
num_gene_total = 376
num_known_gene = 200
training_sample_name = 'Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE'
validation_sample_name = 'Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE'
model_dir = 'imputation_model_tonsil'
imputation_results_dir = 'results'

# Training
trainset = xMINTDataset(slice_folder=f'{training_sample_name}_outs/slices', 
                         WSI_path=f'{training_sample_name}_outs/{training_sample_name}_he_image.tif', 
                         coor_folder=f'{training_sample_name}_outs/relative_cor_group', 
                         gene_folder=f'{training_sample_name}_outs/gene_group',
                         num_known_gene=num_known_gene)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

model_train = xMINT(
    num_gene_total=num_gene_total, 
    num_known_gene=num_known_gene, 
    embed_size = 256, 
    transformer_heads=8, 
    num_transformer_layers=4, 
    transformer_forward_expansion=8)

train(
    model=model_train, 
    train_loader=train_loader, 
    epochs=100, 
    lr=1e-4, 
    model_dir=model_dir)

# Validation
valset = xMINTDataset(slice_folder=f'{validation_sample_name}_outs/slices', 
                       WSI_path=f'{validation_sample_name}_outs/{validation_sample_name}_he_image.tif', 
                       coor_folder=f'{validation_sample_name}_outs/relative_cor_group', 
                       gene_folder=f'{validation_sample_name}_outs/gene_group',
                       num_known_gene=num_known_gene)

model_imputation = xMINT(
    num_gene_total=num_gene_total, 
    num_known_gene=num_known_gene, 
    embed_size = 256, 
    transformer_heads=8, 
    num_transformer_layers=4, 
    transformer_forward_expansion=8)

model_imputation.load_state_dict(torch.load(f'{model_dir}/model_100.pth'))
validate_loader = DataLoader(valset, batch_size=1, shuffle=False)

validate_and_save(
    model=model_imputation, 
    val_loader=validate_loader, 
    results_dir=imputation_results_dir)
