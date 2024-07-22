import torch
import os
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd


def train(model, train_loader, epochs, lr, model_dir):
    """Training function. Trained models will be saved in model_dir directory every 10 epochs."""  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss_train = 0
        num_batches_train = 0

        for _, (image_tensor, coors, known_gene, targets_train, _, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            image_tensor, coors, known_gene, targets_train = [x.to(device) for x in [image_tensor, coors, known_gene, targets_train]]

            perm_idx = torch.randperm(coors.size(1))
            coors = coors[:, perm_idx]
            known_gene = known_gene[:, perm_idx]
            targets_train = targets_train[:, perm_idx]

            optimizer.zero_grad()
            output_embeddings_train = model(image_tensor, coors, known_gene)
            probs_train = torch.nn.functional.softmax(output_embeddings_train, dim=-1)
            loss_train = lossf(probs_train, targets_train)
            loss_train.backward()
            optimizer.step()

            total_loss_train += loss_train.item()
            num_batches_train += 1

        avg_loss_train = total_loss_train / num_batches_train
        print(f'Epoch {epoch + 1} Training Loss: {avg_loss_train:.8f}')

        if (epoch+1) % 10==0:
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{model_dir}/model_{epoch+1}.pth')


def validate_and_save(model, val_loader, results_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  

    all_targets = []  
    all_outputs = []
    with torch.no_grad():  

        for _, (image_tensor, coors_tensor, normalized_known_gene, normalized_unknown_gene, _, cells, unknown_gene_ids) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating"):
            image_tensor, coors_tensor, normalized_known_gene, normalized_unknown_gene = (
                image_tensor.to(device),
                coors_tensor.to(device),
                normalized_known_gene.to(device), 
                normalized_unknown_gene.to(device)
            )
            output_embeddings = model(image_tensor, coors_tensor, normalized_known_gene)
            probs_unknown_genes = torch.softmax(output_embeddings, dim=-1)

            cells_np = [str(cell[0].item()) for cell in cells]
            unknown_gene_ids_np = [unknown_gene_id[0] for unknown_gene_id in unknown_gene_ids]

            targets_df = pd.DataFrame(normalized_unknown_gene.squeeze().cpu().numpy(), index=cells_np, columns=unknown_gene_ids_np)
            outputs_df = pd.DataFrame(probs_unknown_genes.squeeze().cpu().numpy(), index=cells_np, columns=unknown_gene_ids_np)

            all_targets.append(targets_df)
            all_outputs.append(outputs_df)

            # Group every 500 batches to manage memory better
            if len(all_targets) >= 500:
                combined_targets_partial = pd.concat(all_targets).groupby(level=0).mean()
                combined_outputs_partial = pd.concat(all_outputs).groupby(level=0).mean()
                all_targets = [combined_targets_partial]
                all_outputs = [combined_outputs_partial]

        # Final grouping and concatenation of any remaining DataFrames
        combined_targets = pd.concat(all_targets).groupby(level=0).mean()
        combined_outputs = pd.concat(all_outputs).groupby(level=0).mean()

        # Assuming combined_targets and combined_outputs are already defined and have the same columns
        correlations = []

        # Iterate over the columns
        for column in combined_targets.columns:
            if column in combined_outputs.columns:  # Ensure the column exists in both DataFrames
                # Calculate the Spearman correlation for this column
                corr, _ = spearmanr(combined_targets[column], combined_outputs[column])
                correlations.append(corr)

        # Calculate the median of the correlations
        median_correlation = pd.Series(correlations).median()

        # Print the median Spearman correlation
        print("Median Spearman Correlation:", median_correlation)

        # Save the results to CSV files.
        os.makedirs(results_dir, exist_ok=True)  # Ensure directory exists.
        combined_targets.to_csv(os.path.join(results_dir, 'Truth.csv'))
        combined_outputs.to_csv(os.path.join(results_dir, 'xMINT.csv'))
        print('All predictions and targets are saved and combined')
        