import os
import pandas as pd
import cv2
import tifffile as tf
import numpy as np
import zarr
from scipy.sparse import csr_matrix


def read_transformation_matrix(filepath):
    """Reads and returns the transformation matrix from a CSV file."""
    try:
        return pd.read_csv(filepath, header=None).values
    except Exception as e:
        raise IOError(f"Error reading transformation matrix: {e}")

def transform_coordinates(df, multiplier, transformation_matrix, downsample):
    """Applies transformation matrix to the coordinates and adjusts for downsampling."""
    df['pixel_x'] = df['x_centroid'] * multiplier
    df['pixel_y'] = df['y_centroid'] * multiplier
    
    transformed = np.dot(np.linalg.inv(transformation_matrix),
                         np.vstack([df['pixel_x'], df['pixel_y'], np.ones(len(df))]))
    
    df['pixel_x'] = (transformed[0, :] // downsample).astype(int)
    df['pixel_y'] = (transformed[1, :] // downsample).astype(int)
    return df

def group_cells(df, patch_size, output_folder):
    """Groups cells based on their transformed coordinates and patch size."""
    df['group_x'] = (df['pixel_x'] // patch_size).astype(int)
    df['group_y'] = (df['pixel_y'] // patch_size).astype(int)
    df['group_name'] = df['group_x'].astype(str) + '_' + df['group_y'].astype(str)
    output_file_path = os.path.join(output_folder, "grouped_cells.csv")
    df.to_csv(output_file_path, index=False)
    print(f'Grouped cells saved to {output_file_path}')

def transform_coordinates_and_group(csv_file, multiplier, patch_size, output_folder, aligned, transformation_matrix_dir, downsample):
    """Main function to process cell data, transform coordinates and group cells."""
    df = pd.read_csv(csv_file)
    if not aligned:
        transformation_matrix = read_transformation_matrix(transformation_matrix_dir)
    else:
        transformation_matrix = np.eye(3)
    transformed = transform_coordinates(df, multiplier, transformation_matrix, downsample)
    group_cells(transformed, patch_size, output_folder)

def convert_image(input_filename, output_filename, downsample):
    """Converts an OME TIFF image from RGB to BGR and downsamples it."""
    with tf.TiffFile(input_filename) as tif:
        main_image = tif.asarray()
        if main_image.shape[0] == 3:
            width, height = main_image.shape[2], main_image.shape[1]
            main_image = np.transpose(main_image, (1, 2, 0))
        else:
            width, height = main_image.shape[1], main_image.shape[0]

        image_bgr = cv2.cvtColor(main_image, cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (width//downsample, height//downsample))

        cv2.imwrite(output_filename, image_bgr)
        print(f"Image saved to {output_filename}")

def hex_to_custom(hex_digit):
    """Converts a hexadecimal digit to a custom alphabet character."""
    if hex_digit.isdigit():
        return chr(ord('a') + int(hex_digit))
    else:
        return chr(ord('k') + ord(hex_digit) - ord('a'))

def convert_cell_id(cell_ids):
    """Converts a list of cell IDs from hexadecimal to a custom format."""
    results = []
    for prefix, suffix in cell_ids:
        hex_prefix = format(prefix, 'x').zfill(8)
        custom_prefix = ''.join(hex_to_custom(digit) for digit in hex_prefix)
        results.append(f"{custom_prefix}-{suffix}")
    return results

def process_sc_matrix_zarr_to_dataframe(zarr_file):
    """Converts a single-cell matrix stored in a Zarr format to a pandas DataFrame."""
    cell_features = zarr.open(zarr_file, mode='r')['cell_features']
    data = cell_features['data'][:]
    indices = cell_features['indices'][:]
    indptr = cell_features['indptr'][:]
    csr_matrix_ = csr_matrix((data, indices, indptr))
    if 'cell_id' in list(cell_features.keys()):
        cell_id = cell_features['cell_id'][:]
        cell_id = convert_cell_id(cell_id)
    else:
        cell_id = range(1, csr_matrix_.shape[1]+1)
    feature_ids = cell_features.attrs['feature_ids']
    df = pd.DataFrame(csr_matrix_.toarray(), index=feature_ids, columns=cell_id)
    return df

def convert_sc(zarr_file, output_folder):
    """
    Processes a single-cell matrix data from a Zarr file into a DataFrame and
    exports the data into multiple CSV files for easier handling.
    """
    sc_df = process_sc_matrix_zarr_to_dataframe(zarr_file)
    directory = output_folder
    os.makedirs(directory, exist_ok=True)
    num_columns = sc_df.shape[1]
    columns_per_file = 10000
    for start in range(0, num_columns, columns_per_file):
        end = start + columns_per_file
        if end > num_columns:
            end = num_columns
        subset = sc_df.iloc[:, start:end]
        filename = f"{directory}/genes_part_{start // columns_per_file + 1}.csv"
        subset.to_csv(filename)
    print(f'Single cell data saved to {directory}')

def create_seq(grouped_cell_path, image_path, genes_folder, output_dir1, output_dir2, output_dir3, num_total_gene, rep):
    # Read input CSV file
    data = pd.read_csv(grouped_cell_path)
    img = cv2.imread(image_path)

    # Ensure output directories exist
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    os.makedirs(output_dir3, exist_ok=True)
    
    # Load data from files in the input directory into a dictionary
    full_data = {}
    for filename in os.listdir(genes_folder):
        if filename.endswith('.csv'):
            full_data[filename] = pd.read_csv(os.path.join(genes_folder, filename), index_col=0)

    # Group and process each group of data
    grouped = data.groupby('group_name')
    for group_name, group_data in grouped:
        group_data = group_data.set_index('cell_id')
        slice_path = os.path.join(output_dir3, f"{group_name}.png")
        
        # Process image and save slices if not exist
        if not os.path.exists(slice_path):
            x_start, y_start = [max(0, int(i) * 1024) for i in group_name.split('_')]
            height, width, _ = img.shape
            pad_bottom = max(0, y_start + 1024 - height)
            pad_right = max(0, x_start + 1024 - width)
            
            if pad_bottom > 0 or pad_right > 0:
                img_padded = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)
            else:
                img_padded = img

            slice = img_padded[y_start:y_start+1024, x_start:x_start+1024, :]
            cv2.imwrite(slice_path, slice)

        # Calculate modified pixel coordinates
        mod_x = group_data['pixel_x'] % 1024
        mod_y = group_data['pixel_y'] % 1024       
        coordinates_df = pd.DataFrame({'mod_x': mod_x, 'mod_y': mod_y})
        coordinates_df.sort_index(inplace=True)

        # Repeat DataFrame to fill at least 1024 rows, calculate chunks and save
        repeat_times = (1024 // len(coordinates_df)) * rep + 2 * rep if len(coordinates_df) > 0 else 1 * rep
        num_chunks = len(coordinates_df) // 1024 * rep + 1 * rep
        padded_coordinates_df = pd.concat([coordinates_df] * repeat_times).head(1024 * num_chunks)
        split_coordinates_dfs = np.array_split(padded_coordinates_df, num_chunks)
        for i, df in enumerate(split_coordinates_dfs):
            df.to_csv(os.path.join(output_dir1, f"{group_name}_mod_coordinates_part_{i}.csv"))

        # Process gene data for each cell from pre-loaded full_data
        gene_data = {}
        for cell_id in group_data.index.unique():
            column_name = str(cell_id)
            for df in full_data.values():
                if column_name in df.columns:
                    matched_data = df[column_name].head(num_total_gene)
                    if cell_id not in gene_data:
                        gene_data[cell_id] = matched_data.tolist()
        
        # Create a DataFrame for the gene data, repeat, and save
        index_names = list(full_data.values())[0].index[:num_total_gene].tolist()  # Assuming all files have same row index names
        match_df = pd.DataFrame.from_dict(gene_data, orient='index', columns=index_names)
        match_df.sort_index(inplace=True)
        
        repeat_times = (1024 // len(match_df)) * rep + 2 * rep if len(match_df) > 0 else 1 * rep
        padded_match_df = pd.concat([match_df] * repeat_times).head(1024 * num_chunks)
        split_match_dfs = np.array_split(padded_match_df, num_chunks)
        for i, df in enumerate(split_match_dfs):
            df.to_csv(os.path.join(output_dir2, f"{group_name}_matched_gene_part_{i}.csv"), index_label='cell_id')
    print(f'Sequences saved to {output_dir1}, {output_dir2}, {output_dir3}.')
