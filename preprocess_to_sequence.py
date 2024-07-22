from preprocess_utils import transform_coordinates_and_group, convert_image, convert_sc, create_seq

data_root = '.'
sample_names = ['Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE', 'Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE']

# preprocess original data into sequence.
for sample_name in sample_names:
    # Process cell data, transform coordinates and group cells into local regions.
    transform_coordinates_and_group(
        csv_file=f'{data_root}/{sample_name}_outs/cells.csv',
        multiplier=1/0.2125,
        patch_size=1024,
        output_folder=f'{data_root}/{sample_name}_outs',
        aligned=True,
        transformation_matrix_dir=f'{data_root}/{sample_name}_outs/{sample_name}_he_imagealignment.csv',
        downsample=2
    )

    # Downsample h&e image. 
    convert_image(
        input_filename=f'{data_root}/{sample_name}_outs/{sample_name}_he_image.ome.tif',
        output_filename=f'{data_root}/{sample_name}_outs/{sample_name}_he_image.tif',
        downsample=2
    )

    # Save single cell data into seperate csv files for easy usage.
    convert_sc(zarr_file=f'{data_root}/{sample_name}_outs/cell_feature_matrix.zarr', 
               output_folder=f"{data_root}/{sample_name}_outs/genes")
    
    # Create sequences for training
    create_seq(grouped_cell_path=f'{data_root}/{sample_name}_outs/grouped_cells.csv',
            image_path=f'{data_root}/{sample_name}_outs/{sample_name}_he_image.tif',
            genes_folder=f'{data_root}/{sample_name}_outs/genes',
            output_dir1=f'{data_root}/{sample_name}_outs/relative_cor_group',
            output_dir2=f'{data_root}/{sample_name}_outs/gene_group',
            output_dir3=f'{data_root}/{sample_name}_outs/slices',
            num_total_gene=376,
            rep=5)

