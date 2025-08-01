from pyhdf.SD import SD, SDC
def inspect_hdf4(filename):
    try:
        hdf = SD(filename, SDC.READ)

        print(f"\n--- GLOBAL ATTRIBUTES in {filename} ---")
        for attr_name, attr_value in hdf.attributes().items():
            print(f"{attr_name} = {attr_value}")

        datasets = hdf.datasets()
        print(f"\n--- DATASETS ({len(datasets)}) ---")
        for i, (name, info_tuple) in enumerate(datasets.items(), 1):
            rank, dims, dtype, nattrs = info_tuple
            print(f"\nDataset {i}: {name}")
            print(f"  Rank (dimensions): {rank}")
            print(f"  Dimension sizes: {dims}")
            print(f"  Data type: {dtype}")
            print(f"  Number of Attributes: {nattrs}")
            
            dset = hdf.select(name)

            # Print attributes of this dataset
            attrs = dset.attributes()
            if attrs:
                print(f"  Attributes:")
                for attr_name, attr_value in attrs.items():
                    print(f"    {attr_name} = {attr_value}")
            else:
                print("  No attributes.")
                
            # Optional: print actual data shape
            try:
                data = dset.get()
                print(f"  Data shape: {data.shape}")
            except Exception as e:
                print(f"  Could not read data: {e}")

    except Exception as e:
        print(f"Error opening file: {e}")