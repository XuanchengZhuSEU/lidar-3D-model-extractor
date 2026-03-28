from pathlib import Path

from utils import (
    extract_from_las,
    get_classification_stats,
    las_to_shp,
    merge_las_files,
    shp_to_blender,
    visualize_las_pointcloud,
    visualize_reflection_intensity,
    visualize_shp_3d,
)


def main():
    """
    Detailed examples for utility functions in utils.py
    Data source: points_cloud_data/test_lidar.las
    """
    # -------------------------------------------------------------------------
    # 1) Input/output paths
    # -------------------------------------------------------------------------
    las_path = Path("points_cloud_data/test_lidar.las")
    results_dir = Path("results")
    shp_dir = results_dir / "shp"
    blender_dir = results_dir / "blender"
    npy_dir = results_dir / "numpy"

    shp_dir.mkdir(parents=True, exist_ok=True)
    blender_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    shp_path = shp_dir / "test_lidar.shp"
    obj_path = blender_dir / "test_lidar.obj"
    npy_path = npy_dir / "test_lidar_points.npy"
    merged_las_path = results_dir / "merged_test_lidar.las"

    if not las_path.exists():
        raise FileNotFoundError(f"Input LAS file not found: {las_path}")

    # -------------------------------------------------------------------------
    # 2) Execution switches
    #    Set True to run that example.
    # -------------------------------------------------------------------------

    ## this part is for point clouds processing
    # run_get_classification_stats = True
    # run_merge_las_files = False
    # run_las_to_shp = True
    # run_visualize_shp_3d = False
    # run_shp_to_blender = True
    # run_visualize_las_pointcloud = False
    # run_extract_from_las = True
    # run_visualize_reflection_intensity = False

    ## this part is for visualization
    run_get_classification_stats = False
    run_merge_las_files = False
    run_las_to_shp = False
    run_visualize_shp_3d = True
    run_shp_to_blender = False
    run_visualize_las_pointcloud = True
    run_extract_from_las = False
    run_visualize_reflection_intensity = True

    # -------------------------------------------------------------------------
    # 3) get_classification_stats example
    # -------------------------------------------------------------------------
    if run_get_classification_stats:
        print("\n=== Example: get_classification_stats ===")
        stats = get_classification_stats(str(las_path))
        for class_code in sorted(stats.keys()):
            print(
                f"class={class_code:>2}, "
                f"name={stats[class_code]['name']}, "
                f"count={stats[class_code]['count']}"
            )

    # -------------------------------------------------------------------------
    # 4) merge_las_files example
    #    Demonstration only: here we merge the same file twice.
    #    In real use, pass multiple different LAS files.
    # -------------------------------------------------------------------------
    if run_merge_las_files:
        print("\n=== Example: merge_las_files ===")
        input_files = [str(las_path), str(las_path)]
        merge_las_files(input_files, str(merged_las_path))
        print(f"Merged LAS saved to: {merged_las_path}")

    # -------------------------------------------------------------------------
    # 5) las_to_shp example
    # -------------------------------------------------------------------------
    if run_las_to_shp:
        print("\n=== Example: las_to_shp ===")
        gdf = las_to_shp(
            las_file_path=str(las_path),
            output_shp_path=str(shp_path),
            epsilon=2,
            min_cluster_size=100,
            alpha_shape_alpha=0.5,
            ground_search_k=50,
            crs="EPSG:26910",
            building_class=6,
            ground_class=2,
            verbose=True,
        )
        if gdf is not None:
            print(f"Shapefile created: {shp_path}")

    # -------------------------------------------------------------------------
    # 6) visualize_shp_3d example
    # -------------------------------------------------------------------------
    if run_visualize_shp_3d:
        print("\n=== Example: visualize_shp_3d ===")
        visualize_shp_3d(
            shp_path=str(shp_path),
            use_attribute_height=True,
            default_height=10.0,
            height_scale=1.0,
            random_color=True,
            base_on_attributes=True,
        )

    # -------------------------------------------------------------------------
    # 7) shp_to_blender example
    # -------------------------------------------------------------------------
    if run_shp_to_blender:
        print("\n=== Example: shp_to_blender ===")
        output_file = shp_to_blender(
            shp_path=str(shp_path),
            output_path=str(obj_path),
            format="obj",
            use_attribute_height=True,
            default_height=10.0,
            height_scale=1.0,
            base_on_attributes=True,
            merge_buildings=True,
            verbose=True,
        )
        print(f"Blender model output: {output_file}")

    # -------------------------------------------------------------------------
    # 8) visualize_las_pointcloud example
    # -------------------------------------------------------------------------
    if run_visualize_las_pointcloud:
        print("\n=== Example: visualize_las_pointcloud ===")
        visualize_las_pointcloud(
            las_file_path=str(las_path),
            use_classification_color=True,
            point_size=2.0,
        )

    # -------------------------------------------------------------------------
    # 9) extract_from_las example
    # -------------------------------------------------------------------------
    if run_extract_from_las:
        print("\n=== Example: extract_from_las ===")
        arr = extract_from_las(
            las_file_path=str(las_path),
            output_np_path=str(npy_path),
        )
        if arr is not None:
            print(f"Extracted numpy shape: {arr.shape}")
            print(f"Numpy data saved to: {npy_path}")

    # -------------------------------------------------------------------------
    # 10) visualize_reflection_intensity example
    # -------------------------------------------------------------------------
    if run_visualize_reflection_intensity:
        print("\n=== Example: visualize_reflection_intensity ===")
        visualize_reflection_intensity(
            las_file_path=str(las_path),
            point_size=2.0,
            percentile_clip=(1.0, 99.0),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

