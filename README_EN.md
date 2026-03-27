# LiDAR Reconstruction Utils Guide

This project provides utility functions in `utils.py` for LAS point-cloud processing, building vectorization, 3D visualization, and export.  
`test.py` includes a runnable end-to-end example pipeline.

---

## 1. File Overview

- `utils.py`: core utility functions (stats, merge, LAS->SHP, visualization, Blender export, intensity rendering, data extraction)
- `test.py`: runnable examples with step switches
- `points_cloud_data/test_lidar.las`: default input sample
- `results/`: output directory for generated SHP/OBJ/NPY, etc.

---

## 2. Environment

Use the project virtual environment `lidar_env`.

### Activate (PowerShell)

```powershell
& .\lidar_env\Scripts\Activate.ps1
```

### Run examples

```powershell
python .\test.py
```

---

## 3. `utils.py` Function Reference

> `_export_obj_with_groups(...)` is an internal helper; you usually do not call it directly.

### 3.1 `get_classification_stats(las_file_path)`

Counts points for each LAS classification code.

- Input: LAS path
- Output: dict  
  `{class_code: {"name": class_name, "count": point_count}}`
- Typical use: quick inspection of class distribution

---

### 3.2 `merge_las_files(las_file_list, output_path)`

Merges multiple LAS files into one output LAS.

- Inputs:
  - `las_file_list`: list of LAS paths
  - `output_path`: output LAS path
- Output: writes merged file
- Note: input files should share the same coordinate system

---

### 3.3 `las_to_shp(...)`

Extracts building footprints from LAS and exports a Shapefile.

- Key parameters:
  - `epsilon`, `min_cluster_size`: DBSCAN parameters
  - `alpha_shape_alpha`: alpha-shape parameter
  - `ground_search_k`: KNN count for ground-height estimation
  - `building_class`: building class code (default `6`)
  - `ground_class`: ground class code (default `2`)
- Output:
  - returns a `GeoDataFrame`
  - writes Shapefile to `output_shp_path`
- Generated fields include: `id`, `height`, `area`, `perimeter`, `local_cx/cy/cz`, `transl_x/y/z`, `pts_number`

---

### 3.4 `visualize_shp_3d(...)`

Extrudes building polygons from SHP into 3D meshes and visualizes them with Open3D.

- Key parameters:
  - `use_attribute_height`
  - `default_height`
  - `height_scale`
  - `random_color`
  - `base_on_attributes`
- Features:
  - tries to recover missing `.shx` automatically (`SHAPE_RESTORE_SHX=YES`)
  - can fallback to Fiona engine when needed

---

### 3.5 `shp_to_blender(...)`

Converts building SHP into Blender-friendly 3D files (OBJ/PLY).

- Key parameters:
  - `format`: `"obj"` or `"ply"`
  - `merge_buildings`: single merged file or per-building outputs
- Return:
  - merged mode: single output path
  - split mode: list of output paths
- Notes:
  - OBJ export includes material (MTL)
  - coordinate conversion (Y-up -> Z-up) is applied for Blender compatibility

---

### 3.6 `visualize_las_pointcloud(las_file_path, use_classification_color=True, point_size=1.0)`

Visualizes a LAS point cloud directly.

- Coloring strategy:
  1. Prefer per-point RGB from LAS
  2. Fallback to classification coloring (or uniform gray)
- `point_size` controls rendered point size

---

### 3.7 `visualize_reflection_intensity(las_file_path, point_size=1.0, percentile_clip=(1.0, 99.0))`

Visualizes point cloud by intensity.

- Mapping: **higher intensity -> deeper red; lower intensity -> lighter color**
- `percentile_clip`:
  - default `(1, 99)` improves contrast and suppresses outliers
  - use `(0, 100)` for full min/max range

---

### 3.8 `extract_from_las(las_file_path, output_np_path=None)`

Extracts point attributes to NumPy.

- Output shape: `(n_points, 8)`
- Column order: `[x, y, z, classification, r, g, b, intensity]`
- Missing RGB/intensity values are zero-filled

---

## 4. `test.py` Guide

`test.py` contains a structured `main()` pipeline with boolean switches.

### 4.1 Default input/output

- Input: `points_cloud_data/test_lidar.las`
- Outputs:
  - `results/shp/`
  - `results/blender/`
  - `results/numpy/`

### 4.2 Execution switches

In `test.py`, enable/disable steps with:

- `run_get_classification_stats`
- `run_merge_las_files`
- `run_las_to_shp`
- `run_visualize_shp_3d`
- `run_shp_to_blender`
- `run_visualize_las_pointcloud`
- `run_extract_from_las`
- `run_visualize_reflection_intensity`

Set `True` to run a step, `False` to skip.

### 4.3 Recommended sequence

1. `get_classification_stats`
2. `las_to_shp`
3. `visualize_shp_3d`
4. `shp_to_blender`
5. `extract_from_las`
6. optional visualization functions

---

## 5. Quick Code Snippets

### 5.1 Minimal pipeline: LAS -> SHP -> OBJ

```python
from utils import las_to_shp, shp_to_blender

las_to_shp("points_cloud_data/test_lidar.las", "results/shp/test_lidar.shp")
shp_to_blender("results/shp/test_lidar.shp", "results/blender/test_lidar.obj", format="obj")
```

### 5.2 Intensity-based visualization

```python
from utils import visualize_reflection_intensity

visualize_reflection_intensity(
    "points_cloud_data/test_lidar.las",
    point_size=2.0,
    percentile_clip=(1.0, 99.0),
)
```

---

## 6. Troubleshooting

- Missing `.shx` when reading SHP  
  - `visualize_shp_3d` already includes recovery logic; still check `.shp/.dbf/.shx` naming and location
- No buildings extracted  
  - verify `building_class` (default `6`) and `ground_class` (default `2`) exist in your LAS
  - tune `epsilon` / `min_cluster_size`
- Weak color contrast in visualization  
  - increase `point_size`
  - adjust `percentile_clip` for intensity rendering

---

If you want, I can also provide a fully bilingual (`Chinese + English`) merged README in one file.
