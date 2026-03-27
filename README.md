# LiDAR Reconstruction 使用说明

本项目围绕 `utils.py` 提供了一组针对 LAS 点云与建筑物矢量化/可视化的工具函数，并在 `test.py` 中给出了可直接运行的示例流程。

---

## 1. 文件说明

- `utils.py`：核心功能函数（统计、合并、LAS->SHP、3D 可视化、导出 Blender、强度可视化、数据提取）
- `test.py`：完整示例入口，通过开关控制每个函数是否执行
- `points_cloud_data/test_lidar.las`：默认示例输入点云
- `results/`：示例运行输出目录（SHP/OBJ/NPY 等）

---

## 2. 环境与依赖

建议在项目虚拟环境 `lidar_env` 中运行。

### 激活虚拟环境（PowerShell）

```powershell
& .\lidar_env\Scripts\Activate.ps1
```

### 运行示例

```powershell
python .\test.py
```

---

## 3. `utils.py` 函数总览

### 3.1 `get_classification_stats(las_file_path)`

统计 LAS 中各 `classification` 的点数。

- 输入：LAS 文件路径
- 输出：字典  
  ` {class_code: {"name": class_name, "count": point_count}} `
- 典型用途：快速了解分类分布（如地面、建筑、植被）

---

### 3.2 `merge_las_files(las_file_list, output_path)`

合并多个 LAS 文件到一个输出 LAS。

- 输入：
  - `las_file_list`：待合并 LAS 路径列表
  - `output_path`：输出 LAS 路径
- 输出：无显式返回（生成文件）
- 注意：要求输入文件坐标系一致

---

### 3.3 `las_to_shp(...)`

从 LAS 中提取建筑物并导出为 Shapefile。

- 关键参数：
  - `epsilon`、`min_cluster_size`：DBSCAN 聚类参数
  - `alpha_shape_alpha`：Alpha Shape 参数（轮廓拟合）
  - `ground_search_k`：估计地面高度时的 KNN 数
  - `building_class`：建筑分类码（默认 `6`）
  - `ground_class`：地面分类码（默认 `2`）
- 输出：
  - 返回 `GeoDataFrame`
  - 同时写出 `output_shp_path`
- 生成字段包括：`id`、`height`、`area`、`perimeter`、`local_cx/cy/cz`、`transl_x/y/z`、`pts_number`

---

### 3.4 `visualize_shp_3d(...)`

将 `las_to_shp` 的 2D 建筑面拉伸为 3D 网格并可视化（Open3D）。

- 关键参数：
  - `use_attribute_height`：是否使用 `height` 字段
  - `default_height`：缺失高度时默认值
  - `height_scale`：高度缩放
  - `random_color`：随机着色或统一灰色
  - `base_on_attributes`：底面是否使用 `local_cz + transl_z`
- 特性：
  - 自动尝试处理缺失 `.shx`（`SHAPE_RESTORE_SHX=YES`）
  - 必要时切换 Fiona 引擎读取 SHP

---

### 3.5 `shp_to_blender(...)`

将建筑 SHP 转为 Blender 可导入的 3D 模型（OBJ/PLY）。

- 关键参数：
  - `format`：`"obj"` 或 `"ply"`
  - `merge_buildings`：合并为单文件或逐建筑导出
  - 其余拉伸参数与 `visualize_shp_3d` 类似
- 输出：
  - `merge_buildings=True`：返回单个输出路径
  - `merge_buildings=False`：返回路径列表
- 说明：
  - OBJ 会附带材质信息（MTL）
  - 含坐标系旋转处理（Y-up -> Z-up）以便 Blender 使用

---

### 3.6 `visualize_las_pointcloud(las_file_path, use_classification_color=True, point_size=1.0)`

直接可视化 LAS 点云。

- 颜色策略：
  1. 优先使用 LAS 自带逐点 RGB
  2. 若无 RGB，则按分类码着色（或统一灰色）
- 参数：
  - `point_size` 控制可视化点大小

---

### 3.7 `visualize_reflection_intensity(las_file_path, point_size=1.0, percentile_clip=(1.0, 99.0))`

按反射强度 `intensity` 着色可视化点云。

- 映射规则：**强度越大，红色越深；强度越小，颜色越浅**
- `percentile_clip`：
  - 默认 `(1, 99)`，提升对比度，减弱极值影响
  - 可设 `(0, 100)` 使用全量最小/最大值

---

### 3.8 `extract_from_las(las_file_path, output_np_path=None)`

提取点属性到 NumPy 数组并可选保存。

- 输出数组形状：`(n_points, 8)`
- 列顺序：`[x, y, z, classification, r, g, b, intensity]`
- RGB 不存在时填充 0；intensity 不存在时填充 0

---

## 4. `test.py` 说明（示例脚本）

`test.py` 的 `main()` 已按流程组织好调用，并通过布尔开关控制每一步是否执行。

### 4.1 默认输入与输出

- 输入：`points_cloud_data/test_lidar.las`
- 输出目录：
  - `results/shp/`
  - `results/blender/`
  - `results/numpy/`

### 4.2 运行开关

在 `test.py` 中可直接修改以下变量：

- `run_get_classification_stats`
- `run_merge_las_files`
- `run_las_to_shp`
- `run_visualize_shp_3d`
- `run_shp_to_blender`
- `run_visualize_las_pointcloud`
- `run_extract_from_las`
- `run_visualize_reflection_intensity`

将对应开关设为 `True` 即执行，设为 `False` 即跳过。

### 4.3 推荐执行顺序

1. `get_classification_stats`（确认分类情况）
2. `las_to_shp`（提取建筑面）
3. `visualize_shp_3d`（检查 3D 拉伸效果）
4. `shp_to_blender`（导出 OBJ/PLY）
5. `extract_from_las`（导出训练/分析数据）
6. 需要时再执行两类点云可视化函数

---

## 5. 快速示例代码

### 5.1 最小工作流：LAS -> SHP -> OBJ

```python
from utils import las_to_shp, shp_to_blender

las_to_shp("points_cloud_data/test_lidar.las", "results/shp/test_lidar.shp")
shp_to_blender("results/shp/test_lidar.shp", "results/blender/test_lidar.obj", format="obj")
```

### 5.2 强度着色可视化

```python
from utils import visualize_reflection_intensity

visualize_reflection_intensity(
    "points_cloud_data/test_lidar.las",
    point_size=2.0,
    percentile_clip=(1.0, 99.0),
)
```


