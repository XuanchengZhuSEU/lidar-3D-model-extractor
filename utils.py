import numpy as np
import laspy
from pathlib import Path
import open3d as o3d
import alphashape as ash
import geopandas as gpd
import pandas as pd
from scipy.spatial import Delaunay

def get_classification_stats(las_file_path):
    """
    Count points per LAS classification code and name.
    
    Args:
        las_file_path (str): Path to the .las file.
    
    Returns:
        dict: Mapping from classification code to a dict with name and count.
              Format: {code: {'name': 'class name', 'count': count}}
    """
    # LAS 1.2 standard classification code -> name
    classification_names = {
        0: 'Never Classified',
        1: 'Unassigned',
        2: 'Ground',
        3: 'Low Vegetation',
        4: 'Medium Vegetation',
        5: 'High Vegetation',
        6: 'Building',
        7: 'Low Point/Noise',
        8: 'Reserved',
        9: 'Water',
        10: 'Rail',
        11: 'Road Surface',
        12: 'Reserved',
    }
    
    # Read LAS file
    las = laspy.read(las_file_path)
    
    # Get classification for all points
    classifications = las.classification
    
    # Count occurrences per class
    unique_classes, counts = np.unique(classifications, return_counts=True)
    
    # Build result mapping
    result = {}
    for class_code, count in zip(unique_classes, counts):
        # Use a fallback name for non-standard/custom codes
        name = classification_names.get(class_code, 'User Defined/Reserved')
        result[int(class_code)] = {
            'name': name,
            'count': int(count)
        }
    
    return result

def merge_las_files(las_file_list, output_path):
    '''
    merge multiple las files into one
    require they share the same xyz coordinate system
    '''
    output_path = Path(output_path)
    
    # Check if the list is empty
    if not las_file_list:
        print("No files to merge.")
        return

    # Process the first file to create the initial merged file
    first_file = laspy.read(las_file_list[0])
    # Create a new LasData object with the first file's header
    las_merged = laspy.LasData(first_file.header)
    las_merged.points = first_file.points.copy()
    las_merged.write(output_path)

    # Loop through the rest of the files and append their points
    for las_file in las_file_list[1:]:
        current_las = laspy.read(las_file)
        with laspy.open(output_path, mode="a") as dst:
            dst.append_points(current_las.points)

    print(f"Merged files into {output_path}")

def las_to_shp(las_file_path, output_shp_path, 
               epsilon=2, min_cluster_size=100, 
               alpha_shape_alpha=0.5, ground_search_k=50,
               crs='EPSG:26910', building_class=6, ground_class=2,
               verbose=True):
    """
    Extract buildings from a LAS file and export to a Shapefile.
    
    Workflow:
    1. Read LAS and extract building/ground points
    2. Unsupervised segmentation with DBSCAN
    3. Build 2D polygons per segment and compute attributes
    4. Assemble a GeoDataFrame with all buildings
    5. Export to Shapefile
    
    Args:
        las_file_path (str): Path to the LAS file
        output_shp_path (str): Output Shapefile path
        epsilon (float): DBSCAN eps, default 2
        min_cluster_size (int): DBSCAN min_points, default 100
        alpha_shape_alpha (float): Alpha-shape alpha, default 0.5
        ground_search_k (int): KNN count for nearest ground points, default 50
        crs (str): CRS string, default 'EPSG:26910'
        building_class (int): Building classification code, default 6
        ground_class (int): Ground classification code, default 2
        verbose (bool): Whether to print progress, default True
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with extracted buildings
    """
    if verbose:
        print(f"Reading LAS file: {las_file_path}")
    
    # Read LAS
    try:
        las = laspy.read(las_file_path)
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        return None
    
    if verbose:
        print(f"Total points: {len(las.points)}")
        print(f"Classification codes: {np.unique(las.classification)}")
    
    # 3.1 Extract building points
    if verbose:
        print("Extracting building points...")
    pts_mask = las.classification == building_class
    if np.sum(pts_mask) == 0:
        print(f"Warning: no building points found for class code {building_class}")
        return None
    
    xyz_t = np.vstack((las.x[pts_mask], las.y[pts_mask], las.z[pts_mask]))
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz_t.transpose())
    pcd_center = pcd_o3d.get_center()
    pcd_o3d.translate(-pcd_center)
    
    # Extract ground points
    if verbose:
        print("Extracting ground points...")
    pts_mask_ground = las.classification == ground_class
    if np.sum(pts_mask_ground) == 0:
        print(f"Warning: no ground points found for class code {ground_class}")
        return None
    
    xyz_ground = np.vstack((las.x[pts_mask_ground], las.y[pts_mask_ground], las.z[pts_mask_ground]))
    ground_pts = o3d.geometry.PointCloud()
    ground_pts.points = o3d.utility.Vector3dVector(xyz_ground.transpose())
    ground_pts.translate(-pcd_center)
    
    # Compute average NN distance between building points
    if verbose:
        nn_distance = np.mean(pcd_o3d.compute_nearest_neighbor_distance())
        print(f"Average NN distance between building points: {nn_distance:.4f}")
    
    # 4. Unsupervised segmentation (DBSCAN)
    if verbose:
        print(f"Running DBSCAN (eps={epsilon}, min_points={min_cluster_size})...")
    labels = np.array(pcd_o3d.cluster_dbscan(eps=epsilon, min_points=min_cluster_size, print_progress=verbose))
    max_label = labels.max()
    
    if max_label < 0:
        print("Warning: no clusters found")
        return None
    
    if verbose:
        print(f"Found {max_label + 1} clusters")
    
    # Initialize GeoDataFrame
    buildings_gdf = gpd.GeoDataFrame(
        columns=['geometry', 'id', 'height', 'area', 'perimeter', 
                 'local_cx', 'local_cy', 'local_cz', 
                 'transl_x', 'transl_y', 'transl_z', 'pts_number'], 
        geometry='geometry', 
        crs=crs
    )
    
    # Process each cluster
    if verbose:
        print("Processing building clusters...")
    
    for sel in range(max_label + 1):
        segment_indices = np.where(labels == sel)[0]
        if len(segment_indices) == 0:
            continue
        
        segment = pcd_o3d.select_by_index(segment_indices)
        
        # Extract 2D polygon
        points_2D = np.asarray(segment.points)[:, :2]
        try:
            building_vector = ash.alphashape(points_2D, alpha=alpha_shape_alpha)
        except Exception as e:
            if verbose:
                print(f"Warning: cluster {sel} failed to generate alphashape: {e}")
            continue
        
        # Compute building height
        query_point = segment.get_center()
        query_point[2] = segment.get_min_bound()[2]
        pcd_tree = o3d.geometry.KDTreeFlann(ground_pts)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(query_point, ground_search_k)
        sample = ground_pts.select_by_index(idx, invert=False)
        ground_zero = sample.get_center()[2]
        height = segment.get_max_bound()[2] - ground_zero
        
        # Create a single-building GeoDataFrame
        building_gdf = gpd.GeoDataFrame(geometry=[building_vector], crs=crs)
        building_gdf['id'] = sel
        building_gdf['height'] = height
        building_gdf['area'] = building_vector.area
        building_gdf['perimeter'] = building_vector.length
        building_gdf['local_cx'] = building_vector.centroid.x
        building_gdf['local_cy'] = building_vector.centroid.y
        building_gdf['local_cz'] = ground_zero
        building_gdf['transl_x'] = pcd_center[0]
        building_gdf['transl_y'] = pcd_center[1]
        building_gdf['transl_z'] = pcd_center[2]
        building_gdf['pts_number'] = len(segment.points)
        
        # Append to overall GeoDataFrame
        buildings_gdf = pd.concat([buildings_gdf, building_gdf], ignore_index=True)
    
    # Export to Shapefile
    if len(buildings_gdf) > 0:
        output_path = Path(output_shp_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"Saving Shapefile to: {output_shp_path}")
        buildings_gdf.to_file(output_shp_path, engine="fiona")
        
        if verbose:
            print(f"Successfully processed {len(buildings_gdf)} buildings")
            print(f"Shapefile saved: {output_shp_path}")
    else:
        print("Warning: no buildings generated; Shapefile was not created")
    
    return buildings_gdf

def visualize_shp_3d(
    shp_path: str,
    use_attribute_height: bool = True,
    default_height: float = 10.0,
    height_scale: float = 1.0,
    random_color: bool = True,
    base_on_attributes: bool = True):
    """
    Visualize a Shapefile exported by `las_to_shp` in 3D.

    The function extrudes 2D polygons along Z into prisms and renders them in Open3D.

    Args:
        shp_path (str): Path to the Shapefile exported by `las_to_shp`
        use_attribute_height (bool): Use the `height` field as extrusion height
        default_height (float): Default height when `height` is missing
        height_scale (float): Height scale factor
        random_color (bool): Random color per building; otherwise uniform gray
        base_on_attributes (bool): Use `local_cz + transl_z` as base Z; if False, extrude from 0
    """
    # Late-import shapely types to avoid import errors in minimal environments
    from shapely.geometry import Polygon, MultiPolygon
    import random as _random

    try:
        gdf = gpd.read_file(shp_path, engine="fiona")
    except Exception as e:
        # Common issue: missing .shx index file (pyogrio/GDAL raises DataSourceError)
        msg = str(e)
        if ("SHAPE_RESTORE_SHX" in msg) or (".shx" in msg.lower() and "Unable to open" in msg):
            import os
            os.environ["SHAPE_RESTORE_SHX"] = "YES"
            try:
                gdf = gpd.read_file(shp_path, engine="fiona")
                print("Note: detected missing .shx; enabled SHAPE_RESTORE_SHX=YES and read successfully.")
            except Exception:
                # In some environments, switching to Fiona is more compatible
                gdf = gpd.read_file(shp_path, engine="fiona")
                print("Note: switched to Fiona engine to read the Shapefile.")
        else:
            raise
    if gdf.empty:
        print("No features found in the Shapefile; nothing to visualize.")
        return

    meshes = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # Handle Polygon and MultiPolygon uniformly
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            # Skip non-area geometries
            continue

        for poly in polys:
            coords = np.asarray(poly.exterior.coords)
            if coords.shape[0] < 3:
                continue

            # Height
            if use_attribute_height and ("height" in row) and (row["height"] is not None):
                h = float(row["height"]) * float(height_scale)
            else:
                h = float(default_height) * float(height_scale)

            # Base Z
            if base_on_attributes and ("local_cz" in row) and ("transl_z" in row):
                try:
                    base_z = float(row["local_cz"]) + float(row["transl_z"])
                except Exception:
                    base_z = 0.0
            else:
                base_z = 0.0

            top_z = base_z + h

            # Get 2D coords (drop duplicated last point if present)
            coords_2d = coords[:-1] if coords.shape[0] > 1 and np.allclose(coords[0], coords[-1]) else coords
            n = coords_2d.shape[0]
            
            if n < 3:
                continue
            
            # Top/bottom contour points
            base = np.hstack((coords_2d, np.full((n, 1), base_z)))
            top = np.hstack((coords_2d, np.full((n, 1), top_z)))
            
            # Vertex order: 0..n-1 bottom ring; n..2n-1 top ring
            points = np.vstack((base, top))

            triangles = []

            # Side walls: split each quad into two triangles.
            # Ensure CCW order when viewed from outside (normals point outward).
            for i in range(n):
                next_i = (i + 1) % n
                # Standard CCW order from outside
                triangles.append([i, next_i, i + n])
                triangles.append([next_i, next_i + n, i + n])

            # Use Delaunay triangulation for top and bottom faces
            try:
                # Delaunay triangulation in 2D
                tri = Delaunay(coords_2d)
                
                # Top face: CCW when viewed from +Z; normals point upward
                for simplex in tri.simplices:
                    # Top: index + n
                    triangles.append([simplex[0] + n, simplex[1] + n, simplex[2] + n])
                
                # Bottom face: CCW when viewed from -Z; normals point downward
                # Reverse winding compared to the top.
                for simplex in tri.simplices:
                    # Bottom: reverse order
                    triangles.append([simplex[0], simplex[2], simplex[1]])
                    
            except Exception as e:
                # Fallback: triangle fan when triangulation fails
                print(f"Warning: triangulation failed; using triangle fan fallback: {e}")
                
                # Compute centroid
                centroid_x = float(poly.centroid.x)
                centroid_y = float(poly.centroid.y)
                bottom_center = np.array([[centroid_x, centroid_y, base_z]])
                top_center = np.array([[centroid_x, centroid_y, top_z]])
                
                # Add center points to vertex list
                points = np.vstack((points, bottom_center, top_center))
                bottom_center_idx = 2 * n
                top_center_idx = 2 * n + 1
                
                # Top triangle fan: CCW from +Z
                for i in range(n):
                    next_i = (i + 1) % n
                    triangles.append([top_center_idx, n + i, n + next_i])
                
                # Bottom triangle fan: CCW from -Z (reverse winding)
                for i in range(n):
                    next_i = (i + 1) % n
                    triangles.append([bottom_center_idx, next_i, i])

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=int))
            
            # Clean mesh: remove degenerate/duplicate primitives
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            # Compute normals
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            # Validate/fix normal orientation to point outward
            vertices_array = np.asarray(mesh.vertices)
            triangles_array = np.asarray(mesh.triangles)
            triangle_normals = np.asarray(mesh.triangle_normals)
            
            # Building center (for normal direction check)
            building_center = np.mean(vertices_array, axis=0)
            
            # Triangle centroids
            triangle_centers = np.mean(vertices_array[triangles_array], axis=1)
            
            # Vectors from building center to triangle centers
            to_center_vectors = triangle_centers - building_center
            
            # If normal points toward the building center, it's likely inward
            dot_products = np.sum(triangle_normals * to_center_vectors, axis=1)
            
            # Separate side vs top/bottom triangles
            side_triangle_count = 2 * n
            top_bottom_triangle_count = len(triangles_array) - side_triangle_count
            
            if top_bottom_triangle_count > 0:
                # Side triangles
                side_dot = dot_products[:side_triangle_count]
                side_inward = np.sum(side_dot > 0)
                
                # Top and bottom triangles
                top_bottom_dot = dot_products[side_triangle_count:]
                top_bottom_inward = np.sum(top_bottom_dot > 0)
                
                # Flip side faces if most are inward
                if side_inward > len(side_dot) / 2:
                    fixed_triangles = np.asarray(mesh.triangles).copy()
                    fixed_triangles[:side_triangle_count] = np.flip(
                        fixed_triangles[:side_triangle_count], axis=1
                    )
                    mesh.triangles = o3d.utility.Vector3iVector(fixed_triangles)
                
                # Flip top/bottom faces if most are inward
                if top_bottom_inward > len(top_bottom_dot) / 2:
                    fixed_triangles = np.asarray(mesh.triangles).copy()
                    fixed_triangles[side_triangle_count:] = np.flip(
                        fixed_triangles[side_triangle_count:], axis=1
                    )
                    mesh.triangles = o3d.utility.Vector3iVector(fixed_triangles)
                
                # Recompute normals
                mesh.compute_vertex_normals()
                mesh.compute_triangle_normals()

            if random_color:
                mesh.paint_uniform_color(
                    [
                        _random.randint(0, 255) / 255.0,
                        _random.randint(0, 255) / 255.0,
                        _random.randint(0, 255) / 255.0,
                    ]
                )
            else:
                mesh.paint_uniform_color([0.6, 0.6, 0.6])

            meshes.append(mesh)

    if not meshes:
        print("No 3D building meshes were generated; check whether the Shapefile contains polygons.")
        return

    print(f"Visualizing {len(meshes)} building 3D meshes.")
    o3d.visualization.draw_geometries(meshes)

def shp_to_blender(
    shp_path: str,
    output_path: str,
    format: str = "obj",
    use_attribute_height: bool = True,
    default_height: float = 10.0,
    height_scale: float = 1.0,
    base_on_attributes: bool = True,
    merge_buildings: bool = True,
    verbose: bool = True):
    """
    Convert a Shapefile exported by `las_to_shp` into a Blender-friendly 3D model format (OBJ or PLY).
    
    The function reads the Shapefile, extrudes each building polygon into a 3D mesh,
    and exports it in a Blender-supported format.
    
    Args:
        shp_path (str): Input Shapefile path (exported by `las_to_shp`)
        output_path (str): Output file path (.obj or .ply)
        format (str): Output format, 'obj' or 'ply' (default 'obj')
        use_attribute_height (bool): Use `height` field as extrusion height
        default_height (float): Default height when `height` is missing
        height_scale (float): Height scale factor
        base_on_attributes (bool): Use `local_cz + transl_z` as base Z
        merge_buildings (bool): Merge all buildings into one file; otherwise export per-building
        verbose (bool): Whether to print progress
    
    Returns:
        list: If merge_buildings=False, returns a list of exported paths; otherwise a single path
    """
    from shapely.geometry import Polygon, MultiPolygon
    import random as _random
    
    # Validate format
    format = format.lower()
    if format not in ['obj', 'ply']:
        raise ValueError(f"Unsupported format: {format}. Only 'obj' or 'ply' are supported.")
    
    # Read Shapefile
    if verbose:
        print(f"Reading Shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path, engine="fiona")
    if gdf.empty:
        print("No features found in the Shapefile; cannot convert.")
        return None if merge_buildings else []
    
    if verbose:
        print(f"Found {len(gdf)} buildings")
    
    # Build meshes for all buildings
    meshes = []
    building_ids = []
    building_colors = []  # Store per-building colors
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        
        # Handle Polygon and MultiPolygon uniformly
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            continue
        
        for poly in polys:
            coords = np.asarray(poly.exterior.coords)
            if coords.shape[0] < 3:
                continue
            
            # Height
            if use_attribute_height and ("height" in row) and (row["height"] is not None):
                h = float(row["height"]) * float(height_scale)
            else:
                h = float(default_height) * float(height_scale)
            
            # Base Z
            if base_on_attributes and ("local_cz" in row) and ("transl_z" in row):
                try:
                    base_z = float(row["local_cz"]) + float(row["transl_z"])
                except Exception:
                    base_z = 0.0
            else:
                base_z = 0.0
            
            top_z = base_z + h
            
            # Get 2D coords (drop duplicated last point)
            coords_2d = coords[:-1] if coords.shape[0] > 1 and np.allclose(coords[0], coords[-1]) else coords
            n = coords_2d.shape[0]
            
            if n < 3:
                continue
            
            # Top/bottom contour points
            base = np.hstack((coords_2d, np.full((n, 1), base_z)))
            top = np.hstack((coords_2d, np.full((n, 1), top_z)))
            points = np.vstack((base, top))
            
            triangles = []
            
            # Side walls
            for i in range(n):
                next_i = (i + 1) % n
                triangles.append([i, next_i, i + n])
                triangles.append([next_i, next_i + n, i + n])
            
            # Use Delaunay triangulation for top and bottom faces
            try:
                tri = Delaunay(coords_2d)
                
                # Top: CCW when viewed from above
                for simplex in tri.simplices:
                    triangles.append([simplex[0] + n, simplex[1] + n, simplex[2] + n])
                
                # Bottom: CCW when viewed from below (reverse winding)
                for simplex in tri.simplices:
                    triangles.append([simplex[0], simplex[2], simplex[1]])
                    
            except Exception as e:
                # Fallback to triangle fan
                if verbose:
                    print(f"Warning: building {idx} triangulation failed; using triangle fan fallback: {e}")
                
                centroid_x = float(poly.centroid.x)
                centroid_y = float(poly.centroid.y)
                bottom_center = np.array([[centroid_x, centroid_y, base_z]])
                top_center = np.array([[centroid_x, centroid_y, top_z]])
                
                points = np.vstack((points, bottom_center, top_center))
                bottom_center_idx = 2 * n
                top_center_idx = 2 * n + 1
                
                # Top triangle fan
                for i in range(n):
                    next_i = (i + 1) % n
                    triangles.append([top_center_idx, n + i, n + next_i])
                
                # Bottom triangle fan
                for i in range(n):
                    next_i = (i + 1) % n
                    triangles.append([bottom_center_idx, next_i, i])
            
            # Create mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=int))
            
            # Clean mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            # Compute normals
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            # Validate and fix normal orientation
            vertices_array = np.asarray(mesh.vertices)
            triangles_array = np.asarray(mesh.triangles)
            triangle_normals = np.asarray(mesh.triangle_normals)
            
            building_center = np.mean(vertices_array, axis=0)
            triangle_centers = np.mean(vertices_array[triangles_array], axis=1)
            to_center_vectors = triangle_centers - building_center
            dot_products = np.sum(triangle_normals * to_center_vectors, axis=1)
            
            side_triangle_count = 2 * n
            top_bottom_triangle_count = len(triangles_array) - side_triangle_count
            
            if top_bottom_triangle_count > 0:
                side_dot = dot_products[:side_triangle_count]
                top_bottom_dot = dot_products[side_triangle_count:]
                
                if np.sum(side_dot > 0) > len(side_dot) / 2:
                    fixed_triangles = np.asarray(mesh.triangles).copy()
                    fixed_triangles[:side_triangle_count] = np.flip(
                        fixed_triangles[:side_triangle_count], axis=1
                    )
                    mesh.triangles = o3d.utility.Vector3iVector(fixed_triangles)
                
                if np.sum(top_bottom_dot > 0) > len(top_bottom_dot) / 2:
                    fixed_triangles = np.asarray(mesh.triangles).copy()
                    fixed_triangles[side_triangle_count:] = np.flip(
                        fixed_triangles[side_triangle_count:], axis=1
                    )
                    mesh.triangles = o3d.utility.Vector3iVector(fixed_triangles)
                
                mesh.compute_vertex_normals()
                mesh.compute_triangle_normals()
            
            # Set color (random or attribute-based)
            if "id" in row:
                # Deterministic color derived from building ID
                _random.seed(int(row["id"]))
            building_color = [
                _random.randint(0, 255) / 255.0,
                _random.randint(0, 255) / 255.0,
                _random.randint(0, 255) / 255.0,
            ]
            mesh.paint_uniform_color(building_color)
            
            # Apply rotation so that X rotation is 0° in Blender
            # Convert from GIS coordinates (Y-up) to Blender coordinates (Z-up)
            # Rotate around X by -90° to convert Y-up -> Z-up
            R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
            mesh.rotate(R, center=(0, 0, 0))
            # Recompute normals (rotation changes normal directions)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            meshes.append(mesh)
            building_ids.append(idx)
            building_colors.append(building_color)  # Store color
    
    if not meshes:
        print("No 3D building meshes were generated.")
        return None if merge_buildings else []
    
    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if merge_buildings:
        # Merge all buildings into one file, keeping each building as a separate object
        if verbose:
            print(f"Merging {len(meshes)} buildings and exporting to: {output_path} (each building as a separate object)")
        
        if format == "obj":
            # Write an OBJ manually to add per-building object groups and materials
            _export_obj_with_groups(meshes, building_ids, building_colors, output_path, verbose)
        else:  # ply
            # PLY does not support object groups; export multiple files or use comments
            if verbose:
                print("Note: PLY does not support object groups; exporting multiple files")
            base_name = output_path.stem
            base_dir = output_path.parent
            output_files = []
            
            for i, mesh in enumerate(meshes):
                building_id = building_ids[i] if i < len(building_ids) else i
                file_name = f"{base_name}_building_{building_id}.ply"
                file_path = base_dir / file_name
                
                o3d.io.write_triangle_mesh(
                    str(file_path),
                    mesh,
                    write_ascii=True,
                    write_vertex_normals=True,
                    write_vertex_colors=True,
                )
                output_files.append(str(file_path))
            
            if verbose:
                print(f"Successfully exported {len(output_files)} PLY files to: {base_dir}")
            return output_files
        
        if verbose:
            print(f"Successfully exported to: {output_path}")
        return str(output_path)
    
    else:
        # Export each building separately
        output_files = []
        base_name = output_path.stem
        base_dir = output_path.parent
        
        for i, mesh in enumerate(meshes):
            building_id = building_ids[i] if i < len(building_ids) else i
            building_color = building_colors[i] if i < len(building_colors) else [0.6, 0.6, 0.6]
            file_name = f"{base_name}_building_{building_id}.{format}"
            file_path = base_dir / file_name
            
            if format == "obj":
                # For OBJ, create a material-backed file manually
                mtl_path = file_path.with_suffix('.mtl')
                mtl_name = mtl_path.name
                material_name = f"Building_{building_id}_Mat"
                
                # Create MTL file
                with open(mtl_path, 'w', encoding='utf-8') as mtl_file:
                    mtl_file.write("# MTL file generated from shp_to_blender\n")
                    mtl_file.write(f"newmtl {material_name}\n")
                    mtl_file.write("Ka 1.000 1.000 1.000\n")
                    mtl_file.write(f"Kd {building_color[0]:.6f} {building_color[1]:.6f} {building_color[2]:.6f}\n")
                    mtl_file.write("Ks 0.500 0.500 0.500\n")
                    mtl_file.write("Ns 32.000\n")
                    mtl_file.write("d 1.0000\n")
                    mtl_file.write("illum 2\n")
                
                # Export with Open3D, then inject material references
                temp_obj = file_path.with_suffix('.tmp.obj')
                o3d.io.write_triangle_mesh(
                    str(temp_obj),
                    mesh,
                    write_ascii=True,
                    write_vertex_normals=True,
                    write_vertex_colors=True,
                )
                
                # Read temp file and add material references
                with open(temp_obj, 'r', encoding='utf-8') as temp_file:
                    content = temp_file.read()
                
                # Add material library and material usage at file header
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# OBJ file generated from shp_to_blender\n")
                    f.write(f"mtllib {mtl_name}\n")
                    f.write(f"usemtl {material_name}\n\n")
                    f.write(content)
                
                # Delete temporary file
                temp_obj.unlink()
                
            else:  # ply
                o3d.io.write_triangle_mesh(
                    str(file_path),
                    mesh,
                    write_ascii=True,
                    write_vertex_normals=True,
                    write_vertex_colors=True,
                )
            
            output_files.append(str(file_path))
        
        if verbose:
            print(f"Successfully exported {len(output_files)} files to: {base_dir}")
        return output_files


def _export_obj_with_groups(meshes, building_ids, building_colors, output_path, verbose):
    """
    Export multiple meshes into a single OBJ file. Each mesh becomes its own object group,
    with colors defined via an accompanying MTL file.

    Args:
        meshes: List of Open3D TriangleMesh
        building_ids: List of building IDs
        building_colors: List of RGB colors in [0, 1]
        output_path: Output OBJ file path
        verbose: Whether to print verbose logs
    """
    output_path = Path(output_path)
    mtl_path = output_path.with_suffix('.mtl')
    mtl_name = mtl_path.name
    
    # Create MTL material file
    with open(mtl_path, 'w', encoding='utf-8') as mtl_file:
        mtl_file.write("# MTL file generated from shp_to_blender\n")
        mtl_file.write("# Material definitions for buildings\n\n")
        
        for i, color in enumerate(building_colors):
            building_id = building_ids[i] if i < len(building_ids) else i
            material_name = f"Building_{building_id}_Mat"
            
            # Write material definition
            mtl_file.write(f"newmtl {material_name}\n")
            mtl_file.write("Ka 1.000 1.000 1.000\n")  # Ambient color (white)
            mtl_file.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")  # Diffuse color (building color)
            mtl_file.write(f"Ks 0.500 0.500 0.500\n")  # Specular color
            mtl_file.write("Ns 32.000\n")  # Specular exponent
            mtl_file.write("d 1.0000\n")  # Opacity
            mtl_file.write("illum 2\n")  # Illumination model (Phong)
            mtl_file.write("\n")
    
    # Create OBJ file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# OBJ file generated from shp_to_blender\n")
        f.write("# Each building is separated as an object group with material\n")
        f.write(f"mtllib {mtl_name}\n\n")
        
        vertex_offset = 0  # Vertex index offset
        
        for i, mesh in enumerate(meshes):
            building_id = building_ids[i] if i < len(building_ids) else i
            material_name = f"Building_{building_id}_Mat"
            
            # Fetch mesh data
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            vertex_normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
            
            # Write object group header and material
            f.write(f"o Building_{building_id}\n")
            f.write(f"g Building_{building_id}\n")
            f.write(f"usemtl {material_name}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write vertex normals
            if vertex_normals is not None and len(vertex_normals) > 0:
                for vn in vertex_normals:
                    f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
            
            # Write faces (with vertex index offset)
            for tri in triangles:
                # OBJ indices start at 1
                v1 = tri[0] + vertex_offset + 1
                v2 = tri[1] + vertex_offset + 1
                v3 = tri[2] + vertex_offset + 1
                
                if vertex_normals is not None and len(vertex_normals) > 0:
                    # If normals exist, use v//vn format
                    n1 = tri[0] + vertex_offset + 1
                    n2 = tri[1] + vertex_offset + 1
                    n3 = tri[2] + vertex_offset + 1
                    f.write(f"f {v1}//{n1} {v2}//{n2} {v3}//{n3}\n")
                else:
                    # Vertices only
                    f.write(f"f {v1} {v2} {v3}\n")
            
            # Update vertex offset
            vertex_offset += len(vertices)
            
            # Add blank line as separator
            f.write("\n")
        
        if verbose:
            print(f"Wrote {len(meshes)} building objects into the OBJ file")
            print(f"Created material file: {mtl_path}")


def visualize_las_pointcloud(las_file_path, use_classification_color=True, point_size=1.0):
    """
    Visualize a LAS point cloud in 3D with Open3D.

    The function reads a LAS file, converts it to an Open3D point cloud,
    and colors points (prefer LAS RGB if present; otherwise by classification).

    Args:
        las_file_path (str): LAS file path
        use_classification_color (bool): Color by classification code if RGB is not available
        point_size (float): Point size for rendering

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud object
    """
    if not Path(las_file_path).exists():
        raise FileNotFoundError(f"LAS file does not exist: {las_file_path}")
    
    print(f"Reading LAS file: {las_file_path}")
    
    # Read LAS
    try:
        las = laspy.read(las_file_path)
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        return None
    
    print(f"Total points: {len(las.points)}")
    print(f"Classification codes: {np.unique(las.classification)}")
    
    # Extract XYZ
    xyz_t = np.vstack((las.x, las.y, las.z))
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz_t.transpose())
    
    # Prefer per-point RGB from LAS if available
    has_rgb = hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue")
    if has_rgb:
        try:
            red = np.asarray(las.red)
            green = np.asarray(las.green)
            blue = np.asarray(las.blue)

            # Typical LAS RGB range is 0-65535 or 0-255; Open3D expects float in [0, 1]
            max_rgb = float(np.max([np.max(red), np.max(green), np.max(blue)])) if len(red) > 0 else 0.0
            if max_rgb <= 0:
                raise ValueError("RGB arrays are empty or all zeros")

            if max_rgb > 255.0:
                rgb01 = np.stack([red, green, blue], axis=1).astype(np.float64) / 65535.0
            else:
                rgb01 = np.stack([red, green, blue], axis=1).astype(np.float64) / 255.0

            rgb01 = np.clip(rgb01, 0.0, 1.0)
            pcd_o3d.colors = o3d.utility.Vector3dVector(rgb01)
            print("Colored using per-point LAS RGB")
        except Exception as e:
            print(f"Warning: failed to read/process RGB; falling back to classification/uniform coloring: {e}")
            has_rgb = False

    # If no RGB, color by classification or uniform gray
    if not has_rgb:
        if use_classification_color:
            classifications = np.asarray(las.classification)
            unique_classes = np.unique(classifications)
            
            # LAS 1.2 standard classification color mapping
            classification_colors = {
                0: [0.5, 0.5, 0.5],      # Never Classified - gray
                1: [0.8, 0.8, 0.8],      # Unassigned - light gray
                2: [0.4, 0.3, 0.2],      # Ground - brown
                3: [0.2, 0.6, 0.2],      # Low Vegetation - light green
                4: [0.1, 0.5, 0.1],      # Medium Vegetation - green
                5: [0.0, 0.4, 0.0],      # High Vegetation - dark green
                6: [0.8, 0.2, 0.2],      # Building - red
                7: [0.9, 0.9, 0.1],      # Low Point/Noise - yellow
                8: [0.5, 0.5, 0.5],      # Reserved - gray
                9: [0.2, 0.4, 0.8],      # Water - blue
                10: [0.6, 0.3, 0.1],     # Rail - brown
                11: [0.4, 0.4, 0.4],     # Road Surface - dark gray
                12: [0.5, 0.5, 0.5],     # Reserved - gray
            }
            
            # Assign colors per point
            colors = np.zeros((len(las.points), 3))
            for class_code in unique_classes:
                mask = classifications == class_code
                color = classification_colors.get(
                    int(class_code),
                    [0.5, 0.5, 0.5]  # Default gray (user-defined/reserved)
                )
                colors[mask] = color
            
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
            print(f"Colored by classification codes ({len(unique_classes)} classes)")
        else:
            # Uniform color
            pcd_o3d.paint_uniform_color([0.5, 0.5, 0.5])
            print("Using uniform gray color")
    
    # Center the cloud for visualization
    pcd_center = pcd_o3d.get_center()
    pcd_o3d.translate(-pcd_center)
    
    print(f"Point cloud center: {pcd_center}")
    print(f"Point cloud bounds: {pcd_o3d.get_axis_aligned_bounding_box()}")
    print("Rendering point cloud...")
    
    # Visualize (ensure point_size takes effect)
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="LAS Point Cloud Visualization",
        width=1920,
        height=1080,
        visible=True,
    )
    vis.add_geometry(pcd_o3d)
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = float(point_size)
    vis.run()
    vis.destroy_window()
    
    return pcd_o3d


def visualize_reflection_intensity(las_file_path, point_size=1.0, percentile_clip=(1.0, 99.0)):
    """
    Visualize a LAS point cloud with per-point colors mapped from reflection intensity.
    Higher intensity -> deeper red; lower intensity -> lighter (near-white / pale pink).

    Args:
        las_file_path (str): LAS file path
        point_size (float): Point size for rendering
        percentile_clip (tuple): Percentile clipping before normalization, default (1, 99)
            to improve contrast; use (0, 100) to use full min/max.

    Returns:
        o3d.geometry.PointCloud or None (on read failure or missing intensity)
    """
    if not Path(las_file_path).exists():
        raise FileNotFoundError(f"LAS file does not exist: {las_file_path}")

    print(f"Reading LAS (color by intensity): {las_file_path}")

    try:
        las = laspy.read(las_file_path)
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        return None

    if not hasattr(las, "intensity"):
        print("Error: this LAS file does not contain an intensity field.")
        return None

    try:
        intensity = np.asarray(las.intensity, dtype=np.float64)
    except Exception as e:
        print(f"Error: unable to read intensity: {e}")
        return None

    n = len(las.points)
    if intensity.size != n:
        print(f"Error: intensity length ({intensity.size}) does not match point count ({n}).")
        return None

    lo_p, hi_p = float(percentile_clip[0]), float(percentile_clip[1])
    lo_p = max(0.0, min(100.0, lo_p))
    hi_p = max(0.0, min(100.0, hi_p))
    if hi_p <= lo_p:
        hi_p = min(100.0, lo_p + 1e-6)

    i_min = float(np.percentile(intensity, lo_p))
    i_max = float(np.percentile(intensity, hi_p))
    if i_max <= i_min:
        i_max = i_min + 1e-9

    t = (intensity - i_min) / (i_max - i_min)
    t = np.clip(t, 0.0, 1.0)
    # Low intensity -> t≈0 -> light (near-white / pale pink);
    # high intensity -> t≈1 -> deep red.
    # Keep hue noticeable by reducing G/B faster than R.
    r = 1.0 + t * (0.52 - 1.0)  # 1.0 -> 0.52
    g = 0.96 * (1.0 - t)
    b = 0.96 * (1.0 - t)
    colors = np.stack([r, g, b], axis=1)
    colors = np.clip(colors, 0.0, 1.0)

    xyz_t = np.vstack((las.x, las.y, las.z))
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz_t.transpose())
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

    print(f"Total points: {n}")
    print(
        f"Intensity range (raw): min={np.min(intensity):.3f}, max={np.max(intensity):.3f}; "
        f"Mapping percentiles [{lo_p}, {hi_p}]%: [{i_min:.3f}, {i_max:.3f}]"
    )

    pcd_center = pcd_o3d.get_center()
    pcd_o3d.translate(-pcd_center)
    print(f"Point cloud center: {pcd_center}")
    print("Rendering point cloud (deep red = strong intensity, pale = weak)...")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="LAS Intensity (red = strong)",
        width=1920,
        height=1080,
        visible=True,
    )
    vis.add_geometry(pcd_o3d)
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = float(point_size)
    vis.run()
    vis.destroy_window()

    return pcd_o3d


def extract_from_las(las_file_path, output_np_path=None):
    """
    Extract per-point x, y, z coordinates, classification, RGB color, and intensity from a LAS file,
    and return them as a NumPy array.

    Args:
        las_file_path (str): LAS file path
        output_np_path (str, optional): Path to save the NumPy array (.npy). If None, do not save.

    Returns:
        np.ndarray: Array of shape (n_points, 8), each row:
                    [x, y, z, classification, r, g, b, intensity]
                    Returns None on failure.
    """
    if not Path(las_file_path).exists():
        print(f"Error: LAS file does not exist: {las_file_path}")
        return None
    
    try:
        # Read LAS
        las = laspy.read(las_file_path)
        
        # Extract x, y, z
        x = np.asarray(las.x)
        y = np.asarray(las.y)
        z = np.asarray(las.z)
        
        # Extract classification
        classification = np.asarray(las.classification)

        # Extract intensity (if present)
        has_intensity = False
        if hasattr(las, "intensity"):
            try:
                intensity = np.asarray(las.intensity)
                if len(intensity) > 0:
                    has_intensity = True
            except Exception as e:
                print(f"Warning: unable to read intensity: {e}")
                has_intensity = False

        # Extract RGB (if present)
        has_rgb = False
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            try:
                red = np.asarray(las.red)
                green = np.asarray(las.green)
                blue = np.asarray(las.blue)
                # Typical RGB range is 0-65535 or 0-255
                if len(red) > 0 and len(green) > 0 and len(blue) > 0:
                    has_rgb = True
                    # Normalize 16-bit RGB to 8-bit if needed
                    if np.max(red) > 255 or np.max(green) > 255 or np.max(blue) > 255:
                        red = (red / 256).astype(np.uint8)
                        green = (green / 256).astype(np.uint8)
                        blue = (blue / 256).astype(np.uint8)
                    else:
                        red = red.astype(np.uint8)
                        green = green.astype(np.uint8)
                        blue = blue.astype(np.uint8)
            except Exception as e:
                print(f"Warning: unable to read RGB color: {e}")
                has_rgb = False
        
        if has_rgb:
            red_arr = red
            green_arr = green
            blue_arr = blue
            print_rgb_msg = "XYZ + classification + RGB"
        else:
            # If no RGB, fill with zeros
            red_arr = np.zeros_like(classification, dtype=np.uint8)
            green_arr = np.zeros_like(classification, dtype=np.uint8)
            blue_arr = np.zeros_like(classification, dtype=np.uint8)
            print_rgb_msg = "XYZ + classification; RGB not available (filled with 0)"

        if has_intensity:
            intensity_arr = intensity.astype(np.float32)
        else:
            intensity_arr = np.zeros_like(classification, dtype=np.float32)

        # Combine into (n_points, 8): [x, y, z, classification, r, g, b, intensity]
        data_array = np.column_stack(
            (x, y, z, classification, red_arr, green_arr, blue_arr, intensity_arr)
        )
        print(f"Successfully extracted {len(data_array)} points ({print_rgb_msg}, with intensity)")

        print("Data ranges:")
        print(f"  X: [{np.min(x):.2f}, {np.max(x):.2f}]")
        print(f"  Y: [{np.min(y):.2f}, {np.max(y):.2f}]")
        print(f"  Z: [{np.min(z):.2f}, {np.max(z):.2f}]")
        print(f"  Classification codes: {np.unique(classification)}")
        if has_rgb:
            print(f"  RGB range: R[{np.min(red)}, {np.max(red)}], G[{np.min(green)}, {np.max(green)}], B[{np.min(blue)}, {np.max(blue)}]")
        if has_intensity:
            print(f"  Intensity range: [{np.min(intensity_arr):.2f}, {np.max(intensity_arr):.2f}]")
        
        # Save if requested
        if output_np_path is not None:
            output_path = Path(output_np_path)
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Auto-add .npy extension if missing
            if output_path.suffix == '':
                output_path = output_path.with_suffix('.npy')
            
            # Save array
            np.save(str(output_path), data_array)
            print(f"Saved data array to: {output_path}")
            print(f"Array shape: {data_array.shape}")
            print("Array format: [x, y, z, classification, r, g, b, intensity]")
        
        return data_array
        
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        return None



