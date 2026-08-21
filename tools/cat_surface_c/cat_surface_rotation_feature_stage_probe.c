/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出官方初始旋转特征的粗球面几何和 depth-potential 中间结果。 */

#include <bicpl.h>
#include <CAT_Curvature.h>
#include <CAT_DepthPotential.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_Smooth.h>

#include <stdio.h>
#include <stdlib.h>

#define compute_depth_potential cat_diag_compute_depth_potential
#define compute_areas cat_diag_compute_areas
#define local_depth_potential cat_diag_local_depth_potential
static double *cat_diag_compute_areas(int, Point[], int *, int **, int);
static double *cat_diag_local_depth_potential(
    int, Point[], double *, struct csr_matrix *, double *, double, double);
#include "../third_party/CAT-Surface/Lib/CAT_DepthPotential.c"
#undef compute_depth_potential
#undef compute_areas
#undef local_depth_potential

typedef struct {
    object_struct **objects;
    int n_objects;
    polygons_struct *polygons;
} LoadedPolygons;

static LoadedPolygons load_polygons(const char *path)
{
    LoadedPolygons loaded = {0};
    File_formats format;
    if (input_graphics_any_format((char *)path, &format, &loaded.n_objects,
                                  &loaded.objects) != OK ||
        loaded.n_objects != 1 || get_object_type(loaded.objects[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个曲面对象: %s\n", path);
        exit(EXIT_FAILURE);
    }
    loaded.polygons = get_polygons_ptr(loaded.objects[0]);
    return loaded;
}

static void write_stage(const char *path, polygons_struct *polygons,
                        const double *areas, const Vector *normals,
                        const double *mean_curvature, const double *raw_values,
                        const double *values, const struct csr_matrix *matrix)
{
    FILE *output = fopen(path, "wb");
    if (output == NULL) {
        perror("打开特征阶段输出失败");
        exit(EXIT_FAILURE);
    }
    fwrite(&polygons->n_points, sizeof(polygons->n_points), 1, output);
    fwrite(polygons->points, sizeof(*polygons->points), polygons->n_points,
           output);
    fwrite(areas, sizeof(*areas), polygons->n_points, output);
    fwrite(normals, sizeof(*normals), polygons->n_points, output);
    fwrite(mean_curvature, sizeof(*mean_curvature), polygons->n_points, output);
    fwrite(raw_values, sizeof(*raw_values), polygons->n_points, output);
    fwrite(values, sizeof(*values), polygons->n_points, output);
    fwrite(&matrix->nnz, sizeof(matrix->nnz), 1, output);
    fwrite(matrix->ia, sizeof(*matrix->ia), matrix->n + 1, output);
    fwrite(matrix->ja, sizeof(*matrix->ja), matrix->nnz, output);
    fwrite(matrix->A, sizeof(*matrix->A), matrix->nnz, output);
    fclose(output);
}

static void write_mapped_points(const char *stage_path,
                                const polygons_struct *polygons)
{
    char output_path[4096];
    FILE *output;
    int written = snprintf(output_path, sizeof(output_path), "%s.mapped",
                           stage_path);
    if (written < 0 || (size_t)written >= sizeof(output_path))
        return;
    output = fopen(output_path, "wb");
    if (output == NULL)
        return;
    fwrite(polygons->points, sizeof(*polygons->points), polygons->n_points,
           output);
    fclose(output);
}

static void make_feature(const char *surface_path, const char *sphere_path,
                         double heat_fwhm, const char *output_path)
{
    LoadedPolygons surface = load_polygons(surface_path);
    LoadedPolygons sphere = load_polygons(sphere_path);
    polygons_struct mapped = {0};
    int *n_neighbours = NULL;
    int **neighbours = NULL;
    double *areas;
    Vector *normals;
    double *mean_curvature;
    struct csr_matrix laplacian;
    double *raw_values;
    double *values;
    const int n_triangles = 81920;

    translate_to_center_of_mass(sphere.polygons);
    for (int i = 0; i < sphere.polygons->n_points; ++i)
        set_vector_length(&sphere.polygons->points[i], 1.0);
    resample_spherical_surface(surface.polygons, sphere.polygons, &mapped,
                               NULL, NULL, n_triangles);
    write_mapped_points(output_path, &mapped);
    smooth_heatkernel(&mapped, NULL, heat_fwhm);
    values = (double *)malloc(sizeof(*values) * mapped.n_points);
    if (values == NULL) {
        fprintf(stderr, "分配特征阶段输出失败\n");
        exit(EXIT_FAILURE);
    }
    create_polygon_point_neighbours(&mapped, TRUE, &n_neighbours, &neighbours,
                                   NULL, NULL);
    normals = (Vector *)malloc(sizeof(*normals) * mapped.n_points);
    if (normals == NULL) {
        fprintf(stderr, "分配官方 stable normals 失败\n");
        free(values);
        exit(EXIT_FAILURE);
    }
    stable_normals(mapped.n_points, mapped.points, normals, n_neighbours,
                   neighbours);
    areas = cat_diag_compute_areas(mapped.n_points, mapped.points,
                                   n_neighbours, neighbours, 2);
    if (areas == NULL) {
        fprintf(stderr, "计算官方 mixed area 失败\n");
        free(normals);
        free(values);
        exit(EXIT_FAILURE);
    }
    init_csr_matrix(mapped.n_points, n_neighbours, neighbours, &laplacian);
    cot_laplacian_operator(mapped.n_points, mapped.points, &laplacian,
                           n_neighbours, neighbours);
    mean_curvature = compute_mean_curvature(mapped.n_points, mapped.points,
                                            areas, normals, &laplacian);
    raw_values = cat_diag_local_depth_potential(
        mapped.n_points, mapped.points, areas, &laplacian, mean_curvature,
        1.0 / 1000.0, 1.90);
    if (raw_values == NULL) {
        fprintf(stderr, "计算官方 raw depth-potential 失败\n");
        free_csr_matrix(&laplacian);
        free(mean_curvature);
        free(areas);
        free(normals);
        free(values);
        exit(EXIT_FAILURE);
    }
    get_smoothed_curvatures(&mapped, values, 50.0, 1000);
    write_stage(output_path, &mapped, areas, normals, mean_curvature,
                raw_values, values, &laplacian);
    free_csr_matrix(&laplacian);
    free(mean_curvature);
    free(areas);
    free(normals);
    free(n_neighbours);
    free(neighbours[0]);
    free(neighbours);
    free(raw_values);
    free(values);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr,
                "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE "
                "SOURCE_STAGE TARGET_STAGE\n",
                argv[0]);
        return 2;
    }
    make_feature(argv[1], argv[2], 15.0, argv[5]);
    make_feature(argv[3], argv[4], 10.0, argv[6]);
    return 0;
}
