/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 测量官方 CAT 初始旋转中曲率生成、单次重采样和完整搜索的阶段时间。 */

#include <bicpl.h>
#include <CAT_Curvature.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_Warp.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    object_struct **objects;
    int n_objects;
    polygons_struct *polygons;
} LoadedPolygons;

static LoadedPolygons load_polygons(const char *path)
{
    LoadedPolygons loaded;
    File_formats format;

    loaded.objects = NULL;
    loaded.n_objects = 0;
    loaded.polygons = NULL;
    if (input_graphics_any_format((char *)path, &format, &loaded.n_objects,
                                  &loaded.objects) != OK ||
        loaded.n_objects != 1 || get_object_type(loaded.objects[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个曲面对象: %s\n", path);
        exit(EXIT_FAILURE);
    }
    loaded.polygons = get_polygons_ptr(loaded.objects[0]);
    return loaded;
}

static double elapsed_seconds(clock_t start)
{
    return (double)(clock() - start) / (double)CLOCKS_PER_SEC;
}

int main(int argc, char **argv)
{
    LoadedPolygons source = {0};
    LoadedPolygons source_sphere = {0};
    LoadedPolygons target = {0};
    LoadedPolygons target_sphere = {0};
    polygons_struct mapped_source = {0};
    polygons_struct mapped_source_sphere = {0};
    polygons_struct mapped_target = {0};
    polygons_struct mapped_target_sphere = {0};
    double *source_curvature;
    double *target_curvature;
    double *resampled_target;
    double rotation[9];
    double angles[3] = {0.13, -0.09, 0.11};
    clock_t start;
    double source_curvature_seconds;
    double target_curvature_seconds;
    double one_resample_seconds;
    double bintree_build_seconds;
    double cached_resample_seconds;
    double full_rotation_seconds;
    const int n_triangles = 81920;

    if (argc != 5) {
        fprintf(stderr, "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE\n", argv[0]);
        return 2;
    }

    source = load_polygons(argv[1]);
    source_sphere = load_polygons(argv[2]);
    target = load_polygons(argv[3]);
    target_sphere = load_polygons(argv[4]);
    translate_to_center_of_mass(source_sphere.polygons);
    translate_to_center_of_mass(target_sphere.polygons);
    for (int i = 0; i < source_sphere.polygons->n_points; ++i)
        set_vector_length(&source_sphere.polygons->points[i], 1.0);
    for (int i = 0; i < target_sphere.polygons->n_points; ++i)
        set_vector_length(&target_sphere.polygons->points[i], 1.0);

    resample_spherical_surface(source.polygons, source_sphere.polygons,
                               &mapped_source, NULL, NULL, n_triangles);
    resample_spherical_surface(source_sphere.polygons, source_sphere.polygons,
                               &mapped_source_sphere, NULL, NULL, n_triangles);
    resample_spherical_surface(target.polygons, target_sphere.polygons,
                               &mapped_target, NULL, NULL, n_triangles);
    resample_spherical_surface(target_sphere.polygons, target_sphere.polygons,
                               &mapped_target_sphere, NULL, NULL, n_triangles);

    source_curvature = (double *)malloc(sizeof(*source_curvature) * mapped_source.n_points);
    target_curvature = (double *)malloc(sizeof(*target_curvature) * mapped_target.n_points);
    resampled_target = (double *)malloc(sizeof(*resampled_target) * mapped_source.n_points);
    if (source_curvature == NULL || target_curvature == NULL || resampled_target == NULL) {
        fprintf(stderr, "分配旋转阶段探针缓冲区失败\n");
        return 1;
    }

    start = clock();
    get_smoothed_curvatures(&mapped_source, source_curvature, 50.0, 1000);
    source_curvature_seconds = elapsed_seconds(start);

    start = clock();
    get_smoothed_curvatures(&mapped_target, target_curvature, 50.0, 1000);
    target_curvature_seconds = elapsed_seconds(start);

    rotation_to_matrix(rotation, angles[0], angles[1], angles[2]);
    rotate_polygons(&mapped_source_sphere, NULL, rotation);
    start = clock();
    resample_values_sphere(target_sphere.polygons, &mapped_source_sphere,
                           target_curvature, resampled_target, 0, 0);
    one_resample_seconds = elapsed_seconds(start);

    /* 单独测量空间树构建，并测量复用已建空间树的查询时间。 */
    start = clock();
    create_polygons_bintree(target_sphere.polygons,
                            ROUND((double)target_sphere.polygons->n_items * 0.5));
    bintree_build_seconds = elapsed_seconds(start);
    start = clock();
    resample_values_sphere_noscale(target_sphere.polygons, &mapped_source_sphere,
                                   target_curvature, resampled_target, 0);
    cached_resample_seconds = elapsed_seconds(start);

    start = clock();
    rotate_polygons_to_atlas(&mapped_source, &mapped_source_sphere,
                             &mapped_target, &mapped_target_sphere,
                             50.0, 1000, rotation, 0);
    full_rotation_seconds = elapsed_seconds(start);

    printf("{\"source_curvature_seconds\": %.9f, "
           "\"target_curvature_seconds\": %.9f, "
           "\"one_resample_seconds\": %.9f, "
           "\"bintree_build_seconds\": %.9f, "
           "\"cached_resample_seconds\": %.9f, "
           "\"full_rotation_seconds\": %.9f}\n",
           source_curvature_seconds, target_curvature_seconds,
           one_resample_seconds, bintree_build_seconds,
           cached_resample_seconds, full_rotation_seconds);

    free(source_curvature);
    free(target_curvature);
    free(resampled_target);
    return 0;
}
