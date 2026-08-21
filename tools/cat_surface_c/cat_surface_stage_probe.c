/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 测量新版 CAT-Surface 球面重采样与曲率图映射的真实阶段耗时。 */

#define _POSIX_C_SOURCE 200809L

#include <bicpl.h>
#include <CAT_Map.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    object_struct **objects;
    int n_objects;
    polygons_struct *polygons;
} LoadedPolygons;

static double monotonic_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static LoadedPolygons load_polygons(const char *path)
{
    LoadedPolygons loaded;
    File_formats format;

    loaded.objects = NULL;
    loaded.n_objects = 0;
    loaded.polygons = NULL;
    if (input_graphics_any_format((char *)path, &format, &loaded.n_objects,
                                  &loaded.objects) != OK ||
        loaded.n_objects != 1 ||
        get_object_type(loaded.objects[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个曲面对象: %s\n", path);
        exit(EXIT_FAILURE);
    }
    loaded.polygons = get_polygons_ptr(loaded.objects[0]);
    return loaded;
}

static void print_resample_stage(const char *name,
                                 polygons_struct *surface,
                                 polygons_struct *sphere,
                                 int n_triangles)
{
    polygons_struct *resampled;
    double start;

    resampled = (polygons_struct *)calloc(1, sizeof(*resampled));
    if (resampled == NULL) {
        fprintf(stderr, "分配重采样曲面失败\n");
        exit(EXIT_FAILURE);
    }
    start = monotonic_seconds();
    resample_spherical_surface(surface, sphere, resampled, NULL, NULL,
                               n_triangles);
    printf("resample_%s_seconds %.9f points %d triangles %d\n",
           name, monotonic_seconds() - start, resampled->n_points,
           resampled->n_items);
    fflush(stdout);
}

static void print_map_stage(const char *name,
                            polygons_struct *surface,
                            polygons_struct *sphere,
                            int dm[3],
                            int curvtype)
{
    size_t n_values = (size_t)dm[0] * (size_t)dm[1];
    double *mapped_data = (double *)malloc(sizeof(*mapped_data) * n_values);
    double start;

    if (mapped_data == NULL) {
        fprintf(stderr, "分配 sheet 映射数组失败\n");
        exit(EXIT_FAILURE);
    }
    start = monotonic_seconds();
    map_sphere_values_to_sheet(surface, sphere, NULL, mapped_data, 0.0, dm,
                               curvtype);
    printf("map_%s_seconds %.9f values %zu first %.17g\n",
           name, monotonic_seconds() - start, n_values, mapped_data[0]);
    fflush(stdout);
    free(mapped_data);
}

int main(int argc, char **argv)
{
    LoadedPolygons source;
    LoadedPolygons source_sphere;
    LoadedPolygons target;
    LoadedPolygons target_sphere;
    int dm[3] = {512, 256, 1};
    const int n_triangles = 81920;
    double start;

    if (argc != 5) {
        fprintf(stderr,
                "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE\n",
                argv[0]);
        return 2;
    }

    source = load_polygons(argv[1]);
    source_sphere = load_polygons(argv[2]);
    target = load_polygons(argv[3]);
    target_sphere = load_polygons(argv[4]);

    start = monotonic_seconds();
    translate_to_center_of_mass(source_sphere.polygons);
    translate_to_center_of_mass(target_sphere.polygons);
    for (int i = 0; i < source_sphere.polygons->n_points; ++i)
        set_vector_length(&source_sphere.polygons->points[i], 1.0);
    for (int i = 0; i < target_sphere.polygons->n_points; ++i)
        set_vector_length(&target_sphere.polygons->points[i], 1.0);
    printf("normalize_spheres_seconds %.9f source_points %d target_points %d\n",
           monotonic_seconds() - start, source_sphere.polygons->n_points,
           target_sphere.polygons->n_points);
    fflush(stdout);

    print_resample_stage("source", source.polygons, source_sphere.polygons,
                         n_triangles);
    print_resample_stage("target", target.polygons, target_sphere.polygons,
                         n_triangles);
    print_resample_stage("source_sphere", source_sphere.polygons,
                         source_sphere.polygons, n_triangles);
    print_resample_stage("target_sphere", target_sphere.polygons,
                         target_sphere.polygons, n_triangles);

    /* 映射阶段内部同时包含曲率计算、球面定位和双线性权重累加。 */
    {
        polygons_struct *mapped_source = (polygons_struct *)calloc(1, sizeof(*mapped_source));
        polygons_struct *mapped_target = (polygons_struct *)calloc(1, sizeof(*mapped_target));
        polygons_struct *mapped_source_sphere = (polygons_struct *)calloc(1, sizeof(*mapped_source_sphere));
        polygons_struct *mapped_target_sphere = (polygons_struct *)calloc(1, sizeof(*mapped_target_sphere));

        if (mapped_source == NULL || mapped_target == NULL ||
            mapped_source_sphere == NULL || mapped_target_sphere == NULL) {
            fprintf(stderr, "分配映射阶段曲面失败\n");
            return 1;
        }
        resample_spherical_surface(source.polygons, source_sphere.polygons,
                                   mapped_source, NULL, NULL, n_triangles);
        resample_spherical_surface(target.polygons, target_sphere.polygons,
                                   mapped_target, NULL, NULL, n_triangles);
        resample_spherical_surface(source_sphere.polygons,
                                   source_sphere.polygons,
                                   mapped_source_sphere, NULL, NULL,
                                   n_triangles);
        resample_spherical_surface(target_sphere.polygons,
                                   target_sphere.polygons,
                                   mapped_target_sphere, NULL, NULL,
                                   n_triangles);
        print_map_stage("source", mapped_source, mapped_source_sphere, dm, 5);
        print_map_stage("target", mapped_target, mapped_target_sphere, dm, 5);
    }

    return 0;
}
