/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 测量官方 CAT 初始旋转搜索及其 depth-potential 阶段。 */

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
    double rotation[3] = {0.0, 0.0, 0.0};
    const int n_triangles = 81920;
    clock_t cpu_start;

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

    cpu_start = clock();
    rotate_polygons_to_atlas(&mapped_source, &mapped_source_sphere,
                             &mapped_target, &mapped_target_sphere,
                             50.0, 1000, rotation, 0);
    printf("{\"cpu_seconds\": %.9f, \"rotation\": [%.17g, %.17g, %.17g]}\n",
           (double)(clock() - cpu_start) / (double)CLOCKS_PER_SEC,
           rotation[0], rotation[1], rotation[2]);
    return 0;
}
