/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出官方初始旋转代价使用的规则球面曲率，供定位器 A/B。 */

#include <bicpl.h>
#include <CAT_Curvature.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    polygons_struct mapped_target = {0};
    double *source_values;
    double *target_values;
    FILE *output;
    const int n_triangles = 81920;
    const double fwhm = 50.0;
    const int curvtype = 1000;

    if (argc != 6) {
        fprintf(stderr, "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE OUTPUT\n",
                argv[0]);
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
    resample_spherical_surface(target.polygons, target_sphere.polygons,
                               &mapped_target, NULL, NULL, n_triangles);
    source_values = (double *)malloc(sizeof(*source_values) * mapped_source.n_points);
    target_values = (double *)malloc(sizeof(*target_values) * mapped_target.n_points);
    if (source_values == NULL || target_values == NULL) {
        fprintf(stderr, "分配旋转曲率缓冲区失败\n");
        return 1;
    }
    get_smoothed_curvatures(&mapped_source, source_values, fwhm, curvtype);
    get_smoothed_curvatures(&mapped_target, target_values, fwhm, curvtype);

    output = fopen(argv[5], "wb");
    if (output == NULL) {
        perror("打开旋转曲率输出失败");
        return 1;
    }
    fwrite(&mapped_source.n_points, sizeof(mapped_source.n_points), 1, output);
    fwrite(&mapped_target.n_points, sizeof(mapped_target.n_points), 1, output);
    fwrite(source_values, sizeof(*source_values), mapped_source.n_points, output);
    fwrite(target_values, sizeof(*target_values), mapped_target.n_points, output);
    fclose(output);
    free(source_values);
    free(target_values);
    return 0;
}
