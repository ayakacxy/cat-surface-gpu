/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出官方初始旋转调用在 heat-kernel 后生成的 depth-potential 特征。 */

#include <bicpl.h>
#include <CAT_Curvature.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_Smooth.h>

#include <stdio.h>
#include <stdlib.h>

static polygons_struct *load_polygons(const char *path,
                                      int *n_objects,
                                      object_struct ***objects)
{
    File_formats format;
    if (input_graphics_any_format((char *)path, &format, n_objects, objects) != OK ||
        *n_objects != 1 || get_object_type((*objects)[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个曲面对象: %s\n", path);
        exit(EXIT_FAILURE);
    }
    return get_polygons_ptr((*objects)[0]);
}

int main(int argc, char **argv)
{
    int source_objects_count = 0;
    int source_sphere_objects_count = 0;
    int target_objects_count = 0;
    int target_sphere_objects_count = 0;
    object_struct **source_objects = NULL;
    object_struct **source_sphere_objects = NULL;
    object_struct **target_objects = NULL;
    object_struct **target_sphere_objects = NULL;
    polygons_struct *source;
    polygons_struct *source_sphere;
    polygons_struct *target;
    polygons_struct *target_sphere;
    polygons_struct mapped_source = {0};
    polygons_struct mapped_source_sphere = {0};
    polygons_struct mapped_target = {0};
    polygons_struct mapped_target_sphere = {0};
    double *source_values;
    double *target_values;
    FILE *output;
    const int n_triangles = 81920;

    if (argc != 6) {
        fprintf(stderr,
                "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE OUTPUT\n",
                argv[0]);
        return 2;
    }

    source = load_polygons(argv[1], &source_objects_count, &source_objects);
    source_sphere = load_polygons(argv[2], &source_sphere_objects_count,
                                  &source_sphere_objects);
    target = load_polygons(argv[3], &target_objects_count, &target_objects);
    target_sphere = load_polygons(argv[4], &target_sphere_objects_count,
                                  &target_sphere_objects);

    translate_to_center_of_mass(source_sphere);
    translate_to_center_of_mass(target_sphere);
    for (int i = 0; i < source_sphere->n_points; ++i)
        set_vector_length(&source_sphere->points[i], 1.0);
    for (int i = 0; i < target_sphere->n_points; ++i)
        set_vector_length(&target_sphere->points[i], 1.0);

    resample_spherical_surface(source, source_sphere, &mapped_source,
                               NULL, NULL, n_triangles);
    resample_spherical_surface(source_sphere, source_sphere,
                               &mapped_source_sphere, NULL, NULL,
                               n_triangles);
    resample_spherical_surface(target, target_sphere, &mapped_target,
                               NULL, NULL, n_triangles);
    resample_spherical_surface(target_sphere, target_sphere,
                               &mapped_target_sphere, NULL, NULL,
                               n_triangles);

    /* 这两次平滑与 CAT_SurfWarpSolveDartelFlow 的初始旋转分支一致。 */
    smooth_heatkernel(&mapped_source, NULL, 15.0);
    smooth_heatkernel(&mapped_target, NULL, 10.0);

    source_values = (double *)malloc(
        sizeof(*source_values) * mapped_source.n_points);
    target_values = (double *)malloc(
        sizeof(*target_values) * mapped_target.n_points);
    if (source_values == NULL || target_values == NULL) {
        fprintf(stderr, "分配初始旋转特征缓冲区失败\n");
        free(source_values);
        free(target_values);
        return 1;
    }
    get_smoothed_curvatures(&mapped_source, source_values, 50.0, 1000);
    get_smoothed_curvatures(&mapped_target, target_values, 50.0, 1000);

    output = fopen(argv[5], "wb");
    if (output == NULL) {
        perror("打开初始旋转特征输出失败");
        free(source_values);
        free(target_values);
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
