/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 用给定特征文件调用官方 compute_cost，核对旋转代价的 C 合同。 */

#include <bicpl.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_Warp.h>

#include <stdio.h>
#include <stdlib.h>

extern double compute_cost(double *angles, void *params);

static polygons_struct *load_polygons(const char *path,
                                      int *n_objects,
                                      object_struct ***objects)
{
    File_formats format;
    if (input_graphics_any_format((char *)path, &format, n_objects, objects) != OK ||
        *n_objects != 1 || get_object_type((*objects)[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个球面对象: %s\n", path);
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
    OptimizationParams params;
    double *source_values;
    double *target_values;
    double angles[3];
    FILE *input;
    int source_count;
    int target_count;

    if (argc != 9) {
        fprintf(stderr,
                "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE VALUES ALPHA BETA GAMMA\n",
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
                               NULL, NULL, 81920);
    resample_spherical_surface(source_sphere, source_sphere,
                               &mapped_source_sphere, NULL, NULL, 81920);
    resample_spherical_surface(target, target_sphere, &mapped_target,
                               NULL, NULL, 81920);
    resample_spherical_surface(target_sphere, target_sphere,
                               &mapped_target_sphere, NULL, NULL, 81920);

    input = fopen(argv[5], "rb");
    if (input == NULL ||
        fread(&source_count, sizeof(source_count), 1, input) != 1 ||
        fread(&target_count, sizeof(target_count), 1, input) != 1 ||
        source_count != mapped_source.n_points ||
        target_count != mapped_target.n_points) {
        fprintf(stderr, "旋转特征文件头与曲面点数不一致\n");
        return 1;
    }
    source_values = (double *)malloc(sizeof(*source_values) * source_count);
    target_values = (double *)malloc(sizeof(*target_values) * target_count);
    if (source_values == NULL || target_values == NULL ||
        fread(source_values, sizeof(*source_values), source_count, input) !=
            (size_t)source_count ||
        fread(target_values, sizeof(*target_values), target_count, input) !=
            (size_t)target_count) {
        fprintf(stderr, "读取旋转特征文件失败\n");
        return 1;
    }
    fclose(input);

    params.src = &mapped_source;
    params.src_sphere = &mapped_source_sphere;
    params.trg_sphere = &mapped_target_sphere;
    params.orig_trg = target_values;
    params.map_trg = (double *)calloc((size_t)source_count, sizeof(double));
    params.map_src = source_values;
    params.pre_rot = NULL;
    create_polygons_bintree(params.trg_sphere,
                            ROUND((double)params.trg_sphere->n_items * 0.5));
    angles[0] = atof(argv[6]);
    angles[1] = atof(argv[7]);
    angles[2] = atof(argv[8]);
    printf("{\"cost\":%.17g}\n", compute_cost(angles, &params));
    delete_the_bintree(&params.trg_sphere->bintree);
    free(params.map_trg);
    free(source_values);
    free(target_values);
    return 0;
}
