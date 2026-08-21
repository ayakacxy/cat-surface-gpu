/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出官方 CAT_SurfWarp 初始旋转调用实际修改后的 source sphere。 */

#include <bicpl.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_SurfWarpDartel.h>

#include <stdio.h>
#include <stdlib.h>

static polygons_struct *load_polygons(const char *path,
                                      File_formats *format,
                                      int *n_objects,
                                      object_struct ***objects)
{
    if (input_graphics_any_format((char *)path, format, n_objects, objects) != OK ||
        *n_objects != 1 || get_object_type((*objects)[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个曲面对象: %s\n", path);
        exit(EXIT_FAILURE);
    }
    return get_polygons_ptr((*objects)[0]);
}

int main(int argc, char **argv)
{
    File_formats source_format;
    File_formats source_sphere_format;
    File_formats target_format;
    File_formats target_sphere_format;
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
    struct dartel_prm prm[6];
    CAT_SurfWarpDartelOptions options;
    int dm[3] = {512, 256, 1};
    double fwhm = 5.0;
    double fwhm_surf = 0.0;
    double rot[3] = {0.0, 0.0, 0.0};
    double *flow;

    if (argc != 6) {
        fprintf(stderr,
                "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE OUTPUT\n",
                argv[0]);
        return 2;
    }

    source = load_polygons(argv[1], &source_format, &source_objects_count,
                           &source_objects);
    source_sphere = load_polygons(argv[2], &source_sphere_format,
                                  &source_sphere_objects_count,
                                  &source_sphere_objects);
    target = load_polygons(argv[3], &target_format, &target_objects_count,
                           &target_objects);
    target_sphere = load_polygons(argv[4], &target_sphere_format,
                                  &target_sphere_objects_count,
                                  &target_sphere_objects);

    translate_to_center_of_mass(source_sphere);
    translate_to_center_of_mass(target_sphere);
    for (int i = 0; i < source_sphere->n_points; ++i)
        set_vector_length(&source_sphere->points[i], 1.0);
    for (int i = 0; i < target_sphere->n_points; ++i)
        set_vector_length(&target_sphere->points[i], 1.0);

    for (int i = 0; i < 6; ++i) {
        prm[i].rtype = 1;
        prm[i].rparam[0] = 1.0;
        prm[i].rparam[1] = 1.0;
        prm[i].rparam[2] = 0.125;
        prm[i].rparam[3] = 0.0;
        prm[i].rparam[4] = 0.0;
        prm[i].lmreg = 1e-3;
        prm[i].cycles = 3;
        prm[i].its = 3;
        prm[i].k = i;
        prm[i].code = 1;
        if ((i + 1) % 4 == 0)
            prm[i].rparam[2] /= 1.25;
    }

    options.multires_levels = 0;
    options.n_triangles = 81920;
    options.verbose = 0;
    options.debug = 0;
    options.rotate = 1;
    options.curvtype0 = 5;
    options.curvtype1 = 5;
    options.curvtype2 = 2;
    options.fwhm = &fwhm;
    options.fwhm_surf = &fwhm_surf;
    options.jacdet_file = NULL;

    flow = (double *)calloc((size_t)2 * 512 * 256, sizeof(*flow));
    if (flow == NULL) {
        fprintf(stderr, "分配初始旋转 flow 失败\n");
        return 1;
    }
    if (CAT_SurfWarpSolveDartelFlow(
            source, source_sphere, target, target_sphere, prm, dm, 2, rot,
            flow, -1, &options) != OK) {
        fprintf(stderr, "官方初始旋转调用失败\n");
        free(flow);
        return 1;
    }

    if (output_graphics_any_format(argv[5], source_sphere_format,
                                   source_sphere_objects_count,
                                   source_sphere_objects, NULL) != OK) {
        fprintf(stderr, "写出初始旋转 source sphere 失败\n");
        free(flow);
        return 1;
    }
    printf("{\"rotation\":[%.17g,%.17g,%.17g],\"fwhm_after\":%.17g}\n",
           rot[0], rot[1], rot[2], fwhm);
    free(flow);
    return 0;
}
