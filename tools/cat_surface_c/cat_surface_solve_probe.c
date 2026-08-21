/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出官方 CAT_SurfWarp 单次默认 DARTEL solve 的最终二维变换。 */

#include <bicpl.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_SurfWarpDartel.h>
#include <CAT_Warp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        loaded.n_objects != 1 ||
        get_object_type(loaded.objects[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个曲面对象: %s\n", path);
        exit(EXIT_FAILURE);
    }
    loaded.polygons = get_polygons_ptr(loaded.objects[0]);
    return loaded;
}

int main(int argc, char **argv)
{
    LoadedPolygons source;
    LoadedPolygons source_sphere;
    LoadedPolygons target;
    LoadedPolygons target_sphere;
    const int dm[3] = {512, 256, 1};
    const int n_loops = 6;
    int n_steps = 2;
    const int n_triangles = 81920;
    const size_t xy_size = (size_t)dm[0] * (size_t)dm[1];
    struct dartel_prm prm[6];
    CAT_SurfWarpDartelOptions options;
    double *flow;
    double fwhm = 5.0;
    double fwhm_surf = 0.0;
    double rot[3] = {0.0, 0.0, 0.0};
    FILE *output;
    FILE *sphere_output;
    int normalise_source_sphere = 1;
    int normalise_target_sphere = 1;

    if (argc < 6 || argc > 11) {
        fprintf(stderr,
                "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE OUTPUT [SPHERE_OUTPUT] [N_STEPS] [--source-no-normalize] [--fwhm VALUE]\n",
                argv[0]);
        return 2;
    }
    if (argc >= 9) {
        if (strcmp(argv[8], "--source-no-normalize") != 0 &&
            strcmp(argv[8], "--no-normalize-both") != 0) {
            fprintf(stderr, "未知选项: %s\n", argv[8]);
            return 2;
        }
        normalise_source_sphere = 0;
        if (strcmp(argv[8], "--no-normalize-both") == 0)
            normalise_target_sphere = 0;
    }
    if (argc == 11) {
        if (strcmp(argv[9], "--fwhm") != 0) {
            fprintf(stderr, "未知选项: %s\n", argv[9]);
            return 2;
        }
        fwhm = atof(argv[10]);
    }
    if (argc == 8) {
        n_steps = atoi(argv[7]);
        if (n_steps < 1 || n_steps > 3) {
            fprintf(stderr, "N_STEPS 必须在 1 到 3 之间\n");
            return 2;
        }
    }

    source = load_polygons(argv[1]);
    source_sphere = load_polygons(argv[2]);
    target = load_polygons(argv[3]);
    target_sphere = load_polygons(argv[4]);
    if (normalise_source_sphere)
        translate_to_center_of_mass(source_sphere.polygons);
    if (normalise_target_sphere)
        translate_to_center_of_mass(target_sphere.polygons);
    for (int i = 0; i < source_sphere.polygons->n_points; ++i)
        if (normalise_source_sphere)
            set_vector_length(&source_sphere.polygons->points[i], 1.0);
    for (int i = 0; i < target_sphere.polygons->n_points; ++i)
        if (normalise_target_sphere)
            set_vector_length(&target_sphere.polygons->points[i], 1.0);

    for (int i = 0; i < n_loops; ++i) {
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
        for (int j = i + 1; j < n_loops; ++j)
            prm[j].rparam[2] = prm[i].rparam[2];
    }

    /* 复现 CAT_SurfWarp.c 中 mu 逐 loop 变化的参数表。 */
    {
        double mu = 0.125;
        for (int i = 0; i < n_loops; ++i) {
            prm[i].rparam[2] = mu;
            if ((i + 1) % 4 == 0)
                mu /= 1.25;
        }
    }

    options.multires_levels = 0;
    options.n_triangles = n_triangles;
    options.verbose = 0;
    options.debug = 0;
    options.rotate = 0;
    options.curvtype0 = 5;
    options.curvtype1 = 5;
    options.curvtype2 = 2;
    options.fwhm = &fwhm;
    options.fwhm_surf = &fwhm_surf;
    options.jacdet_file = NULL;

    flow = (double *)calloc(2 * xy_size, sizeof(*flow));
    if (flow == NULL) {
        fprintf(stderr, "分配 solve flow 失败\n");
        return 1;
    }
    if (CAT_SurfWarpSolveDartelFlow(
            source.polygons, source_sphere.polygons,
            target.polygons, target_sphere.polygons,
            prm, (int *)dm, n_steps, rot, flow, n_loops, &options) != OK) {
        fprintf(stderr, "官方 CAT_SurfWarp solve 失败\n");
        free(flow);
        return 1;
    }

    output = fopen(argv[5], "wb");
    if (output == NULL) {
        perror("打开 solve 输出失败");
        free(flow);
        return 1;
    }
    fwrite(flow, sizeof(*flow), 2 * xy_size, output);
    fclose(output);
    if (argc >= 7) {
        apply_warp(source_sphere.polygons, source_sphere.polygons, flow,
                   (int *)dm, 1);
        sphere_output = fopen(argv[6], "wb");
        if (sphere_output == NULL) {
            perror("打开 warped sphere 输出失败");
            free(flow);
            return 1;
        }
        fwrite(&source_sphere.polygons->n_points,
               sizeof(source_sphere.polygons->n_points), 1, sphere_output);
        fwrite(source_sphere.polygons->points,
               sizeof(*source_sphere.polygons->points),
               source_sphere.polygons->n_points, sphere_output);
        fclose(sphere_output);
    }
    free(flow);
    return 0;
}
