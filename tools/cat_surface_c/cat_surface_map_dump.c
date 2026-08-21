/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出官方 CAT_SurfWarp 的真实曲率 sheet 图，供 Python/Triton 阶段对照。 */

#include <bicpl.h>
#include <CAT_Map.h>
#include <CAT_Resample.h>
#include <CAT_Curvature.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>

#include <stdint.h>
#include <string.h>
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
        loaded.n_objects != 1 ||
        get_object_type(loaded.objects[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个曲面对象: %s\n", path);
        exit(EXIT_FAILURE);
    }
    loaded.polygons = get_polygons_ptr(loaded.objects[0]);
    return loaded;
}

static void write_map(FILE *output,
                      polygons_struct *surface,
                      polygons_struct *sphere,
                      double fwhm,
                      int *dm,
                      int curvtype)
{
    size_t count = (size_t)dm[0] * (size_t)dm[1];
    double *values = (double *)malloc(sizeof(*values) * count);
    if (values == NULL) {
        fprintf(stderr, "分配曲率图失败\n");
        exit(EXIT_FAILURE);
    }
    map_sphere_values_to_sheet(surface, sphere, NULL, values, fwhm, dm,
                               curvtype);
    if (fwrite(values, sizeof(*values), count, output) != count) {
        fprintf(stderr, "写入曲率图失败\n");
        free(values);
        exit(EXIT_FAILURE);
    }
    free(values);
}

int main(int argc, char **argv)
{
    LoadedPolygons source;
    LoadedPolygons source_sphere;
    LoadedPolygons target;
    LoadedPolygons target_sphere;
    polygons_struct mapped_source;
    polygons_struct mapped_target;
    polygons_struct mapped_source_sphere;
    polygons_struct mapped_target_sphere;
    int dm[3] = {512, 256, 1};
    const int n_triangles = 81920;
    const int n_steps = 2;
    const int32_t header[3] = {n_steps, dm[0], dm[1]};
    double fwhm = 5.0;
    FILE *output;
    FILE *surface_output = NULL;
    FILE *curvature_output = NULL;

    int normalise_source_sphere = 1;
    if (argc == 11) {
        if (strcmp(argv[10], "--source-no-normalize") != 0) {
            fprintf(stderr, "未知 source sphere 选项: %s\n", argv[10]);
            return 2;
        }
        normalise_source_sphere = 0;
    }
    if (argc < 6 || argc > 11) {
        fprintf(stderr,
                "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE OUTPUT [SURFACE_OUTPUT] [CURVATURE_OUTPUT] [CURVTYPE0] [CURVTYPE1] [--source-no-normalize]\n",
                argv[0]);
        return 2;
    }

    int curvtypes[2] = {5, 5};
    if (argc >= 9)
        curvtypes[0] = atoi(argv[8]);
    if (argc >= 10)
        curvtypes[1] = atoi(argv[9]);

    source = load_polygons(argv[1]);
    source_sphere = load_polygons(argv[2]);
    target = load_polygons(argv[3]);
    target_sphere = load_polygons(argv[4]);
    if (normalise_source_sphere)
        translate_to_center_of_mass(source_sphere.polygons);
    translate_to_center_of_mass(target_sphere.polygons);
    if (normalise_source_sphere) {
        for (int i = 0; i < source_sphere.polygons->n_points; ++i)
            set_vector_length(&source_sphere.polygons->points[i], 1.0);
    }
    for (int i = 0; i < target_sphere.polygons->n_points; ++i)
        set_vector_length(&target_sphere.polygons->points[i], 1.0);

    mapped_source = (polygons_struct){0};
    mapped_target = (polygons_struct){0};
    mapped_source_sphere = (polygons_struct){0};
    mapped_target_sphere = (polygons_struct){0};
    resample_spherical_surface(source.polygons, source_sphere.polygons,
                               &mapped_source, NULL, NULL, n_triangles);
    resample_spherical_surface(target.polygons, target_sphere.polygons,
                               &mapped_target, NULL, NULL, n_triangles);
    resample_spherical_surface(source_sphere.polygons,
                               source_sphere.polygons,
                               &mapped_source_sphere, NULL, NULL, n_triangles);
    resample_spherical_surface(target_sphere.polygons,
                               target_sphere.polygons,
                               &mapped_target_sphere, NULL, NULL, n_triangles);

    output = fopen(argv[5], "wb");
    if (output == NULL) {
        perror("打开曲率图输出失败");
        return 1;
    }
    if (fwrite(header, sizeof(*header), 3, output) != 3) {
        fprintf(stderr, "写入曲率图头失败\n");
        fclose(output);
        return 1;
    }
    if (argc >= 7) {
        surface_output = fopen(argv[6], "wb");
        if (surface_output == NULL) {
            perror("打开重采样曲面输出失败");
            fclose(output);
            return 1;
        }
        fwrite(&mapped_source.n_points, sizeof(mapped_source.n_points), 1,
               surface_output);
        fwrite(mapped_source.points, sizeof(*mapped_source.points),
               mapped_source.n_points, surface_output);
        fwrite(mapped_target.points, sizeof(*mapped_target.points),
               mapped_target.n_points, surface_output);
        fwrite(mapped_source_sphere.points, sizeof(*mapped_source_sphere.points),
               mapped_source_sphere.n_points, surface_output);
        fwrite(mapped_target_sphere.points, sizeof(*mapped_target_sphere.points),
               mapped_target_sphere.n_points, surface_output);
        fclose(surface_output);
    }
    if (argc >= 8) {
        double *source_curvature = (double *)malloc(
            sizeof(*source_curvature) * mapped_source.n_points);
        double *target_curvature = (double *)malloc(
            sizeof(*target_curvature) * mapped_target.n_points);
        if (source_curvature == NULL || target_curvature == NULL) {
            fprintf(stderr, "分配曲率输出数组失败\n");
            fclose(output);
            free(source_curvature);
            free(target_curvature);
            return 1;
        }
        curvature_output = fopen(argv[7], "wb");
        if (curvature_output == NULL) {
            perror("打开曲率输出失败");
            fclose(output);
            free(source_curvature);
            free(target_curvature);
            return 1;
        }
        double curvature_fwhm = fwhm;
        for (int step = 0; step < n_steps; ++step) {
            get_smoothed_curvatures(&mapped_source, source_curvature,
                                    curvature_fwhm, curvtypes[step]);
            get_smoothed_curvatures(&mapped_target, target_curvature,
                                    curvature_fwhm, curvtypes[step]);
            fwrite(source_curvature, sizeof(*source_curvature),
                   mapped_source.n_points, curvature_output);
            fwrite(target_curvature, sizeof(*target_curvature),
                   mapped_target.n_points, curvature_output);
            curvature_fwhm /= 3.0;
        }
        fclose(curvature_output);
        free(source_curvature);
        free(target_curvature);
    }
    for (int step = 0; step < n_steps; ++step) {
        int curvtype = curvtypes[step];
        write_map(output, &mapped_source, &mapped_source_sphere, fwhm, dm,
                  curvtype);
        write_map(output, &mapped_target, &mapped_target_sphere, fwhm, dm,
                  curvtype);
        fwhm /= 3.0;
    }
    fclose(output);
    return 0;
}
