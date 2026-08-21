/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 生成官方 apply_warp 的零形变输出，校验 GPU 球面坐标映射约定。 */

#include <bicpl.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_Warp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    File_formats format;
    int n_objects = 0;
    object_struct **objects = NULL;
    polygons_struct *sphere;
    int dm[3] = {512, 256, 1};
    size_t map_size = (size_t)dm[0] * (size_t)dm[1];
    double *flow;
    FILE *output;
    FILE *unit_output;
    polygons_struct unit_sphere;
    Point unit_point;

    int normalise_input = 1;

    if (argc < 3 || argc > 5) {
        fprintf(stderr,
                "用法: %s INPUT_SPHERE OUTPUT [UNIT_OUTPUT] [--no-normalize]\n",
                argv[0]);
        return 2;
    }
    if (argc == 5 && strcmp(argv[4], "--no-normalize") != 0) {
        fprintf(stderr, "未知选项: %s\n", argv[4]);
        return 2;
    }
    if (argc == 5)
        normalise_input = 0;
    if (input_graphics_any_format(argv[1], &format, &n_objects, &objects) != OK ||
        n_objects != 1 || get_object_type(objects[0]) != POLYGONS) {
        fprintf(stderr, "无法读取球面对象\n");
        return 1;
    }
    sphere = get_polygons_ptr(objects[0]);
    if (normalise_input)
        translate_to_center_of_mass(sphere);
    for (int i = 0; i < sphere->n_points; ++i)
        if (normalise_input)
            set_vector_length(&sphere->points[i], 1.0);
    if (argc >= 4) {
        copy_polygons(sphere, &unit_sphere);
        for (int i = 0; i < unit_sphere.n_points; ++i)
            set_vector_length(&unit_sphere.points[i], 1.0);
        create_polygons_bintree(sphere, ROUND((double)sphere->n_items * 0.5));
        unit_output = fopen(argv[3], "wb");
        if (unit_output == NULL) {
            perror("打开 unit sphere 输出失败");
            return 1;
        }
        fwrite(&sphere->n_points, sizeof(sphere->n_points), 1, unit_output);
        for (int i = 0; i < sphere->n_points; ++i) {
            map_point_to_unit_sphere(sphere, &sphere->points[i], &unit_sphere,
                                     &unit_point);
            fwrite(&unit_point, sizeof(unit_point), 1, unit_output);
        }
        fclose(unit_output);
        delete_the_bintree(&sphere->bintree);
        delete_polygons(&unit_sphere);
    }
    flow = (double *)malloc(sizeof(*flow) * 2 * map_size);
    if (flow == NULL) {
        fprintf(stderr, "分配零流场失败\n");
        return 1;
    }
    for (int y = 0; y < dm[1]; ++y) {
        for (int x = 0; x < dm[0]; ++x) {
            size_t index = (size_t)x + (size_t)dm[0] * (size_t)y;
            flow[index] = (double)x + 1.0;
            flow[index + map_size] = (double)y + 1.0;
        }
    }
    apply_warp(sphere, sphere, flow, dm, 1);
    output = fopen(argv[2], "wb");
    if (output == NULL) {
        perror("打开零流输出失败");
        free(flow);
        return 1;
    }
    fwrite(&sphere->n_points, sizeof(sphere->n_points), 1, output);
    fwrite(sphere->points, sizeof(*sphere->points), sphere->n_points, output);
    fclose(output);
    free(flow);
    return 0;
}
