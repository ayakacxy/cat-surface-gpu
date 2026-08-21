/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 使用官方 CAT_Warp 对已导出的 flow 生成球面输出，供 GPU 最终阶段对照。 */

#include <bicpl.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_Warp.h>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    File_formats format;
    int n_objects = 0;
    object_struct **objects = NULL;
    polygons_struct *sphere;
    const int dm[3] = {512, 256, 1};
    const size_t map_size = (size_t)dm[0] * (size_t)dm[1];
    double *flow;
    FILE *flow_input;
    FILE *output;

    if (argc != 4) {
        fprintf(stderr, "用法: %s INPUT_SPHERE FLOW OUTPUT\n", argv[0]);
        return 2;
    }
    if (input_graphics_any_format(argv[1], &format, &n_objects, &objects) != OK ||
        n_objects != 1 || get_object_type(objects[0]) != POLYGONS) {
        fprintf(stderr, "无法读取球面对象\n");
        return 1;
    }
    sphere = get_polygons_ptr(objects[0]);
    translate_to_center_of_mass(sphere);
    for (int i = 0; i < sphere->n_points; ++i)
        set_vector_length(&sphere->points[i], 1.0);

    flow = (double *)malloc(sizeof(*flow) * 2 * map_size);
    if (flow == NULL) {
        fprintf(stderr, "分配 flow 失败\n");
        return 1;
    }
    flow_input = fopen(argv[2], "rb");
    if (flow_input == NULL ||
        fread(flow, sizeof(*flow), 2 * map_size, flow_input) != 2 * map_size) {
        fprintf(stderr, "读取 flow 失败\n");
        if (flow_input != NULL)
            fclose(flow_input);
        free(flow);
        return 1;
    }
    fclose(flow_input);
    apply_warp(sphere, sphere, flow, (int *)dm, 1);

    output = fopen(argv[3], "wb");
    if (output == NULL) {
        perror("打开 warped 输出失败");
        free(flow);
        return 1;
    }
    fwrite(&sphere->n_points, sizeof(sphere->n_points), 1, output);
    fwrite(sphere->points, sizeof(*sphere->points), sphere->n_points, output);
    fclose(output);
    free(flow);
    return 0;
}
