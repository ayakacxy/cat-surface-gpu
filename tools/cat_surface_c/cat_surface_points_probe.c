/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出 CAT 读取后的曲面顶点，核对 GIFTI 输入的数值表示。 */

#include <bicpl.h>
#include <CAT_SurfaceIO.h>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    File_formats format;
    int n_objects = 0;
    object_struct **objects = NULL;
    polygons_struct *polygons;
    FILE *output;

    if (argc != 3) {
        fprintf(stderr, "用法: %s INPUT_GIFTI OUTPUT\n", argv[0]);
        return 2;
    }
    if (input_graphics_any_format(argv[1], &format, &n_objects, &objects) != OK ||
        n_objects != 1 || get_object_type(objects[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个曲面对象: %s\n", argv[1]);
        return 1;
    }
    polygons = get_polygons_ptr(objects[0]);
    output = fopen(argv[2], "wb");
    if (output == NULL) {
        perror("打开顶点输出失败");
        return 1;
    }
    fwrite(&polygons->n_points, sizeof(polygons->n_points), 1, output);
    fwrite(polygons->points, sizeof(*polygons->points), polygons->n_points,
           output);
    fclose(output);
    return 0;
}
