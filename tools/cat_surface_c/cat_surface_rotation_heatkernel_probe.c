/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出官方初始旋转分支在 heat-kernel 后的 coarse 曲面坐标。 */

#include <bicpl.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
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
        return NULL;
    }
    return get_polygons_ptr((*objects)[0]);
}

int main(int argc, char **argv)
{
    int source_objects_count = 0;
    int sphere_objects_count = 0;
    object_struct **source_objects = NULL;
    object_struct **sphere_objects = NULL;
    polygons_struct *source;
    polygons_struct *sphere;
    polygons_struct mapped = {0};
    FILE *output;
    double heat_fwhm;
    const int n_triangles = 81920;

    if (argc != 5) {
        fprintf(stderr, "用法: %s SURFACE SPHERE HEAT_FWHM OUTPUT\n", argv[0]);
        return 2;
    }
    heat_fwhm = atof(argv[3]);
    source = load_polygons(argv[1], &source_objects_count, &source_objects);
    sphere = load_polygons(argv[2], &sphere_objects_count, &sphere_objects);
    if (source == NULL || sphere == NULL)
        return 1;

    translate_to_center_of_mass(sphere);
    for (int i = 0; i < sphere->n_points; ++i)
        set_vector_length(&sphere->points[i], 1.0);
    resample_spherical_surface(source, sphere, &mapped, NULL, NULL,
                               n_triangles);
    if (heat_fwhm >= 0.0)
        smooth_heatkernel(&mapped, NULL, heat_fwhm);

    output = fopen(argv[4], "wb");
    if (output == NULL) {
        perror("打开 heat-kernel probe 输出失败");
        return 1;
    }
    fwrite(&mapped.n_points, sizeof(mapped.n_points), 1, output);
    for (int i = 0; i < mapped.n_points; ++i) {
        double point[3] = {
            Point_x(mapped.points[i]),
            Point_y(mapped.points[i]),
            Point_z(mapped.points[i]),
        };
        fwrite(point, sizeof(*point), 3, output);
    }
    fclose(output);
    return 0;
}
