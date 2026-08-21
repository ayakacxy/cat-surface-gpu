/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出官方 coarse 重采样使用的三角形和重心权重。 */

#include <bicpl.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>

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
    int sphere_objects = 0;
    int surface_objects = 0;
    object_struct **sphere_objects_ptr = NULL;
    object_struct **surface_objects_ptr = NULL;
    polygons_struct *sphere;
    polygons_struct *surface;
    polygons_struct output_sphere = {0};
    Point centre;
    double radius = 0.0;
    double bounds[6];
    FILE *output;
    const int n_triangles = 81920;

    if (argc != 3 && argc != 4) {
        fprintf(stderr, "用法: %s INPUT_SPHERE OUTPUT [INPUT_SURFACE]\n", argv[0]);
        return 2;
    }
    sphere = load_polygons(argv[1], &sphere_objects, &sphere_objects_ptr);
    if (sphere == NULL)
        return 1;
    surface = sphere;
    if (argc == 4) {
        surface = load_polygons(argv[3], &surface_objects, &surface_objects_ptr);
        if (surface == NULL)
            return 1;
    }
    translate_to_center_of_mass(sphere);
    for (int i = 0; i < sphere->n_points; ++i) {
        set_vector_length(&sphere->points[i], 1.0);
        radius += sqrt(
            Point_x(sphere->points[i]) * Point_x(sphere->points[i]) +
            Point_y(sphere->points[i]) * Point_y(sphere->points[i]) +
            Point_z(sphere->points[i]) * Point_z(sphere->points[i]));
    }
    radius = 0.975 * radius / (double)sphere->n_points;
    get_bounds(sphere, bounds);
    fill_Point(centre, bounds[0] + bounds[1], bounds[2] + bounds[3],
               bounds[4] + bounds[5]);
    create_tetrahedral_sphere(&centre, radius, radius, radius, n_triangles,
                              &output_sphere);
    create_polygons_bintree(sphere,
                            ROUND((double)sphere->n_items * 0.5));

    output = fopen(argv[2], "wb");
    if (output == NULL) {
        perror("打开 coarse 映射输出失败");
        return 1;
    }
    fwrite(&output_sphere.n_points, sizeof(output_sphere.n_points), 1, output);
    for (int i = 0; i < output_sphere.n_points; ++i) {
        Point point_on_sphere;
        Point polygon_points[MAX_POINTS_PER_POLYGON];
        double polygon_weights[MAX_POINTS_PER_POLYGON];
        int polygon = find_closest_polygon_point(&output_sphere.points[i], sphere,
                                                 &point_on_sphere);
        int size = get_polygon_points(sphere, polygon, polygon_points);
        if (size != 3) {
            fprintf(stderr, "输入球面存在非三角形面片: %d\n", size);
            fclose(output);
            return 1;
        }
        get_polygon_interpolation_weights(&point_on_sphere, size,
                                          polygon_points, polygon_weights);
        fwrite(&polygon, sizeof(polygon), 1, output);
        for (int corner = 0; corner < 3; ++corner) {
            int index = surface->indices[
                POINT_INDEX(surface->end_indices, polygon, corner)];
            fwrite(&index, sizeof(index), 1, output);
            fwrite(&polygon_weights[corner], sizeof(polygon_weights[corner]), 1,
                   output);
        }
    }
    fclose(output);
    delete_the_bintree(&sphere->bintree);
    return 0;
}
