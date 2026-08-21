/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 导出官方 bintree 三角形定位结果，验证 GPU 空间索引的候选覆盖。 */

#include <bicpl.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_Warp.h>

#include <stdint.h>
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
        loaded.n_objects != 1 || get_object_type(loaded.objects[0]) != POLYGONS) {
        fprintf(stderr, "无法读取单个球面对象: %s\n", path);
        exit(EXIT_FAILURE);
    }
    loaded.polygons = get_polygons_ptr(loaded.objects[0]);
    return loaded;
}

int main(int argc, char **argv)
{
    LoadedPolygons source;
    LoadedPolygons target;
    polygons_struct mapped_source = {0};
    polygons_struct mapped_target = {0};
    double rotation[9];
    Point point_on_target;
    Point polygon_points[MAX_POINTS_PER_POLYGON];
    double weights[MAX_POINTS_PER_POLYGON];
    FILE *output;
    FILE *rotated_output = NULL;
    FILE *geometry_output = NULL;
    const int n_triangles = 81920;

    if (argc != 7 && argc != 8 && argc != 9) {
        fprintf(stderr,
                "用法: %s SOURCE_SPHERE TARGET_SPHERE OUTPUT ALPHA BETA GAMMA [ROTATED_OUTPUT] [GEOMETRY_OUTPUT]\n",
                argv[0]);
        return 2;
    }
    source = load_polygons(argv[1]);
    target = load_polygons(argv[2]);
    translate_to_center_of_mass(source.polygons);
    translate_to_center_of_mass(target.polygons);
    for (int i = 0; i < source.polygons->n_points; ++i)
        set_vector_length(&source.polygons->points[i], 1.0);
    for (int i = 0; i < target.polygons->n_points; ++i)
        set_vector_length(&target.polygons->points[i], 1.0);
    resample_spherical_surface(source.polygons, source.polygons, &mapped_source,
                               NULL, NULL, n_triangles);
    resample_spherical_surface(target.polygons, target.polygons, &mapped_target,
                               NULL, NULL, n_triangles);
    rotation_to_matrix(rotation, atof(argv[4]), atof(argv[5]), atof(argv[6]));
    rotate_polygons(&mapped_source, NULL, rotation);
    if (argc == 8) {
        rotated_output = fopen(argv[7], "wb");
        if (rotated_output == NULL) {
            perror("打开旋转点输出失败");
            return 1;
        }
        fwrite(&mapped_source.n_points, sizeof(mapped_source.n_points), 1,
               rotated_output);
        fwrite(mapped_source.points, sizeof(*mapped_source.points),
               mapped_source.n_points, rotated_output);
        fclose(rotated_output);
    }
    if (argc == 9) {
        geometry_output = fopen(argv[8], "wb");
        if (geometry_output == NULL) {
            perror("打开旋转几何输出失败");
            return 1;
        }
        fwrite(&mapped_target.n_points, sizeof(mapped_target.n_points), 1,
               geometry_output);
        fwrite(&mapped_target.n_items, sizeof(mapped_target.n_items), 1,
               geometry_output);
        fwrite(mapped_target.points, sizeof(*mapped_target.points),
               mapped_target.n_points, geometry_output);
        for (int face = 0; face < mapped_target.n_items; ++face) {
            for (int corner = 0; corner < 3; ++corner) {
                int index = mapped_target.indices[
                    POINT_INDEX(mapped_target.end_indices, face, corner)];
                fwrite(&index, sizeof(index), 1, geometry_output);
            }
        }
        fclose(geometry_output);
    }
    create_polygons_bintree(&mapped_target,
                            ROUND((double)mapped_target.n_items * 0.5));

    output = fopen(argv[3], "wb");
    if (output == NULL) {
        perror("打开旋转映射输出失败");
        return 1;
    }
    fwrite(&mapped_source.n_points, sizeof(mapped_source.n_points), 1, output);
    for (int i = 0; i < mapped_source.n_points; ++i) {
        int polygon = find_closest_polygon_point(&mapped_source.points[i],
                                                 &mapped_target,
                                                 &point_on_target);
        int size = get_polygon_points(&mapped_target, polygon, polygon_points);
        if (size != 3) {
            fprintf(stderr, "输出球面存在非三角形面片: %d\n", size);
            fclose(output);
            return 1;
        }
        get_polygon_interpolation_weights(&point_on_target, size,
                                          polygon_points, weights);
        fwrite(&polygon, sizeof(polygon), 1, output);
        fwrite(weights, sizeof(*weights), 3, output);
    }
    fclose(output);
    return 0;
}
