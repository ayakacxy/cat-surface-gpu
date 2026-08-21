/* SPDX-License-Identifier: GPL-3.0-or-later */

/*
 * 对最新版 CAT-Surface 的 CAT_Surf2Sphere 阶段进行可复现计时。
 *
 * 这个程序只复用公开 API 重新编排官方 surf_to_sphere 的阶段，
 * 不改变每个阶段的参数、迭代次数或调用顺序。
 */

#include <bicpl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "CAT_Surf.h"
#include "CAT_SurfaceIO.h"

static double
now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void
print_stage(const char *name, double start, double end)
{
    fprintf(stderr, "CAT_Surf2Sphere stage %-16s %.6f s\n",
            name, end - start);
}

static void
usage(const char *executable)
{
    fprintf(stderr,
            "用法: %s surface_file output_surface_file [stop_at]\n",
            executable);
}

int
main(int argc, char **argv)
{
    char *input_file;
    char *output_file;
    int n_objects;
    int stop_at = 10;
    int finger_smoothing_iters = 0;
    int areal_smoothing_iters;
    int verbose = 0;
    File_formats format;
    object_struct **object_list;
    polygons_struct *polygons;
    double surface_area;
    double factor;
    double total_start;
    double stage_start;
    double stage_end;

    if (argc < 3 || argc > 4) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    input_file = argv[1];
    output_file = argv[2];
    if (argc == 4)
        stop_at = atoi(argv[3]);
    if (stop_at < 1) {
        fprintf(stderr, "stop_at 必须大于等于 1。\n");
        return EXIT_FAILURE;
    }

    stage_start = now_seconds();
    if (input_graphics_any_format(input_file, &format, &n_objects,
                                  &object_list) != OK || n_objects != 1 ||
        get_object_type(object_list[0]) != POLYGONS) {
        fprintf(stderr, "读取表面失败: %s\n", input_file);
        return EXIT_FAILURE;
    }
    stage_end = now_seconds();
    print_stage("input_read", stage_start, stage_end);

    polygons = get_polygons_ptr(object_list[0]);
    fprintf(stderr, "CAT_Surf2Sphere mesh points=%d polygons=%d stop_at=%d\n",
            polygons->n_points, polygons->n_items, stop_at);

    surface_area = get_polygons_surface_area(polygons);
    factor = 1.0;
    if (polygons->n_items > 350000)
        factor = (double)polygons->n_items / 350000.0;

    total_start = now_seconds();

    stage_start = now_seconds();
    inflate_surface_and_smooth_fingers(polygons, 1, 0.2,
                                       round(factor * 50), 1.0, 3.0,
                                       1.0, 0);
    stage_end = now_seconds();
    print_stage("low_smooth", stage_start, stage_end);

    if (stop_at > 1) {
        stage_start = now_seconds();
        finger_smoothing_iters = 30;
        inflate_surface_and_smooth_fingers(polygons, 2, 1.0,
                                           round(factor * 30), 1.4, 3.0,
                                           1.0, finger_smoothing_iters);
        stage_end = now_seconds();
        print_stage("inflate", stage_start, stage_end);
    }

    if (stop_at > 2) {
        stage_start = now_seconds();
        inflate_surface_and_smooth_fingers(polygons, 4, 1.0,
                                           round(factor * 30), 1.1, 3.0,
                                           1.0, 0);
        stage_end = now_seconds();
        print_stage("very_inflate", stage_start, stage_end);
    }

    if (stop_at > 3) {
        stage_start = now_seconds();
        finger_smoothing_iters = 60;
        inflate_surface_and_smooth_fingers(polygons, 6, 1.0,
                                           round(factor * 60), 1.6, 3.0,
                                           1.0, finger_smoothing_iters);
        stage_end = now_seconds();
        print_stage("high_smooth", stage_start, stage_end);
    }

    if (stop_at > 4) {
        stage_start = now_seconds();
        inflate_surface_and_smooth_fingers(polygons, 6, 1.0,
                                           round(factor * 50), 1.4, 4.0,
                                           1.0, finger_smoothing_iters);
        convert_ellipsoid_to_sphere_with_surface_area(polygons, surface_area);
        stage_end = now_seconds();
        print_stage("ellipsoid", stage_start, stage_end);
    }

    if (stop_at > 5) {
        stage_start = now_seconds();
        areal_smoothing_iters = 1000 * (stop_at - 5);
        areal_smoothing(polygons, 1.0, areal_smoothing_iters, 1, NULL, 1000);
        convert_ellipsoid_to_sphere_with_surface_area(polygons, surface_area);
        stage_end = now_seconds();
        print_stage("areal_smoothing", stage_start, stage_end);
    }

    stage_start = now_seconds();
    compute_polygon_normals(polygons);
    stage_end = now_seconds();
    print_stage("normals", stage_start, stage_end);
    print_stage("solver_total", total_start, now_seconds());

    stage_start = now_seconds();
    if (output_graphics_any_format(output_file, format, 1, object_list,
                                   NULL) != OK) {
        fprintf(stderr, "写出表面失败: %s\n", output_file);
        delete_object_list(n_objects, object_list);
        return EXIT_FAILURE;
    }
    stage_end = now_seconds();
    print_stage("output_write", stage_start, stage_end);

    delete_object_list(n_objects, object_list);
    (void)verbose;
    return EXIT_SUCCESS;
}
