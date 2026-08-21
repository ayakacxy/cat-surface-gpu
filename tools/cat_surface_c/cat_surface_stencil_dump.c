/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 为一个输入球面预计算 CAT 球面重采样和 sheet 映射的三角形 stencil。 */

#include <bicpl.h>
#include <CAT_Map.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#ifndef STENCIL_THREADS
#define STENCIL_THREADS 8
#endif

typedef struct {
    object_struct **objects;
    int n_objects;
    polygons_struct *polygons;
} LoadedPolygons;

typedef struct {
    int32_t magic;
    int32_t version;
    int32_t n_points;
    int32_t n_triangles;
    int32_t n_sheet;
    int32_t nx;
    int32_t ny;
    int32_t source_points;
} StencilHeader;

typedef struct {
    polygons_struct *source_sphere;
    polygons_struct *surface;
    polygons_struct *output_sphere;
    int *indices;
    double *weights;
    int start;
    int end;
    int invalid;
} SurfaceStencilThreadArgs;

typedef struct {
    polygons_struct *unit_sphere;
    int *indices;
    double *weights;
    int nx;
    int ny;
    int start;
    int end;
    int invalid;
} SheetStencilThreadArgs;

typedef struct {
    polygons_struct *polygons;
    polygons_struct *unit_sphere;
    Point *points;
    int start;
    int end;
} UnitSphereThreadArgs;

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
        fprintf(stderr, "无法读取单个球面对象: %s\n", path);
        exit(EXIT_FAILURE);
    }
    loaded.polygons = get_polygons_ptr(loaded.objects[0]);
    return loaded;
}

static void *build_surface_stencil_chunk(void *argument)
{
    SurfaceStencilThreadArgs *args = (SurfaceStencilThreadArgs *)argument;
    polygons_struct *index_surface =
        args->surface == NULL ? args->source_sphere : args->surface;
    for (int i = args->start; i < args->end; ++i) {
        Point point_on_sphere;
        Point polygon_points[MAX_POINTS_PER_POLYGON];
        double polygon_weights[MAX_POINTS_PER_POLYGON];
        int polygon = find_closest_polygon_point(
            &args->output_sphere->points[i], args->source_sphere,
            &point_on_sphere);
        int size = get_polygon_points(args->source_sphere, polygon,
                                      polygon_points);
        if (size != 3) {
            args->invalid = 1;
            continue;
        }
        get_polygon_interpolation_weights(&point_on_sphere, size,
                                          polygon_points, polygon_weights);
        for (int corner = 0; corner < 3; ++corner) {
            int offset = 3 * i + corner;
            /* 权重来自 sphere；surface 索引沿用官方同面片 corner 顺序。 */
            args->indices[offset] = index_surface->indices[
                POINT_INDEX(index_surface->end_indices, polygon, corner)];
            args->weights[offset] = polygon_weights[corner];
        }
    }
    return NULL;
}

static void *build_sheet_stencil_chunk(void *argument)
{
    SheetStencilThreadArgs *args = (SheetStencilThreadArgs *)argument;
    for (int index = args->start; index < args->end; ++index) {
        Point unit_point;
        Point point_on_sphere;
        Point polygon_points[MAX_POINTS_PER_POLYGON];
        double polygon_weights[MAX_POINTS_PER_POLYGON];
        int x = index % args->nx;
        int y = index / args->nx;
        double u = ((double)x + 0.5) / (double)args->nx;
        double v = ((double)y + 0.5) / (double)args->ny;
        int polygon;
        int size;

        uv_to_point(u, v, &unit_point);
        polygon = find_closest_polygon_point(&unit_point, args->unit_sphere,
                                             &point_on_sphere);
        size = get_polygon_points(args->unit_sphere, polygon, polygon_points);
        if (size != 3) {
            args->invalid = 1;
            continue;
        }
        get_polygon_interpolation_weights(&point_on_sphere, size,
                                          polygon_points, polygon_weights);
        for (int corner = 0; corner < 3; ++corner) {
            int offset = 3 * index + corner;
            args->indices[offset] = args->unit_sphere->indices[
                POINT_INDEX(args->unit_sphere->end_indices, polygon, corner)];
            args->weights[offset] = polygon_weights[corner];
        }
    }
    return NULL;
}

static void *write_unit_sphere_points_chunk(void *argument)
{
    UnitSphereThreadArgs *args = (UnitSphereThreadArgs *)argument;
    for (int i = args->start; i < args->end; ++i) {
        Point unit_point;
        map_point_to_unit_sphere(args->polygons, &args->polygons->points[i],
                                 args->unit_sphere, &unit_point);
        if (isnan(Point_x(unit_point)) || isnan(Point_y(unit_point)) ||
            isnan(Point_z(unit_point)))
            unit_point = args->unit_sphere->points[i];
        args->points[i] = unit_point;
    }
    return NULL;
}

static void build_surface_stencil(polygons_struct *source_sphere,
                                  polygons_struct *surface,
                                  polygons_struct *output_sphere,
                                  int **indices_output,
                                  double **weights_output)
{
    int *indices = (int *)malloc(sizeof(*indices) * 3 * output_sphere->n_points);
    double *weights = (double *)malloc(sizeof(*weights) * 3 * output_sphere->n_points);
    pthread_t threads[STENCIL_THREADS];
    SurfaceStencilThreadArgs arguments[STENCIL_THREADS];
    int invalid = 0;
    int chunk_size;
    int remainder;

    if (indices == NULL || weights == NULL) {
        fprintf(stderr, "分配球面重采样 stencil 失败\n");
        exit(EXIT_FAILURE);
    }
    create_polygons_bintree(source_sphere,
                            ROUND((double)source_sphere->n_items * 0.5));
    chunk_size = output_sphere->n_points / STENCIL_THREADS;
    remainder = output_sphere->n_points % STENCIL_THREADS;
    for (int thread = 0; thread < STENCIL_THREADS; ++thread) {
        arguments[thread].source_sphere = source_sphere;
        arguments[thread].surface = surface;
        arguments[thread].output_sphere = output_sphere;
        arguments[thread].indices = indices;
        arguments[thread].weights = weights;
        arguments[thread].start = thread * chunk_size;
        arguments[thread].end = (thread == STENCIL_THREADS - 1)
                                    ? (thread + 1) * chunk_size + remainder
                                    : (thread + 1) * chunk_size;
        arguments[thread].invalid = 0;
        if (pthread_create(&threads[thread], NULL,
                           build_surface_stencil_chunk, &arguments[thread]) != 0)
            exit(EXIT_FAILURE);
    }
    for (int thread = 0; thread < STENCIL_THREADS; ++thread)
        pthread_join(threads[thread], NULL);
    for (int thread = 0; thread < STENCIL_THREADS; ++thread)
        invalid |= arguments[thread].invalid;
    if (invalid) {
        fprintf(stderr, "输入球面存在非三角形面片\n");
        exit(EXIT_FAILURE);
    }
    delete_the_bintree(&source_sphere->bintree);
    *indices_output = indices;
    *weights_output = weights;
}

static void write_sheet_stencil(FILE *output,
                                polygons_struct *output_sphere,
                                int nx,
                                int ny)
{
    int n_sheet = nx * ny;
    int *indices = (int *)malloc(sizeof(*indices) * 3 * n_sheet);
    double *weights = (double *)malloc(sizeof(*weights) * 3 * n_sheet);
    polygons_struct unit_sphere;
    pthread_t threads[STENCIL_THREADS];
    SheetStencilThreadArgs arguments[STENCIL_THREADS];
    int invalid = 0;
    int chunk_size;
    int remainder;

    if (indices == NULL || weights == NULL) {
        fprintf(stderr, "分配 sheet stencil 失败\n");
        exit(EXIT_FAILURE);
    }
    copy_polygons(output_sphere, &unit_sphere);
    for (int i = 0; i < unit_sphere.n_points; ++i)
        set_vector_length(&unit_sphere.points[i], 1.0);
    create_polygons_bintree(&unit_sphere,
                            ROUND((double)unit_sphere.n_items * 0.5));
    chunk_size = n_sheet / STENCIL_THREADS;
    remainder = n_sheet % STENCIL_THREADS;
    for (int thread = 0; thread < STENCIL_THREADS; ++thread) {
        arguments[thread].unit_sphere = &unit_sphere;
        arguments[thread].indices = indices;
        arguments[thread].weights = weights;
        arguments[thread].nx = nx;
        arguments[thread].ny = ny;
        arguments[thread].start = thread * chunk_size;
        arguments[thread].end = (thread == STENCIL_THREADS - 1)
                                    ? (thread + 1) * chunk_size + remainder
                                    : (thread + 1) * chunk_size;
        arguments[thread].invalid = 0;
        if (pthread_create(&threads[thread], NULL,
                           build_sheet_stencil_chunk, &arguments[thread]) != 0)
            exit(EXIT_FAILURE);
    }
    for (int thread = 0; thread < STENCIL_THREADS; ++thread)
        pthread_join(threads[thread], NULL);
    for (int thread = 0; thread < STENCIL_THREADS; ++thread)
        invalid |= arguments[thread].invalid;
    if (invalid) {
        fprintf(stderr, "输出球面存在非三角形面片\n");
        exit(EXIT_FAILURE);
    }
    delete_the_bintree(&unit_sphere.bintree);
    delete_polygons(&unit_sphere);
    fwrite(indices, sizeof(*indices), 3 * n_sheet, output);
    fwrite(weights, sizeof(*weights), 3 * n_sheet, output);
    free(indices);
    free(weights);
}

static void write_unit_sphere_points(FILE *output, polygons_struct *polygons)
{
    polygons_struct unit_sphere;
    Point *points = (Point *)malloc(sizeof(*points) * polygons->n_points);
    pthread_t threads[STENCIL_THREADS];
    UnitSphereThreadArgs arguments[STENCIL_THREADS];
    int chunk_size;
    int remainder;

    if (points == NULL)
        exit(EXIT_FAILURE);

    /* 这一步复现 CAT_Warp.c 的 map_point_to_unit_sphere，避免最终 warp
       阶段每次重新构建 bintree 和重复做三角形定位。 */
    copy_polygons(polygons, &unit_sphere);
    for (int i = 0; i < unit_sphere.n_points; ++i)
        set_vector_length(&unit_sphere.points[i], 1.0);
    create_polygons_bintree(polygons,
                            ROUND((double)polygons->n_items * 0.5));
    chunk_size = polygons->n_points / STENCIL_THREADS;
    remainder = polygons->n_points % STENCIL_THREADS;
    for (int thread = 0; thread < STENCIL_THREADS; ++thread) {
        arguments[thread].polygons = polygons;
        arguments[thread].unit_sphere = &unit_sphere;
        arguments[thread].points = points;
        arguments[thread].start = thread * chunk_size;
        arguments[thread].end = (thread == STENCIL_THREADS - 1)
                                    ? (thread + 1) * chunk_size + remainder
                                    : (thread + 1) * chunk_size;
        if (pthread_create(&threads[thread], NULL,
                           write_unit_sphere_points_chunk, &arguments[thread]) != 0)
            exit(EXIT_FAILURE);
    }
    for (int thread = 0; thread < STENCIL_THREADS; ++thread)
        pthread_join(threads[thread], NULL);
    fwrite(points, sizeof(*points), polygons->n_points, output);
    free(points);
    delete_the_bintree(&polygons->bintree);
    delete_polygons(&unit_sphere);
}

int main(int argc, char **argv)
{
    LoadedPolygons loaded;
    LoadedPolygons loaded_surface = {0};
    polygons_struct output_sphere;
    polygons_struct resampled_sphere;
    Point centre;
    double radius = 0.0;
    double bounds[6];
    StencilHeader header;
    FILE *output;
    int *surface_indices = NULL;
    double *surface_weights = NULL;
    const int n_triangles = 81920;
    const int nx = 512;
    const int ny = 256;
    int normalise_input = 1;
    const char *surface_path = NULL;

    for (int argument = 3; argument < argc; ++argument) {
        if (strcmp(argv[argument], "--no-normalize") == 0) {
            normalise_input = 0;
        } else if (strcmp(argv[argument], "--surface") == 0 &&
                   argument + 1 < argc) {
            surface_path = argv[++argument];
        } else {
            fprintf(stderr, "未知 stencil 选项: %s\n", argv[argument]);
            return 2;
        }
    }
    if (argc < 3) {
        fprintf(stderr,
                "用法: %s INPUT_SPHERE OUTPUT_STENCIL [--no-normalize] [--surface SURFACE]\n",
                argv[0]);
        return 2;
    }
    loaded = load_polygons(argv[1]);
    if (surface_path != NULL) {
        loaded_surface = load_polygons(surface_path);
        if (loaded_surface.polygons->n_points != loaded.polygons->n_points ||
            loaded_surface.polygons->n_items != loaded.polygons->n_items) {
            fprintf(stderr, "surface 与 sphere 的点数或面片数不一致\n");
            return 1;
        }
    }
    if (normalise_input)
        translate_to_center_of_mass(loaded.polygons);
    for (int i = 0; i < loaded.polygons->n_points; ++i) {
        if (normalise_input)
            set_vector_length(&loaded.polygons->points[i], 1.0);
        radius += sqrt(
            Point_x(loaded.polygons->points[i]) * Point_x(loaded.polygons->points[i]) +
            Point_y(loaded.polygons->points[i]) * Point_y(loaded.polygons->points[i]) +
            Point_z(loaded.polygons->points[i]) * Point_z(loaded.polygons->points[i]));
    }
    radius = 0.975 * radius / (double)loaded.polygons->n_points;
    get_bounds(loaded.polygons, bounds);
    fill_Point(centre, bounds[0] + bounds[1], bounds[2] + bounds[3],
               bounds[4] + bounds[5]);
    create_tetrahedral_sphere(&centre, radius, radius, radius, n_triangles,
                              &output_sphere);
    /* 先建立 surface stencil，再生成 sphere->sphere 重采样；这与
       CAT_SurfWarpSolveDartelFlow 中 source/target 的调用顺序一致，
       也避免 bintree 边界候选在极少数几何 tie 上改变。 */
    build_surface_stencil(
        loaded.polygons,
        surface_path == NULL ? NULL : loaded_surface.polygons,
        &output_sphere,
        &surface_indices,
        &surface_weights);
    resampled_sphere = (polygons_struct){0};
    resample_spherical_surface(loaded.polygons, loaded.polygons,
                               &resampled_sphere, NULL, NULL, n_triangles);

    output = fopen(argv[2], "wb");
    if (output == NULL) {
        perror("打开 stencil 输出失败");
        return 1;
    }
    header.magic = 0x46534354;
    header.version = 2;
    header.n_points = output_sphere.n_points;
    header.n_triangles = output_sphere.n_items;
    header.n_sheet = nx * ny;
    header.nx = nx;
    header.ny = ny;
    header.source_points = surface_path == NULL
                               ? loaded.polygons->n_points
                               : loaded_surface.polygons->n_points;
    fwrite(&header, sizeof(header), 1, output);
    for (int i = 0; i < resampled_sphere.n_points; ++i) {
        double point[3] = {
            Point_x(resampled_sphere.points[i]),
            Point_y(resampled_sphere.points[i]),
            Point_z(resampled_sphere.points[i]),
        };
        fwrite(point, sizeof(*point), 3, output);
    }
    for (int polygon = 0; polygon < output_sphere.n_items; ++polygon) {
        int face[3];
        for (int corner = 0; corner < 3; ++corner)
            face[corner] = output_sphere.indices[
                POINT_INDEX(output_sphere.end_indices, polygon, corner)];
        fwrite(face, sizeof(*face), 3, output);
    }
    fwrite(surface_indices, sizeof(*surface_indices),
           3 * output_sphere.n_points, output);
    fwrite(surface_weights, sizeof(*surface_weights),
           3 * output_sphere.n_points, output);
    free(surface_indices);
    free(surface_weights);
    write_sheet_stencil(output, &resampled_sphere, nx, ny);
    write_unit_sphere_points(output, loaded.polygons);
    fclose(output);
    return 0;
}
