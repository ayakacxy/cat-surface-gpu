/* SPDX-License-Identifier: GPL-3.0-or-later */

/* 并行计算官方初始旋转的 source/target depth-potential 特征。 */

#define _XOPEN_SOURCE 700

#include <bicpl.h>
#include <CAT_Curvature.h>
#include <CAT_Resample.h>
#include <CAT_SurfaceIO.h>
#include <CAT_SurfUtils.h>
#include <CAT_Smooth.h>

#include <signal.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    int start_idx;
    int end_idx;
    polygons_struct *polygons;
    polygons_struct *source_sphere;
    polygons_struct *resampled_source;
    Point *new_points;
} ResampleThreadArgs;

static double monotonic_seconds(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        return 0.0;
    return (double)value.tv_sec + 1.0e-9 * (double)value.tv_nsec;
}

static void *resample_points(void *argument)
{
    ResampleThreadArgs *args = (ResampleThreadArgs *)argument;
    for (int i = args->start_idx; i < args->end_idx; ++i) {
        Point point_on_sphere;
        Point scaled_point;
        Point polygon_points[MAX_POINTS_PER_POLYGON];
        Point polygon_points_src[MAX_POINTS_PER_POLYGON];
        double weights[MAX_POINTS_PER_POLYGON];
        int polygon = find_closest_polygon_point(
            &args->resampled_source->points[i], args->source_sphere,
            &point_on_sphere);
        int size = get_polygon_points(args->source_sphere, polygon,
                                      polygon_points_src);
        get_polygon_interpolation_weights(&point_on_sphere, size,
                                          polygon_points_src, weights);
        if (get_polygon_points(args->polygons, polygon, polygon_points) != size)
            fprintf(stderr, "map_point_between_polygons\n");
        fill_Point(args->new_points[i], 0.0, 0.0, 0.0);
        for (int corner = 0; corner < size; ++corner) {
            SCALE_POINT(scaled_point, polygon_points[corner],
                        (double)weights[corner]);
            ADD_POINTS(args->new_points[i], args->new_points[i], scaled_point);
        }
    }
    return NULL;
}

/*
 * 复用已经建立的 source sphere bintree，保持官方重采样的半径、查询、
 * 重心累加和法向计算顺序。这个函数只消除重复建树和释放树的开销。
 */
static void resample_spherical_surface_keep_bintree(
    polygons_struct *polygons, polygons_struct *source_sphere,
    polygons_struct *resampled_source, int n_triangles)
{
    const int num_threads = 8;
    double sphere_radius = 0.0;
    double bounds[6];
    Point centre;
    Point *new_points;
    pthread_t threads[num_threads];
    ResampleThreadArgs arguments[num_threads];
    int chunk_size;
    int remainder;

    for (int i = 0; i < source_sphere->n_points; ++i) {
        double radius_squared = 0.0;
        for (int coordinate = 0; coordinate < 3; ++coordinate)
            radius_squared += Point_coord(source_sphere->points[i], coordinate) *
                              Point_coord(source_sphere->points[i], coordinate);
        sphere_radius += sqrt(radius_squared);
    }
    sphere_radius /= source_sphere->n_points;
    get_bounds(source_sphere, bounds);
    fill_Point(centre, bounds[0] + bounds[1], bounds[2] + bounds[3],
               bounds[4] + bounds[5]);
    sphere_radius *= 0.975;
    create_tetrahedral_sphere(&centre, sphere_radius, sphere_radius,
                              sphere_radius, n_triangles, resampled_source);

    new_points = (Point *)malloc(sizeof(*new_points) *
                                 (size_t)resampled_source->n_points);
    if (new_points == NULL) {
        fprintf(stderr, "分配 coarse 重采样缓冲区失败\n");
        exit(EXIT_FAILURE);
    }
    chunk_size = resampled_source->n_points / num_threads;
    remainder = resampled_source->n_points % num_threads;
    for (int thread = 0; thread < num_threads; ++thread) {
        arguments[thread].start_idx = thread * chunk_size;
        arguments[thread].end_idx =
            (thread == num_threads - 1)
                ? (thread + 1) * chunk_size + remainder
                : (thread + 1) * chunk_size;
        arguments[thread].polygons = polygons;
        arguments[thread].source_sphere = source_sphere;
        arguments[thread].resampled_source = resampled_source;
        arguments[thread].new_points = new_points;
        if (pthread_create(&threads[thread], NULL, resample_points,
                           &arguments[thread]) != 0) {
            perror("启动 coarse 重采样线程失败");
            exit(EXIT_FAILURE);
        }
    }
    for (int thread = 0; thread < num_threads; ++thread) {
        if (pthread_join(threads[thread], NULL) != 0) {
            perror("等待 coarse 重采样线程失败");
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < resampled_source->n_points; ++i)
        resampled_source->points[i] = new_points[i];
    compute_polygon_normals(resampled_source);
    free(new_points);
}

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

/* 在 raw depth 模式下同时导出已经完成官方 coarse heat-kernel 的几何，
 * 供 GPU 直接复用，避免另起 geometry probe 重复做同一次重采样和平滑。 */
static int write_geometry_sidecar(const char *feature_path,
                                  const polygons_struct *mapped)
{
    char geometry_path[4096];
    FILE *output;
    int written;

    written = snprintf(geometry_path, sizeof(geometry_path), "%s.geometry",
                       feature_path);
    if (written < 0 || (size_t)written >= sizeof(geometry_path))
        return 1;
    output = fopen(geometry_path, "wb");
    if (output == NULL) {
        perror("打开 raw depth 几何 sidecar 失败");
        return 1;
    }
    if (fwrite(&mapped->n_points, sizeof(mapped->n_points), 1, output) != 1) {
        fclose(output);
        return 1;
    }
    for (int i = 0; i < mapped->n_points; ++i) {
        double point[3] = {
            Point_x(mapped->points[i]),
            Point_y(mapped->points[i]),
            Point_z(mapped->points[i]),
        };
        if (fwrite(point, sizeof(*point), 3, output) != 3) {
            fclose(output);
            return 1;
        }
    }
    if (fclose(output) != 0)
        return 1;
    return 0;
}

static int write_one_feature(const char *surface_path,
                             const char *sphere_path,
                             double heat_fwhm,
                             const char *output_path)
{
    int surface_objects_count = 0;
    int sphere_objects_count = 0;
    object_struct **surface_objects = NULL;
    object_struct **sphere_objects = NULL;
    polygons_struct *surface;
    polygons_struct *sphere;
    polygons_struct mapped_surface = {0};
    double *values = NULL;
    FILE *output = NULL;
    const int n_triangles = 81920;
    const int emit_timing = getenv("FAST_CHARM_ROTATION_TIMING") != NULL;
    const int emit_raw_depth = getenv("FAST_CHARM_ROTATION_RAW_DEPTH") != NULL;
    const int emit_raw_geometry =
        emit_raw_depth &&
        getenv("FAST_CHARM_ROTATION_RAW_DEPTH_GEOMETRY") != NULL;
    double stage_start = monotonic_seconds();
    double after_load;
    double after_normalize;
    double after_resample;
    double after_heat;

    surface = load_polygons(surface_path, &surface_objects_count,
                             &surface_objects);
    sphere = load_polygons(sphere_path, &sphere_objects_count,
                           &sphere_objects);
    if (surface == NULL || sphere == NULL)
        return 1;
    after_load = monotonic_seconds();

    /* 保持官方 source/target sphere 的质心平移和逐点单位化。 */
    translate_to_center_of_mass(sphere);
    for (int i = 0; i < sphere->n_points; ++i)
        set_vector_length(&sphere->points[i], 1.0);
    after_normalize = monotonic_seconds();

    /* 特征只读取 mapped_surface；source sphere 的 coarse 几何由 stencil
       helper 单独生成，这里不再重复创建一份不会被读取的 mapped_sphere。 */
    create_polygons_bintree(sphere, ROUND((float)sphere->n_items * 0.5));
    resample_spherical_surface_keep_bintree(surface, sphere, &mapped_surface,
                                             n_triangles);
    delete_the_bintree(&sphere->bintree);
    after_resample = monotonic_seconds();
    smooth_heatkernel(&mapped_surface, NULL, heat_fwhm);
    after_heat = monotonic_seconds();
    if (emit_raw_geometry &&
        write_geometry_sidecar(output_path, &mapped_surface) != 0) {
        fprintf(stderr, "写入 raw depth 几何 sidecar 失败\n");
        return 1;
    }

    values = (double *)malloc(sizeof(*values) * mapped_surface.n_points);
    if (values == NULL) {
        fprintf(stderr, "分配单侧初始旋转特征失败\n");
        return 1;
    }
    if (emit_raw_depth) {
        int *n_neighbours = NULL;
        int **neighbours = NULL;
        /* 只导出官方 depth-potential 的未平滑值；50 mm heat-kernel
           平滑由 GPU 后端接管，保留 depth-potential 线性系统本身。 */
        get_all_polygon_point_neighbours(
            &mapped_surface, &n_neighbours, &neighbours);
        get_polygon_vertex_curvatures_cg(
            &mapped_surface, n_neighbours, neighbours, 0.0, 1000, values);
        free(n_neighbours);
        if (neighbours) {
            free(neighbours[0]);
            free(neighbours);
        }
    } else {
        get_smoothed_curvatures(&mapped_surface, values, 50.0, 1000);
    }

    if (emit_timing) {
        fprintf(stderr,
                "CAT_ROT_TIMING side=%s load=%.6f normalize=%.6f resample=%.6f heat=%.6f depth=%.6f total=%.6f\n",
                surface_path,
                after_load - stage_start,
                after_normalize - after_load,
                after_resample - after_load,
                after_heat - after_resample,
                monotonic_seconds() - after_heat,
                monotonic_seconds() - stage_start);
    }

    output = fopen(output_path, "wb");
    if (output == NULL) {
        perror("打开单侧初始旋转特征输出失败");
        free(values);
        return 1;
    }
    if (fwrite(&mapped_surface.n_points, sizeof(mapped_surface.n_points), 1,
               output) != 1 ||
        fwrite(values, sizeof(*values), mapped_surface.n_points, output) !=
            (size_t)mapped_surface.n_points) {
        fprintf(stderr, "写入单侧初始旋转特征失败\n");
        fclose(output);
        free(values);
        return 1;
    }
    fclose(output);
    free(values);
    return 0;
}

static int make_temp_file(char *path, size_t path_size)
{
    int file_descriptor;
    const char *temp_directory = getenv("TMPDIR");
    if (temp_directory == NULL || temp_directory[0] == '\0')
        temp_directory = P_tmpdir;
    if (snprintf(path, path_size, "%s/fast_charm_rotation_side_XXXXXX",
                 temp_directory) >= (int)path_size)
        return -1;
    file_descriptor = mkstemp(path);
    if (file_descriptor < 0)
        return -1;
    close(file_descriptor);
    return 0;
}

static int read_feature(const char *path, int *count, double **values)
{
    FILE *input;
    double *buffer;

    input = fopen(path, "rb");
    if (input == NULL || fread(count, sizeof(*count), 1, input) != 1 ||
        *count <= 0) {
        if (input != NULL)
            fclose(input);
        return 1;
    }
    buffer = (double *)malloc(sizeof(*buffer) * *count);
    if (buffer == NULL ||
        fread(buffer, sizeof(*buffer), *count, input) != (size_t)*count) {
        free(buffer);
        fclose(input);
        return 1;
    }
    fclose(input);
    *values = buffer;
    return 0;
}

static int wait_child(pid_t pid, const char *label)
{
    int status;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0) {
        fprintf(stderr, "%s 初始旋转特征子进程失败\n", label);
        return 1;
    }
    return 0;
}

static int combine_features(const char *source_path,
                            const char *target_path,
                            const char *output_path)
{
    int source_count = 0;
    int target_count = 0;
    double *source_values = NULL;
    double *target_values = NULL;
    FILE *output = NULL;
    int result = 1;

    if (read_feature(source_path, &source_count, &source_values) != 0 ||
        read_feature(target_path, &target_count, &target_values) != 0) {
        fprintf(stderr, "读取 source/target 特征中间文件失败\n");
        goto cleanup;
    }
    output = fopen(output_path, "wb");
    if (output == NULL) {
        perror("打开合并特征输出失败");
        goto cleanup;
    }
    if (fwrite(&source_count, sizeof(source_count), 1, output) != 1 ||
        fwrite(&target_count, sizeof(target_count), 1, output) != 1 ||
        fwrite(source_values, sizeof(*source_values), source_count, output) !=
            (size_t)source_count ||
        fwrite(target_values, sizeof(*target_values), target_count, output) !=
            (size_t)target_count) {
        fprintf(stderr, "写入合并特征失败\n");
        goto cleanup;
    }
    result = 0;

cleanup:
    if (output != NULL)
        fclose(output);
    free(source_values);
    free(target_values);
    return result;
}

static int publish_geometry_sidecars(const char *source_path,
                                     const char *target_path,
                                     const char *output_path)
{
    char source_geometry[4096];
    char target_geometry[4096];
    char output_source_geometry[4096];
    char output_target_geometry[4096];
    int source_written;
    int target_written;
    int output_source_written;
    int output_target_written;

    source_written = snprintf(source_geometry, sizeof(source_geometry),
                              "%s.geometry", source_path);
    target_written = snprintf(target_geometry, sizeof(target_geometry),
                              "%s.geometry", target_path);
    output_source_written = snprintf(
        output_source_geometry, sizeof(output_source_geometry),
        "%s.source.geometry", output_path);
    output_target_written = snprintf(
        output_target_geometry, sizeof(output_target_geometry),
        "%s.target.geometry", output_path);
    if (source_written < 0 ||
        (size_t)source_written >= sizeof(source_geometry) ||
        target_written < 0 ||
        (size_t)target_written >= sizeof(target_geometry) ||
        output_source_written < 0 ||
        (size_t)output_source_written >= sizeof(output_source_geometry) ||
        output_target_written < 0 ||
        (size_t)output_target_written >= sizeof(output_target_geometry))
        return 1;
    if (rename(source_geometry, output_source_geometry) != 0 ||
        rename(target_geometry, output_target_geometry) != 0) {
        perror("发布 raw depth 几何 sidecar 失败");
        return 1;
    }
    return 0;
}

static int run_parallel(const char *source_path,
                        const char *source_sphere_path,
                        const char *target_path,
                        const char *target_sphere_path,
                        const char *output_path)
{
    char source_tmp[64];
    char target_tmp[64];
    pid_t source_pid;
    pid_t target_pid;
    int result = 1;

    if (make_temp_file(source_tmp, sizeof(source_tmp)) != 0 ||
        make_temp_file(target_tmp, sizeof(target_tmp)) != 0) {
        fprintf(stderr, "创建并行特征临时文件失败\n");
        return 1;
    }

    source_pid = fork();
    if (source_pid < 0) {
        perror("启动 source 特征子进程失败");
        goto cleanup;
    }
    if (source_pid == 0)
        _exit(write_one_feature(source_path, source_sphere_path, 15.0,
                                source_tmp));

    target_pid = fork();
    if (target_pid < 0) {
        perror("启动 target 特征子进程失败");
        kill(source_pid, SIGTERM);
        waitpid(source_pid, NULL, 0);
        goto cleanup;
    }
    if (target_pid == 0)
        _exit(write_one_feature(target_path, target_sphere_path, 10.0,
                                target_tmp));

    if (wait_child(source_pid, "source") != 0 ||
        wait_child(target_pid, "target") != 0)
        goto cleanup;
    result = combine_features(source_tmp, target_tmp, output_path);
    if (result == 0 &&
        getenv("FAST_CHARM_ROTATION_RAW_DEPTH_GEOMETRY") != NULL)
        result = publish_geometry_sidecars(source_tmp, target_tmp, output_path);

cleanup:
    unlink(source_tmp);
    unlink(target_tmp);
    {
        char sidecar_path[4096];
        if (snprintf(sidecar_path, sizeof(sidecar_path), "%s.geometry",
                     source_tmp) >= 0)
            unlink(sidecar_path);
        if (snprintf(sidecar_path, sizeof(sidecar_path), "%s.geometry",
                     target_tmp) >= 0)
            unlink(sidecar_path);
    }
    return result;
}

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr,
                "用法: %s SOURCE SOURCE_SPHERE TARGET TARGET_SPHERE OUTPUT\n",
                argv[0]);
        return 2;
    }
    return run_parallel(argv[1], argv[2], argv[3], argv[4], argv[5]);
}
