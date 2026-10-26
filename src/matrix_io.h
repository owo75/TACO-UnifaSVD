#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#include <thrust/device_vector.h>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <exception>
#include <cuda_runtime.h>

enum class MatrixFileFormat { Text = 0, Binary = 1, LibSVM = 2 };
enum class MatrixFileLayout { RowMajor = 0, ColumnMajor = 1 };

// A LIBSVM label is ignored; each remaining token must be index:value.
// Report malformed data instead of allowing conversion exceptions to escape.
inline bool parse_libsvm_feature(const std::string& token, const std::string& path,
                                long int& feature, double& value)
{
    const size_t colon = token.find(':');
    if (colon != std::string::npos && colon > 0 && colon + 1 < token.size()) {
        try {
            size_t index_end = 0, value_end = 0;
            const std::string index_text = token.substr(0, colon);
            const std::string value_text = token.substr(colon + 1);
            feature = std::stol(index_text, &index_end);
            value = std::stod(value_text, &value_end);
            if (index_end == index_text.size() && value_end == value_text.size() &&
                feature > 0 && std::isfinite(value)) return true;
        } catch (const std::exception&) {
            // Invalid and out-of-range numeric fields share the same diagnostic.
        }
    }
    std::fprintf(stderr, "[Error] Invalid LIBSVM token '%s' in '%s'.\n",
                 token.c_str(), path.c_str());
    return false;
}

template<typename T>
inline bool read_matrix_text(const std::string& path, size_t expected_elems, std::vector<T>& values)
{
    std::ifstream ifs(path);
    if (!ifs) {
        std::printf("[Error] Failed to open matrix file '%s' in text mode.\n", path.c_str());
        return false;
    }
    values.clear();
    values.reserve(expected_elems);
    T val;
    while (ifs >> val) {
        values.push_back(val);
    }
    if (!ifs.eof()) {
        std::printf("[Error] Parsing matrix file '%s' failed (non-numeric token encountered).\n", path.c_str());
        return false;
    }
    if (values.size() != expected_elems) {
        std::printf("[Error] Matrix file '%s' contains %zu values, but %zu are required.\n",
                   path.c_str(), values.size(), expected_elems);
        return false;
    }
    return true;
}

inline bool infer_text_matrix_shape(const std::string& path,
                                    MatrixFileLayout layout,
                                    long int& rows,
                                    long int& cols)
{
    std::ifstream ifs(path);
    if (!ifs) {
        std::printf("[Error] Failed to open matrix file '%s' while inferring shape.\n", path.c_str());
        return false;
    }

    rows = 0;
    cols = 0;
    long int nonempty_lines = 0;
    long int values_per_line = -1;
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        double tmp = 0.0;
        long int count = 0;
        while (iss >> tmp) ++count;
        if (!iss.eof()) {
            std::printf("[Error] Parsing matrix file '%s' failed while inferring shape (non-numeric token encountered).\n",
                        path.c_str());
            return false;
        }
        if (count == 0) continue;
        if (values_per_line < 0) {
            values_per_line = count;
        } else if (count != values_per_line) {
            std::printf("[Error] Matrix file '%s' has inconsistent text row lengths (%ld vs %ld).\n",
                        path.c_str(), values_per_line, count);
            return false;
        }
        ++nonempty_lines;
    }

    if (nonempty_lines <= 0 || values_per_line <= 0) {
        std::printf("[Error] Matrix file '%s' does not contain a non-empty rectangular text matrix.\n", path.c_str());
        return false;
    }

    if (layout == MatrixFileLayout::RowMajor) {
        rows = nonempty_lines;
        cols = values_per_line;
    } else {
        rows = values_per_line;
        cols = nonempty_lines;
    }
    return true;
}

inline bool infer_libsvm_shape(const std::string& path,
                               long int& rows,
                               long int& cols)
{
    std::ifstream ifs(path);
    if (!ifs) {
        std::printf("[Error] Failed to open LIBSVM file '%s' while inferring shape.\n", path.c_str());
        return false;
    }

    rows = 0;
    cols = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        std::string token;
        bool saw_anything = false;
        bool skipped_label = false;
        while (iss >> token) {
            saw_anything = true;
            if (!skipped_label) {
                skipped_label = true;
                continue;
            }
            long int feat = 0;
            double value = 0.0;
            if (!parse_libsvm_feature(token, path, feat, value)) return false;
            if (feat > cols) cols = feat;
        }
        if (!iss.eof()) {
            std::printf("[Error] Parsing LIBSVM file '%s' failed while inferring shape.\n", path.c_str());
            return false;
        }
        if (saw_anything) ++rows;
    }

    if (rows <= 0 || cols <= 0) {
        std::printf("[Error] LIBSVM file '%s' does not contain any valid samples/features.\n", path.c_str());
        return false;
    }
    return true;
}

template<typename T>
inline bool read_matrix_libsvm(const std::string& path,
                               long int rows_input,
                               long int rows_padded,
                               long int cols,
                               std::vector<T>& host_col)
{
    std::ifstream ifs(path);
    if (!ifs) {
        std::printf("[Error] Failed to open LIBSVM file '%s'.\n", path.c_str());
        return false;
    }
    host_col.assign(static_cast<size_t>(rows_padded) * static_cast<size_t>(cols), T(0));

    std::string line;
    long int row = 0;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        std::string token;
        if (!(iss >> token)) continue;  // Match shape inference: skip blank lines.
        if (row >= rows_input) {
            std::printf("[Error] LIBSVM file '%s' has more rows than expected (%ld).\n", path.c_str(), rows_input);
            return false;
        }
        while (iss >> token) {
            long int feat = 0;
            double value = 0.0;
            if (!parse_libsvm_feature(token, path, feat, value)) return false;
            if (feat <= 0 || feat > cols) {
                std::printf("[Error] Feature index %ld out of range [1, %ld] in '%s'.\n", feat, cols, path.c_str());
                return false;
            }
            host_col[(size_t)(feat - 1) * (size_t)rows_padded + (size_t)row] = static_cast<T>(value);
        }
        if (!iss.eof()) {
            std::printf("[Error] Parsing LIBSVM file '%s' failed.\n", path.c_str());
            return false;
        }
        ++row;
    }

    if (row != rows_input) {
        std::printf("[Error] LIBSVM file '%s' has %ld rows, but %ld are required.\n", path.c_str(), row, rows_input);
        return false;
    }
    return true;
}

template<typename T>
inline bool read_matrix_binary(const std::string& path, size_t expected_elems, std::vector<T>& values)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::printf("[Error] Failed to open matrix file '%s' in binary mode.\n", path.c_str());
        return false;
    }
    values.resize(expected_elems);
    ifs.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected_elems * sizeof(T)));
    if (ifs.gcount() != static_cast<std::streamsize>(expected_elems * sizeof(T))) {
        std::printf("[Error] Matrix file '%s' provides %lld bytes, but %zu bytes are required.\n",
                   path.c_str(), static_cast<long long>(ifs.gcount()), expected_elems * sizeof(T));
        return false;
    }
    return true;
}

template<typename T>
inline bool load_matrix_from_file(cudaStream_t stream,
                                  long int rows_input,
                                  long int rows_padded,
                                  long int cols,
                                  const std::string& path,
                                  MatrixFileFormat format,
                                  MatrixFileLayout layout,
                                  thrust::device_vector<T>& d_out)
{
    if (rows_input <= 0 || cols <= 0 || rows_padded < rows_input) {
        std::printf("[Error] Invalid matrix dimensions when loading user matrix (rows=%ld, padded_rows=%ld, cols=%ld).\n",
                   rows_input, rows_padded, cols);
        return false;
    }
    const size_t expected = static_cast<size_t>(rows_input) * static_cast<size_t>(cols);
    std::vector<T> host_raw;
    const long int padded = rows_padded;
    std::vector<T> host_col;
    if (format == MatrixFileFormat::Text || format == MatrixFileFormat::Binary) {
        bool ok = (format == MatrixFileFormat::Text)
                      ? read_matrix_text(path, expected, host_raw)
                      : read_matrix_binary(path, expected, host_raw);
        if (!ok) {
            return false;
        }
        host_col.assign(static_cast<size_t>(padded) * static_cast<size_t>(cols), T(0));
        if (layout == MatrixFileLayout::ColumnMajor) {
            for (long int col = 0; col < cols; ++col) {
                const size_t src_off = static_cast<size_t>(col) * static_cast<size_t>(rows_input);
                const size_t dst_off = static_cast<size_t>(col) * static_cast<size_t>(padded);
                std::copy(host_raw.begin() + src_off,
                          host_raw.begin() + src_off + rows_input,
                          host_col.begin() + dst_off);
            }
        } else {
            for (long int row = 0; row < rows_input; ++row) {
                for (long int col = 0; col < cols; ++col) {
                    const size_t src_idx = static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
                    const size_t dst_idx = static_cast<size_t>(col) * static_cast<size_t>(padded) + static_cast<size_t>(row);
                    host_col[dst_idx] = host_raw[src_idx];
                }
            }
        }
    } else {
        if (!read_matrix_libsvm(path, rows_input, rows_padded, cols, host_col)) {
            return false;
        }
    }

    d_out.resize(host_col.size());
    T* dst_ptr = thrust::raw_pointer_cast(d_out.data());
    const size_t bytes = host_col.size() * sizeof(T);
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(dst_ptr, host_col.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            std::printf("[Error] cudaMemcpyAsync failed while loading matrix: %s\n", cudaGetErrorString(err));
            return false;
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::printf("[Error] cudaStreamSynchronize failed while loading matrix: %s\n", cudaGetErrorString(err));
            return false;
        }
    } else {
        err = cudaMemcpy(dst_ptr, host_col.data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::printf("[Error] cudaMemcpy failed while loading matrix: %s\n", cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

template<typename T>
inline bool load_matrix_from_file(long int rows,
                                  long int cols,
                                  const std::string& path,
                                  MatrixFileFormat format,
                                  MatrixFileLayout layout,
                                  thrust::device_vector<T>& d_out)
{
    return load_matrix_from_file<T>(nullptr, rows, rows, cols, path, format, layout, d_out);
}

#endif // MATRIX_IO_H
