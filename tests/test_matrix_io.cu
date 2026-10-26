#include "matrix_io.h"
#include "cli_utils.h"
#include <filesystem>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char* argv[]) {
    if (argc != 2) return EXIT_FAILURE;
    try {
        long integer = 0;
        float real = 0.0f;
        require(cli::positive_integer("64", "m", integer) && integer == 64, "integer parser");
        require(cli::positive_real("1e-6", "eps", real) && real > 0, "real parser");

        const auto directory = std::filesystem::path(argv[1]);
        const std::string text = (directory / "matrix.txt").string();
        std::ofstream(text) << "1 2 3\n4 5 6\n";
        long rows = 0, cols = 0;
        require(infer_text_matrix_shape(text, MatrixFileLayout::RowMajor, rows, cols) &&
                rows == 2 && cols == 3, "row-major text shape");
        require(infer_text_matrix_shape(text, MatrixFileLayout::ColumnMajor, rows, cols) &&
                rows == 3 && cols == 2, "column-major text shape");
        std::vector<float> values;
        require(read_matrix_text(text, 6, values) && values.front() == 1 && values.back() == 6,
                "text values");
        require(!read_matrix_text(text, 5, values), "reject wrong element count");
        std::ofstream(text) << "1 2\n3\n";
        require(!infer_text_matrix_shape(text, MatrixFileLayout::RowMajor, rows, cols),
                "reject ragged text");

        const std::string sparse = (directory / "matrix.libsvm").string();
        std::ofstream(sparse) << "\n1 1:2 3:4\n\n-1 2:5\n\n";
        require(infer_libsvm_shape(sparse, rows, cols) && rows == 2 && cols == 3,
                "LIBSVM shape with blank lines");
        require(read_matrix_libsvm(sparse, 2, 4, 3, values), "LIBSVM reading with blank lines");
        require(values == std::vector<float>({2, 0, 0, 0, 0, 5, 0, 0, 4, 0, 0, 0}),
                "LIBSVM column-major layout and zero padding");
        require(!read_matrix_libsvm(sparse, 2, 2, 2, values), "reject feature outside shape");

        const std::string binary = (directory / "matrix.bin").string();
        const float original[] = {1.25f, -2.5f};
        std::ofstream stream(binary, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(original), sizeof(original));
        stream.close();
        require(read_matrix_binary(binary, 2, values) && values[0] == 1.25f && values[1] == -2.5f,
                "binary FP32 reading");
        require(!read_matrix_binary(binary, 3, values), "reject truncated binary input");
        std::puts("Host input checks passed.");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Host input check failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
