from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ("UnifaSVD", "cusolverDnXgesvd", "cusolverDnXgesvdr")
SYNTHETIC = {
    "UnifaSVD": ["64", "32", "16", "8", "1e-6", "geometric", "1", "1e-4", "float"],
    "cusolverDnXgesvd": ["64", "32", "8", "geometric", "1", "1e-4"],
    "cusolverDnXgesvdr": ["64", "32", "8", "geometric", "1", "1e-4"],
}


class CommandLineTests(unittest.TestCase):
    def run_program(self, program, args):
        return subprocess.run(
            [str(ROOT / program), *args], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15,
        )

    def assert_rejected(self, program, args, diagnostic="Error"):
        result = self.run_program(program, args)
        self.assertGreater(result.returncode, 0, result.stdout)
        self.assertIn(diagnostic, result.stdout)
        self.assertNotIn("terminate called", result.stdout)
        self.assertNotIn("CUDA error", result.stdout)

    def test_help_exits_successfully_without_gpu(self):
        for program in PROGRAMS:
            for option in ("--help", "-h"):
                with self.subTest(program=program, option=option):
                    result = self.run_program(program, [option])
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn("Usage", result.stdout)

    def test_missing_arguments_fail(self):
        for program in PROGRAMS:
            with self.subTest(program=program):
                self.assert_rejected(program, [], "Usage")

    def test_unknown_options_and_extra_arguments_fail(self):
        for program, args in SYNTHETIC.items():
            for extra in ("--precison=half", "extra"):
                with self.subTest(program=program, extra=extra):
                    self.assert_rejected(program, args + [extra])

    def test_invalid_dimensions_and_spectra_fail(self):
        for program, args in SYNTHETIC.items():
            for invalid in ("0", "-1", "64junk", "2147483648", "999999999999999999999999"):
                with self.subTest(program=program, dimension=invalid):
                    self.assert_rejected(program, [invalid, *args[1:]])
            spectrum_max_index = 6 if program == "UnifaSVD" else 4
            for invalid in ("nan", "inf", "0", "1junk", "1e999"):
                with self.subTest(program=program, s_max=invalid):
                    bad_args = args.copy()
                    bad_args[spectrum_max_index] = invalid
                    self.assert_rejected(program, bad_args)
            bad_args = args.copy()
            bad_args[spectrum_max_index] = "1e-6"
            self.assert_rejected(program, bad_args)

    def test_full_svd_rejects_wide_matrices_and_impossible_rank(self):
        args = SYNTHETIC["cusolverDnXgesvd"].copy()
        args[:2] = ["16", "32"]
        self.assert_rejected("cusolverDnXgesvd", args)
        args[:3] = ["64", "32", "33"]
        self.assert_rejected("cusolverDnXgesvd", args)

    def test_unifasvd_rejects_invalid_controls(self):
        for option in ("--accuracy=typo", "--vectors=typo", "--padding=typo",
                       "--energy-tol=1", "--energy-tol=nan", "--tail-norm-eps=0"):
            with self.subTest(option=option):
                self.assert_rejected("UnifaSVD", SYNTHETIC["UnifaSVD"] + [option])
        for invalid in ("0", "nan", "1e-6junk"):
            args = SYNTHETIC["UnifaSVD"].copy()
            args[4] = invalid
            self.assert_rejected("UnifaSVD", args)

    def test_randomized_svd_rejects_invalid_iterations(self):
        for invalid in ("0", "2junk", "2147483648"):
            with self.subTest(niters=invalid):
                self.assert_rejected("cusolverDnXgesvdr", SYNTHETIC["cusolverDnXgesvdr"] +
                                     [f"--niters={invalid}"])

    def test_file_shape_must_match_explicit_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory) / "matrix.txt"
            matrix.write_text("1 0\n0 1\n", encoding="utf-8")
            for m, n in (("auto", "3"), ("3", "auto")):
                for program in ("UnifaSVD", "cusolverDnXgesvdr"):
                    args = ([m, n, "1", "auto", "1e-6", "float"] if program == "UnifaSVD"
                            else [m, n, "auto"])
                    with self.subTest(program=program, m=m, n=n):
                        self.assert_rejected(program, args + [f"--matrix-file={matrix}"])

    def test_malformed_libsvm_reports_error_instead_of_aborting(self):
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory) / "bad.libsvm"
            for token in ("1:bad", "1junk:2", "1:2junk", "1:nan", "0:1",
                          "999999999999999999999999:1"):
                matrix.write_text(f"1 {token}\n", encoding="utf-8")
                for program in ("UnifaSVD", "cusolverDnXgesvdr"):
                    args = (["auto", "auto", "1", "auto", "1e-6", "float"]
                            if program == "UnifaSVD" else ["auto", "auto", "auto"])
                    with self.subTest(program=program, token=token):
                        self.assert_rejected(program, args + [f"--matrix-file={matrix}",
                                                             "--matrix-format=libsvm"])

    def test_host_matrix_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_program("tests/test_matrix_io", [directory])
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("Host input checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
