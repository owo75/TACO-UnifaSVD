# Bundled matrix inputs

These files use LIBSVM-style sparse rows: the first token is a label (ignored by
the SVD programs), followed by 1-based `index:value` features. The loaders expand
them into dense matrices. Multi-label fields, such as those in `bibtex.txt`, are
also ignored.

The shapes below are measured from the checked-in files: the row count is the
number of nonempty samples and the column count is the largest feature index.
Use these dimensions rather than assuming they match another dataset release.

| File | Rows | Columns |
| --- | ---: | ---: |
| `bibtex.txt` | 7,395 | 1,836 |
| `letter.scale.txt` | 15,000 | 16 |
| `pendigits.txt` | 7,494 | 16 |
| `phishing.txt` | 11,055 | 68 |
| `svmguide4.txt` | 300 | 10 |

For background and upstream dataset references, see the
[LIBSVM dataset collection](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
The table describes the bundled copies; it does not establish that they are
byte-for-byte copies of an upstream download.

Example with adaptive rank selection:

```bash
./UnifaSVD --matrix-file=dataset/svmguide4.txt --matrix-format=libsvm \
  --k=8 --rank=auto --eps=1e-6 --precision=float --debug
```
