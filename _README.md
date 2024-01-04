# Mltemplate
An end-to-end starter template for machine learning projects.

# Installation

```commandline
$ git clone git@github.com:JeremyWurbs/mltemplate.git && cd mltemplate
```

You may use standard python tools (pip) as desired, but it is recommended to use 
[Rye](https://github.com/mitsuhiko/rye) (recommended), in which case all you need to do is:

```commandline
$ rye sync
```

# Usage

Rye will handle your virtual environment and dependencies for you. There are a number of useful commands available, 
which you can use through `rye run <command>`:

To lint your code with [pylint](https://www.pylint.org/), [isort](https://pycqa.github.io/isort/) and [black]():

```commandline
$ rye run lint
```

```text 
pylint mltemplate/

------------------------------------
Your code has been rated at 10.00/10

pylint --disable=protected-access tests/

-------------------------------------------------------------------
Your code has been rated at 10.00/10


isort -l 120 --check mltemplate/

isort -l 120 --check tests/

black -l 120 --check mltemplate/
All done! ✨ 🍰 ✨
43 files would be left unchanged.

black -l 120 --check tests/
All done! ✨ 🍰 ✨
14 files would be left unchanged.
```

To run unit tests with [pytest](https://docs.pytest.org/en/6.2.x/):

```commandline
$ rye run test
```

```text 
pytest -rs --cov=mltemplate --maxfail=1 --cov-report term-missing -W ignore::DeprecationWarning tests/
======================================== test session starts ========================================
platform linux -- Python 3.11.6, pytest-7.4.4, pluggy-1.3.0
rootdir: ~/mltemplate
plugins: cov-4.1.0, anyio-4.2.0, hydra-core-1.3.2
collected 13 items                                                                                  

tests/mltemplate/core/test_config.py ...                                                      [ 23%]
tests/mltemplate/core/test_registry.py s                                                      [ 30%]
tests/mltemplate/utils/test_checks.py .                                                       [ 38%]
tests/mltemplate/utils/test_conversions.py .....                                              [ 76%]
tests/mltemplate/utils/test_logging.py .                                                      [ 84%]
tests/mltemplate/utils/test_timer.py .                                                        [ 92%]
tests/mltemplate/utils/test_timer_collection.py .                                             [100%]

---------- coverage: platform linux, python 3.11.6-final-0 -----------
Name                                               Stmts   Miss  Cover   Missing
--------------------------------------------------------------------------------
mltemplate/__init__.py                                 3      0   100%
mltemplate/backend/__init__.py                         0      0   100%
mltemplate/backend/discord/__init__.py                 1      1     0%   2
mltemplate/backend/discord/discord_client.py         284    284     0%   2-409
mltemplate/backend/gateway/__init__.py                 3      3     0%   2-4
mltemplate/backend/gateway/connection_client.py       66     66     0%   2-137
mltemplate/backend/gateway/server.py                 150    150     0%   2-238
mltemplate/backend/gateway/types.py                   23     23     0%   2-39
mltemplate/backend/training/__init__.py                3      3     0%   2-4
mltemplate/backend/training/connection_client.py      11     11     0%   2-25
mltemplate/backend/training/server.py                 46     46     0%   2-87
mltemplate/backend/training/types.py                   5      5     0%   2-9
mltemplate/configs/__init__.py                         0      0   100%
mltemplate/configs/dataset/__init__.py                 0      0   100%
mltemplate/configs/experiments/__init__.py             0      0   100%
mltemplate/configs/hydra/__init__.py                   0      0   100%
mltemplate/configs/logger/__init__.py                  0      0   100%
mltemplate/configs/mlflow/__init__.py                  0      0   100%
mltemplate/configs/model/__init__.py                   0      0   100%
mltemplate/configs/paths/__init__.py                   0      0   100%
mltemplate/configs/trainer/__init__.py                 0      0   100%
mltemplate/core/__init__.py                            0      0   100%
mltemplate/core/base.py                               30     11    63%   23, 27, 35, 38-39, 42-47
mltemplate/core/config.py                             26      0   100%
mltemplate/core/registry.py                           92     38    59%   51-54, 72-73, 77, 80, 83-86, 91, 96, 100, 104-113, 117-118, 122-128, 145-146, 152, 154, 162, 167-173
mltemplate/data/__init__.py                            1      1     0%   2
mltemplate/data/mnist.py                              45     45     0%   2-122
mltemplate/models/__init__.py                          2      2     0%   2-3
mltemplate/models/cnn.py                              37     37     0%   2-70
mltemplate/models/mlp.py                              29     29     0%   2-76
mltemplate/scripts/__init__.py                         0      0   100%
mltemplate/scripts/train.py                           53     53     0%   2-100
mltemplate/types/__init__.py                           1      1     0%   2
mltemplate/types/message.py                            8      8     0%   2-14
mltemplate/utils/__init__.py                           8      0   100%
mltemplate/utils/checks.py                             3      0   100%
mltemplate/utils/conversions.py                       68      4    94%   136-137, 147-148
mltemplate/utils/dynamic.py                            9      6    33%   7-10, 30-31
mltemplate/utils/lightning.py                         42     23    45%   13-16, 22, 25, 28, 31-32, 35-45, 48, 51, 54, 57, 60, 63, 66-67
mltemplate/utils/logging.py                           27      0   100%
mltemplate/utils/mlflow.py                             6      4    33%   16-20
mltemplate/utils/timer.py                             24      0   100%
mltemplate/utils/timer_collection.py                  28      0   100%
--------------------------------------------------------------------------------
TOTAL                                               1134    854    25%

====================================== short test summary info ======================================
SKIPPED [1] tests/mltemplate/core/test_registry.py:12: No runs found in MLflow.
=================================== 12 passed, 1 skipped in 7.51s ===================================

```

To auto-format your code with [isort](https://pycqa.github.io/isort/) and [black](https://github.com/psf/black):

```commandline
$ rye run format 
```

```text
isort -l 120 mltemplate/

black -l 120 mltemplate/
All done! ✨ 🍰 ✨
43 files left unchanged.

isort -l 120 tests/

black -l 120 tests/
All done! ✨ 🍰 ✨
14 files left unchanged.
```

You have a starter CI workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml) that will lint and test your 
project on Linux/MacOS/Windows. By default they will run with every push / pull request and can be accessed directly 
from [GithubActions](https://github.com/JeremyWurbs/mltemplate/actions). 

