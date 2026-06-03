import sys

if len(sys.argv) > 1 and sys.argv[1] == "run":
    from .run_bench import main

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    raise SystemExit(main())

from .generate import main

raise SystemExit(main())
