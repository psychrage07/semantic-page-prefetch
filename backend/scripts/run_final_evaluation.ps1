param(
    [string]$RunRoot = "runs/experiment-latest"
)

$ErrorActionPreference = "Stop"
$db = Join-Path $RunRoot "pagewise-medium.sqlite"
$models = Join-Path $RunRoot "models"
$strategies = "stock,none,sequential,markov1,markov2,model_a,model_b,hybrid"
$common = @(
    "--db", $db, "--model-dir", $models, "--queries", "100", "--seeds", "901,902,903",
    "--strategies", $strategies, "--cache-kib", "4096", "--cache-pages", "32", "--degree", "2",
    "--queue-size", "32", "--predict-on", "miss", "--predict-every", "1", "--max-score-candidates", "12"
)

python -m pagewise.cli run-suite @common --miss-delay-ms 0.2 --out (Join-Path $RunRoot "final-latency-0p2ms")
python -m pagewise.cli run-suite @common --miss-delay-ms 1.0 --out (Join-Path $RunRoot "final-latency-1p0ms")
python -m pagewise.cli run-suite @common --miss-delay-ms 0.0 --out (Join-Path $RunRoot "final-real-storage")
