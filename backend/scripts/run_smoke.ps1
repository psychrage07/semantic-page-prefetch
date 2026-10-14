param(
    [int]$Queries = 150
)

$ErrorActionPreference = "Stop"
$BackendRoot = Split-Path -Parent $PSScriptRoot
$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$RunRoot = Join-Path $BackendRoot "runs\smoke-$RunId"
$DatabasePath = Join-Path $RunRoot "pagewise-smoke.sqlite"
$TraceDir = Join-Path $RunRoot "trace"
$ModelDir = Join-Path $RunRoot "models"
$EvalDir = Join-Path $RunRoot "evaluation"

function Invoke-Pagewise {
    param([Parameter(Mandatory=$true)][string[]]$CliArgs)
    & python -m pagewise.cli @CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Pagewise command failed (exit $LASTEXITCODE): python -m pagewise.cli $($CliArgs -join ' ')"
    }
}

Push-Location $BackendRoot
try {
    Write-Host "Checking environment..." -ForegroundColor Cyan
    & python -c "import apsw, torch; print('APSW', apsw.apswversion(), 'SQLite', apsw.sqlitelibversion()); print('PyTorch', torch.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "APSW or PyTorch is missing. Activate the project's venv and run: python -m pip install -e `".[test]`""
    }

    New-Item -ItemType Directory -Force -Path $RunRoot, $TraceDir, $ModelDir, $EvalDir | Out-Null

    Write-Host "Generating small reproducible database..." -ForegroundColor Cyan
    Invoke-Pagewise @(
        "generate-db", "--db", $DatabasePath,
        "--customers", "500", "--products", "150", "--orders", "3000",
        "--avg-items", "2", "--graph-nodes", "600", "--edges-per-node", "2",
        "--seed", "23"
    )

    Write-Host "Tracing $Queries actual query executions..." -ForegroundColor Cyan
    Invoke-Pagewise @(
        "trace", "--db", $DatabasePath, "--out", $TraceDir,
        "--queries", "$Queries", "--seed", "1234", "--cache-kib", "128"
    )

    Write-Host "Building per-query page sequences..." -ForegroundColor Cyan
    Invoke-Pagewise @(
        "build-sequences", "--trace", (Join-Path $TraceDir "trace.jsonl"),
        "--page-map", (Join-Path $TraceDir "page_map.json"),
        "--out", (Join-Path $TraceDir "sequences.jsonl")
    )

    Write-Host "Training Model A and Model B..." -ForegroundColor Cyan
    Invoke-Pagewise @(
        "train", "--sequences", (Join-Path $TraceDir "sequences.jsonl"),
        "--page-map", (Join-Path $TraceDir "page_map.json"),
        "--out", $ModelDir, "--seed", "7", "--epochs", "5",
        "--candidates", "16", "--hidden", "32"
    )

    Write-Host "Evaluating held-out predictions..." -ForegroundColor Cyan
    Invoke-Pagewise @(
        "evaluate", "--sequences", (Join-Path $ModelDir "test_sequences.jsonl"),
        "--page-map", (Join-Path $TraceDir "page_map.json"),
        "--candidate-index", (Join-Path $ModelDir "candidate_index.json"),
        "--model-a", (Join-Path $ModelDir "model_a.pt"),
        "--model-b", (Join-Path $ModelDir "model_b.pt"),
        "--out", $EvalDir
    )

    Write-Host "`nSmoke pipeline complete." -ForegroundColor Green
    Write-Host "Run directory: $RunRoot"
    Write-Host "Prediction metrics: $(Join-Path $EvalDir 'predictor_metrics.json')"
    Write-Host "This smoke run validates the pipeline only; do not present its metrics as final research results."
}
finally {
    Pop-Location
}
