param(
    [int]$Queries = 300,
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss")
)

$ErrorActionPreference = "Stop"
$BackendRoot = $PSScriptRoot
if (-not $BackendRoot) { $BackendRoot = "." }
if (Test-Path (Join-Path $BackendRoot "pagewise")) {
    # already in backend root
} elseif (Test-Path (Join-Path (Split-Path -Parent $BackendRoot) "pagewise")) {
    $BackendRoot = Split-Path -Parent $BackendRoot
} else {
    $BackendRoot = "."
}

$RunRoot = Join-Path $BackendRoot "runs\experiment-$RunId"
$DatabasePath = Join-Path $RunRoot "pagewise-medium.sqlite"
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
    Write-Host "`n=== PAGEWISE MEDIUM-SCALE EXPERIMENT ===" -ForegroundColor Cyan
    Write-Host "Run ID: $RunId"
    Write-Host "Output: $RunRoot"

    Write-Host "[1/8] Generating medium database..." -ForegroundColor Yellow
    Invoke-Pagewise @(
        "generate-db", "--db", $DatabasePath,
        "--customers", "2000", "--products", "500", "--orders", "15000",
        "--avg-items", "3", "--graph-nodes", "2000", "--edges-per-node", "3",
        "--seed", "42"
    )

    Write-Host "[2/8] Tracing $Queries queries..." -ForegroundColor Yellow
    Invoke-Pagewise @(
        "trace", "--db", $DatabasePath, "--out", $TraceDir,
        "--queries", "$Queries", "--seed", "1234", "--cache-kib", "128"
    )

    Write-Host "[3/8] Building page sequences..." -ForegroundColor Yellow
    Invoke-Pagewise @(
        "build-sequences",
        "--trace", (Join-Path $TraceDir "trace.jsonl"),
        "--page-map", (Join-Path $TraceDir "page_map.json"),
        "--out", (Join-Path $TraceDir "sequences.jsonl")
    )

    Write-Host "[4/8] Training models with ablations..." -ForegroundColor Yellow
    Invoke-Pagewise @(
        "train", "--sequences", (Join-Path $TraceDir "sequences.jsonl"),
        "--page-map", (Join-Path $TraceDir "page_map.json"),
        "--out", $ModelDir, "--seed", "7", "--epochs", "12",
        "--candidates", "24", "--hidden", "64",
        "--semantic-ablations", "no_page_metadata=page_metadata;no_query_context=query_context;no_object=object"
    )

    Write-Host "[5/8] Evaluating predictions..." -ForegroundColor Yellow
    Invoke-Pagewise @(
        "evaluate",
        "--sequences", (Join-Path $ModelDir "test_sequences.jsonl"),
        "--page-map", (Join-Path $TraceDir "page_map.json"),
        "--candidate-index", (Join-Path $ModelDir "candidate_index.json"),
        "--model-a", (Join-Path $ModelDir "model_a.pt"),
        "--model-b", (Join-Path $ModelDir "model_b.pt"),
        "--out", $EvalDir,
        "--semantic-ablations", "no_page_metadata=page_metadata;no_query_context=query_context",
        "--retrained-ablation-dir", $ModelDir
    )

    Write-Host "[6/8] Running all-strategy suite (cache=32, degree=2, miss, delay=0.2ms)..." -ForegroundColor Yellow
    $SuiteDir = Join-Path $RunRoot "suite-tuned-all"
    Invoke-Pagewise @(
        "run-suite",
        "--db", $DatabasePath, "--model-dir", $ModelDir,
        "--out", $SuiteDir,
        "--queries", "200", "--seeds", "101,202,303",
        "--strategies", "stock,none,sequential,markov1,markov2,model_a,model_b",
        "--cache-pages", "32", "--degree", "2",
        "--predict-on", "miss", "--queue-size", "128",
        "--miss-delay-ms", "0.2"
    )

    Write-Host "[7/8] Running optimized Model B suite (max-score-candidates=12)..." -ForegroundColor Yellow
    $SuiteOptDir = Join-Path $RunRoot "suite-model-b-optimized"
    Invoke-Pagewise @(
        "run-suite",
        "--db", $DatabasePath, "--model-dir", $ModelDir,
        "--out", $SuiteOptDir,
        "--queries", "200", "--seeds", "101,202,303",
        "--strategies", "none,model_a,model_b",
        "--cache-pages", "32", "--degree", "2",
        "--predict-on", "miss", "--queue-size", "128",
        "--miss-delay-ms", "0.2",
        "--max-score-candidates", "12"
    )

    Write-Host "[8/8] Generating reports..." -ForegroundColor Yellow
    Invoke-Pagewise @("report", "--input", (Join-Path $SuiteDir "suite_summary.json"))
    Invoke-Pagewise @("report", "--input", (Join-Path $SuiteOptDir "suite_summary.json"))

    Write-Host "`n=== EXPERIMENT COMPLETE ===" -ForegroundColor Green
    Write-Host "Run root: $RunRoot"
    Write-Host "Prediction metrics: $(Join-Path $EvalDir 'predictor_metrics.json')"
    Write-Host "Suite report: $(Join-Path $SuiteDir 'suite_report.md')"
    Write-Host "Optimized suite: $(Join-Path $SuiteOptDir 'suite_report.md')"
}
finally {
    Pop-Location
}
