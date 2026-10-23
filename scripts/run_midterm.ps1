# DS340W Group 2 midterm pipeline: train baseline -> eval_shift (baseline, fmfp) -> plots
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File scripts\run_midterm.ps1 -FmfpCkpt C:\path\to\fmfp\model.pth
#
# Skip steps with switches, for example re-plot only:
#   ... -SkipTrain -SkipEvalBaseline -SkipEvalFmfp
# Recompute metrics from the inference caches without the GPU:
#   ... -SkipTrain -UseCache
# Quick end-to-end test:
#   ... -SkipTrain -Limit 200 -Corruptions gaussian_noise,fog
#
# Every step's output, start/end time and duration go to output_shift\run_midterm_<time>.log

param(
    [string]$Python = "",
    [string]$FmfpCkpt = "",
    [string]$BaselineCkpt = "",
    [string]$DataPath = "./data/",
    [string]$Cifar10cPath = "./data/CIFAR-10-C/",
    [int]$Epochs = 201,
    [int]$Limit = 0,
    [string[]]$Corruptions = @(),
    [int]$RaMembers = 10,
    [ValidateSet("logits", "logits+emb")][string]$RaFeatures = "logits",
    [int]$NumWorkers = 0,
    [switch]$UseCache,
    [switch]$SkipTrain,
    [switch]$SkipEvalBaseline,
    [switch]$SkipEvalFmfp,
    [switch]$SkipPlot
)

# with -File, "a,b" arrives as one string; split it here
$Corruptions = @($Corruptions | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

if ($Python -eq "") {
    $venv = Join-Path $repo ".venv\Scripts\python.exe"
    if (Test-Path $venv) { $Python = $venv } else { $Python = "python" }
}
# main_base.py saves to save_path + data + '_' + model + '_' + method
if ($BaselineCkpt -eq "") {
    $BaselineCkpt = Join-Path $repo (Join-Path "output" (Join-Path "cifar10_resnet18_Baseline" "model.pth"))
}

$outDir = Join-Path $repo "output_shift"
New-Item -ItemType Directory -Force $outDir | Out-Null
$log = Join-Path $outDir ("run_midterm_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$runStart = Get-Date
$TimeFmt = "yyyy-MM-dd HH:mm:ss"

# Tee-Object in Windows PowerShell 5.1 writes UTF-16; append ASCII lines instead
function Write-Log([string]$msg) {
    $msg
    Add-Content -Path $log -Value $msg -Encoding ASCII
}

function Invoke-Step([string]$name, [string[]]$pyArgs) {
    $start = Get-Date
    Write-Log ""
    Write-Log ("=== {0}" -f $name)
    Write-Log ("START  : {0}" -f $start.ToString($TimeFmt))
    Write-Log ("CMD    : {0} {1}" -f $Python, ($pyArgs -join " "))
    & $Python -u @pyArgs 2>&1 | ForEach-Object { Write-Log "$_" }
    $code = $LASTEXITCODE
    $end = Get-Date
    Write-Log ("END    : {0}" -f $end.ToString($TimeFmt))
    Write-Log ("TIME   : {0:hh\:mm\:ss}" -f ($end - $start))
    if ($code -ne 0) {
        Write-Log ("FAILED : {0} exited with code {1}" -f $name, $code)
        Write-Log ("TOTAL  : {0:hh\:mm\:ss}" -f ((Get-Date) - $runStart))
        exit $code
    }
}

function Get-EvalArgs([string]$ckpt, [string]$tag) {
    $a = @("eval_shift.py", "--tag", $tag, "--data_path", $DataPath,
           "--cifar10c_path", $Cifar10cPath, "--num_workers", "$NumWorkers",
           "--ra_members", "$RaMembers", "--ra_features", $RaFeatures)
    # PowerShell 5.1 drops empty native arguments, so only pass --ckpt when it is set
    if ($ckpt -ne "") { $a += @("--ckpt", $ckpt) }
    if ($Limit -gt 0) { $a += @("--limit", "$Limit") }
    if ($Corruptions.Count -gt 0) { $a += "--corruptions"; $a += $Corruptions }
    if ($UseCache) { $a += "--use_cache" }
    return ,$a
}

Write-Log ("RUN START : {0}" -f $runStart.ToString($TimeFmt))
Write-Log ("REPO      : {0}" -f $repo)
Write-Log ("PYTHON    : {0}" -f $Python)
& $Python -c "import torch; print('TORCH     :', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>&1 |
    ForEach-Object { Write-Log "$_" }

if (-not $SkipTrain) {
    Invoke-Step "1. train baseline" @("main_base.py", "--model", "resnet18", "--epochs", "$Epochs",
                                      "--run", "1", "--gpu", "0", "--data_path", $DataPath)
}

if (-not $SkipEvalBaseline) {
    if (-not $UseCache -and -not (Test-Path $BaselineCkpt)) {
        Write-Log ("baseline checkpoint not found: {0}" -f $BaselineCkpt)
        exit 1
    }
    Invoke-Step "2a. eval_shift baseline" (Get-EvalArgs $BaselineCkpt "baseline")
}

if (-not $SkipEvalFmfp) {
    if (-not $UseCache -and ($FmfpCkpt -eq "" -or -not (Test-Path $FmfpCkpt))) {
        Write-Log ("pass the FMFP checkpoint with -FmfpCkpt (got '{0}')" -f $FmfpCkpt)
        exit 1
    }
    Invoke-Step "2b. eval_shift fmfp" (Get-EvalArgs $FmfpCkpt "fmfp")
}

if (-not $SkipPlot) {
    foreach ($m in @("auroc", "aurc")) {
        Invoke-Step ("3a. plot_shift {0}" -f $m) @("plot_shift.py", "--metric", $m)
    }
    Invoke-Step "3b. plot_monitor" @("plot_monitor.py")
}

Write-Log ""
Write-Log ("RUN END   : {0}" -f (Get-Date).ToString($TimeFmt))
Write-Log ("TOTAL     : {0:hh\:mm\:ss}" -f ((Get-Date) - $runStart))
Write-Log ("LOG       : {0}" -f $log)
