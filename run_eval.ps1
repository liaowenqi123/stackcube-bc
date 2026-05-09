# Run evaluation and save output
$ErrorActionPreference = "Continue"
$start = Get-Date
$logFile = "d:\文件\lwq临时文件夹\机器人学\stackcube_repro\outputs\eval_consistency_v1_100ep\eval_log.txt"

# Ensure output dir exists
$outDir = "d:\文件\lwq临时文件夹\机器人学\stackcube_repro\outputs\eval_consistency_v1_100ep"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Start the evaluation process
$proc = Start-Process -FilePath "C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe" `
    -ArgumentList "src/eval_policy.py", "--algo", "diffusion", "--ckpt", "outputs/diffusion_consistency_v1/best.pt", "--episodes", "100", "--output-dir", "outputs/eval_consistency_v1_100ep", "--max-steps", "400", "--sampler", "ddim", "--T-inf", "20", "--eta", "0.0", "--save-gif", "--gif-episodes", "3" `
    -WorkingDirectory "d:\文件\lwq临时文件夹\机器人学\stackcube_repro" `
    -PassThru `
    -RedirectStandardOutput "$outDir\stdout.txt" `
    -RedirectStandardError "$outDir\stderr.txt"

Write-Host "Started process with PID: $($proc.Id)"
Write-Host "Waiting for completion..."

$proc.WaitForExit()
$exitCode = $proc.ExitCode
$elapsed = (Get-Date) - $start

Write-Host "Process completed with exit code: $exitCode"
Write-Host "Elapsed time: $($elapsed.TotalMinutes.ToString('F1')) minutes"

# Read logs
if (Test-Path "$outDir\stdout.txt") {
    $stdout = Get-Content "$outDir\stdout.txt" -Raw
    Write-Host "=== STDOUT ==="
    Write-Host $stdout
}
if (Test-Path "$outDir\stderr.txt") {
    $stderr = Get-Content "$outDir\stderr.txt" -Raw
    if ($stderr) {
        Write-Host "=== STDERR ==="
        Write-Host $stderr
    }
}

# Check metrics
$metricsFile = "$outDir\metrics.json"
if (Test-Path $metricsFile) {
    $metrics = Get-Content $metricsFile -Raw | ConvertFrom-Json
    Write-Host "=== FINAL METRICS ==="
    Write-Host "Success Rate: $($metrics.success_rate * 100)%"
    Write-Host "Avg Return: $($metrics.avg_return)"
    Write-Host "Avg Ep Len: $($metrics.avg_ep_len)"
    Write-Host "Episodes: $($metrics.episodes)"
}
