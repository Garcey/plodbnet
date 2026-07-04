# Launch the PUBLIC build on :8770, loading secrets from .env.public.
# Usage:  .\run_public.ps1            (foreground; Ctrl+C stops)
#         .\run_public.ps1 -Detached  (background via Start-Process)
param([switch]$Detached)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$envFile = Join-Path $root ".env.public"
if (-not (Test-Path $envFile)) {
    Write-Error ".env.public not found -- see PUBLIC_SETUP.md"
    exit 1
}

# KEY="value" lines -> process env (quotes optional, # comments ignored)
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
    $k, $v = $_ -split '=', 2
    $v = $v.Trim().Trim('"')
    if ($v -ne "") { Set-Item -Path "Env:$($k.Trim())" -Value $v }
}
$env:PLO5BP_PUBLIC = "1"

if ($env:GOOGLE_CLIENT_ID) { Write-Host "[ok] Google sign-in configured" -ForegroundColor Green }
else { Write-Host "[--] Google sign-in NOT configured (only dev login will work)" -ForegroundColor Yellow }
if ($env:STRIPE_SECRET_KEY) { Write-Host "[ok] Stripe configured" -ForegroundColor Green }
else { Write-Host "[--] Stripe NOT configured (checkout will 503; comp grants still work)" -ForegroundColor Yellow }
if ($env:PLO5BP_DEV_LOGIN) {
    Write-Host "[!!] DEV LOGIN IS ON -- loopback-only, but NEVER expose a tunnel like this" -ForegroundColor Red
}
Write-Host "URL: $($env:PLO5BP_BASE_URL)   admin: $($env:PLO5BP_ADMIN_EMAILS)"

# PRODUCTION NOTE: wrapgto.com is served by the Hetzner VPS (87.99.132.209,
# systemd units wrapgto.service + cloudflared). This script is for LOCAL DEV
# only and must NOT start a tunnel — a second connector on the 'wrapgto'
# tunnel would route live traffic to this laptop.
if ($env:PLO5BP_BASE_URL -like "https*") {
    Write-Host "[!!] BASE_URL is public ($($env:PLO5BP_BASE_URL)) but prod lives on the VPS." -ForegroundColor Red
    Write-Host "     For local dev set PLO5BP_BASE_URL=http://127.0.0.1:8770 in .env.public." -ForegroundColor Red
}

if ($Detached) {
    Start-Process -WindowStyle Hidden -FilePath "$root\.venv\Scripts\python.exe" `
        -ArgumentList "-m", "uvicorn", "plo5bp.ui.server:app", "--port", "8770"
    Write-Host "started detached on :8770"
} else {
    & "$root\.venv\Scripts\python.exe" -m uvicorn plo5bp.ui.server:app --port 8770
}
