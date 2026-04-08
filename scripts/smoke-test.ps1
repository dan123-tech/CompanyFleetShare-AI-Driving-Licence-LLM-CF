param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,
    [string]$BaseUrl = "http://localhost:8080"
)

if (-not (Test-Path $ImagePath)) {
    Write-Error "Image file not found: $ImagePath"
    exit 1
}

Write-Host "Checking health endpoint..."
try {
    $health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health"
    $healthJson = $health | ConvertTo-Json -Depth 5
    Write-Host $healthJson
} catch {
    Write-Error "Health check failed. Is the API running on $BaseUrl?"
    exit 1
}

if (-not $health.cloudflare_configured) {
    Write-Warning "cloudflare_configured is false. Check CF_ACCOUNT_ID / CF_API_TOKEN in .env"
}

Write-Host "Calling /validate with image..."
try {
    $response = curl.exe -s -X POST "$BaseUrl/validate" -F "file=@$ImagePath"
    Write-Host $response
} catch {
    Write-Error "Validation request failed."
    exit 1
}
