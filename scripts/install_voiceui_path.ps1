$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($null -eq $currentUserPath) {
    $currentUserPath = ""
}

function Normalize-PathEntry {
    param([string] $PathEntry)
    return $PathEntry.Trim().TrimEnd("\").ToLowerInvariant()
}

$normalizedProjectRoot = Normalize-PathEntry $projectRoot
$entries = @(
    $currentUserPath -split ";" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
$alreadyInstalled = $false
foreach ($entry in $entries) {
    if ((Normalize-PathEntry $entry) -eq $normalizedProjectRoot) {
        $alreadyInstalled = $true
        break
    }
}

if ($alreadyInstalled) {
    Write-Output "VoiceUI path already installed: $projectRoot"
    exit 0
}

$newEntries = @($entries + $projectRoot)
$newUserPath = ($newEntries -join ";")
[Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")

Write-Output "Added VoiceUI to user PATH: $projectRoot"
Write-Output "Open a new cmd window, then run: VoiceUI"
