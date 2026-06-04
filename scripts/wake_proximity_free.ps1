param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
python -m voiceui.wake_proximity wake-live @ExtraArgs
exit $LASTEXITCODE
