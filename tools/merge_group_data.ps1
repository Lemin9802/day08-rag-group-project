$GroupRepo = "C:\Users\THAI NHI\Desktop\VinUni\Day08_RAG_pipeline_cohort2"
$ProjectRepo = "C:\Users\THAI NHI\Desktop\VinUni\day08-rag-group-project"

$Members = @("Nhi", "Huy", "Nghia")

$Targets = @(
    "$ProjectRepo\data\landing\legal",
    "$ProjectRepo\data\landing\news",
    "$ProjectRepo\data\standardized\legal",
    "$ProjectRepo\data\standardized\news"
)

foreach ($target in $Targets) {
    if (!(Test-Path $target)) {
        New-Item -ItemType Directory -Force -Path $target | Out-Null
    }

    Get-ChildItem -Path $target -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne ".gitkeep" } |
        Remove-Item -Force
}

function Copy-WithPrefix {
    param (
        [string]$SourceDir,
        [string]$TargetDir,
        [string]$Prefix
    )

    if (!(Test-Path $SourceDir)) {
        Write-Host "Skip missing: $SourceDir"
        return
    }

    Get-ChildItem -Path $SourceDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne ".gitkeep" } |
        ForEach-Object {
            $NewName = $Prefix + "_" + $_.Name
            $TargetPath = Join-Path $TargetDir $NewName
            Copy-Item $_.FullName $TargetPath -Force
            Write-Host "Copied: $TargetPath"
        }
}

foreach ($member in $Members) {
    $prefix = $member.ToLower()

    Copy-WithPrefix "$GroupRepo\individual\$member\data\landing\legal" "$ProjectRepo\data\landing\legal" $prefix
    Copy-WithPrefix "$GroupRepo\individual\$member\data\landing\news" "$ProjectRepo\data\landing\news" $prefix
    Copy-WithPrefix "$GroupRepo\individual\$member\data\standardized\legal" "$ProjectRepo\data\standardized\legal" $prefix
    Copy-WithPrefix "$GroupRepo\individual\$member\data\standardized\news" "$ProjectRepo\data\standardized\news" $prefix
}

Write-Host ""
Write-Host "Merge done."
Write-Host "Legal markdown count:"
(Get-ChildItem "$ProjectRepo\data\standardized\legal" -Filter *.md).Count
Write-Host "News markdown count:"
(Get-ChildItem "$ProjectRepo\data\standardized\news" -Filter *.md).Count