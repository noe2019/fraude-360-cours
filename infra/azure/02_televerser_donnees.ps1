# =====================================================================
#  Televerse la zone d'atterrissage locale vers ADLS Gen2 et le paquet
#  Python du projet vers un volume Unity Catalog.
#  Prerequis : 01_creer_ressources.ps1 a deja ete execute.
# =====================================================================
$ErrorActionPreference = "Stop"
$Recap = Get-Content "$PSScriptRoot\ressources_creees.json" | ConvertFrom-Json
$Racine = Resolve-Path "$PSScriptRoot\..\.."

$Cle = az storage account keys list --account-name $Recap.compte_stockage `
        --resource-group $Recap.groupe_ressources --query "[0].value" -o tsv

Write-Host "Televersement des fichiers d'atterrissage..." -ForegroundColor Cyan
az storage fs directory upload `
    -f $Recap.conteneur --account-name $Recap.compte_stockage --account-key $Cle `
    -s "$Racine\data\00_landing" -d "00_landing" --recursive --output none

Write-Host "Construction du paquet Python..." -ForegroundColor Cyan
Push-Location $Racine
python -m pip install --quiet build
python -m build --wheel
Pop-Location

Write-Host "Termine. Le fichier .whl se trouve dans dist\ :" -ForegroundColor Green
Get-ChildItem "$Racine\dist\*.whl" | ForEach-Object { Write-Host "  $($_.Name)" }
Write-Host "Le televerser dans Databricks : Catalog > fraude360 > gold > bibliotheques" -ForegroundColor Yellow
