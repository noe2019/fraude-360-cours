# =====================================================================
#  Fraude-360 - creation des ressources Azure (PowerShell / Azure CLI)
#
#  Prerequis :  az login       (compte d'essai Azure accepte)
#  Execution :  .\01_creer_ressources.ps1
#  Suppression totale : .\99_supprimer_ressources.ps1
#
#  Toutes les ressources sont dans UN SEUL groupe de ressources : supprimer
#  le groupe suffit a tout effacer et a arreter la facturation.
# =====================================================================

$ErrorActionPreference = "Stop"

# --- Parametres (modifiables) ----------------------------------------
$Region          = "canadacentral"      # donnees hebergees au Canada
$GroupeRessource = "rg-fraude360"
$Suffixe         = -join ((48..57) + (97..122) | Get-Random -Count 6 | ForEach-Object {[char]$_})
$CompteStockage  = "stfraude360$Suffixe" # doit etre unique mondialement
$Conteneur       = "lakehouse"
$EspaceDatabricks= "dbw-fraude360"
$CoffreFort      = "kv-fraude360-$Suffixe"

Write-Host "=== Fraude-360 : creation de l'infrastructure Azure ===" -ForegroundColor Cyan
Write-Host "Region : $Region | Groupe : $GroupeRessource"

# --- 1. Groupe de ressources -----------------------------------------
az group create --name $GroupeRessource --location $Region --output none
Write-Host "[1/6] Groupe de ressources cree." -ForegroundColor Green

# --- 2. Stockage ADLS Gen2 (espace de noms hierarchique) --------------
az storage account create `
    --name $CompteStockage --resource-group $GroupeRessource --location $Region `
    --sku Standard_LRS --kind StorageV2 --enable-hierarchical-namespace true `
    --min-tls-version TLS1_2 --allow-blob-public-access false --output none

$Cle = az storage account keys list --account-name $CompteStockage `
        --resource-group $GroupeRessource --query "[0].value" -o tsv

az storage container create --name $Conteneur --account-name $CompteStockage `
    --account-key $Cle --output none
foreach ($dossier in @("00_landing", "01_bronze", "02_silver", "03_gold")) {
    az storage fs directory create -n $dossier -f $Conteneur `
        --account-name $CompteStockage --account-key $Cle --output none
}
Write-Host "[2/6] Lac de donnees pret : abfss://$Conteneur@$CompteStockage.dfs.core.windows.net" -ForegroundColor Green

# --- 3. Coffre de cles -----------------------------------------------
az keyvault create --name $CoffreFort --resource-group $GroupeRessource `
    --location $Region --enable-rbac-authorization false --output none
az keyvault secret set --vault-name $CoffreFort --name "cle-stockage" `
    --value $Cle --output none
Write-Host "[3/6] Coffre de cles cree : $CoffreFort" -ForegroundColor Green

# --- 4. Espace de travail Databricks ---------------------------------
az databricks workspace create `
    --name $EspaceDatabricks --resource-group $GroupeRessource --location $Region `
    --sku premium --output none
$UrlDatabricks = az databricks workspace show --name $EspaceDatabricks `
    --resource-group $GroupeRessource --query workspaceUrl -o tsv
Write-Host "[4/6] Databricks : https://$UrlDatabricks" -ForegroundColor Green

# --- 5. Connecteur d'acces (identite managee pour Unity Catalog) ------
az databricks access-connector create `
    --name "ac-fraude360" --resource-group $GroupeRessource `
    --location $Region --identity-type SystemAssigned --output none
$IdPrincipal = az databricks access-connector show --name "ac-fraude360" `
    --resource-group $GroupeRessource --query identity.principalId -o tsv
$IdStockage = az storage account show --name $CompteStockage `
    --resource-group $GroupeRessource --query id -o tsv
az role assignment create --assignee-object-id $IdPrincipal `
    --assignee-principal-type ServicePrincipal `
    --role "Storage Blob Data Contributor" --scope $IdStockage --output none
Write-Host "[5/6] Connecteur d'acces autorise sur le stockage." -ForegroundColor Green

# --- 6. Recapitulatif -------------------------------------------------
$Recap = [ordered]@{
    region                = $Region
    groupe_ressources     = $GroupeRessource
    compte_stockage       = $CompteStockage
    conteneur             = $Conteneur
    racine_abfss          = "abfss://$Conteneur@$CompteStockage.dfs.core.windows.net"
    espace_databricks     = $EspaceDatabricks
    url_databricks        = "https://$UrlDatabricks"
    coffre_cles           = $CoffreFort
    connecteur_acces      = "ac-fraude360"
}
$Recap | ConvertTo-Json | Out-File -Encoding utf8 "$PSScriptRoot\ressources_creees.json"

Write-Host "`n[6/6] Termine. Recapitulatif ecrit dans ressources_creees.json" -ForegroundColor Cyan
$Recap.GetEnumerator() | ForEach-Object { Write-Host ("  {0,-20} {1}" -f $_.Key, $_.Value) }
Write-Host "`nEtape suivante : ouvrir Databricks et executer notebooks/00_configuration.py" -ForegroundColor Yellow
Write-Host "PENSEZ A SUPPRIMER LES RESSOURCES APRES LA DEMONSTRATION :" -ForegroundColor Red
Write-Host "  az group delete --name $GroupeRessource --yes --no-wait" -ForegroundColor Red
