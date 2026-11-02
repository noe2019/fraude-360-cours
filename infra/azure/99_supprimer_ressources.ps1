# =====================================================================
#  Fraude-360 - suppression complete des ressources Azure
#  A executer imperativement apres la demonstration : un espace de travail
#  Databricks laisse actif continue d'etre facture.
# =====================================================================
$GroupeRessource = "rg-fraude360"

Write-Host "Ressources qui seront supprimees :" -ForegroundColor Yellow
az resource list --resource-group $GroupeRessource --output table

$reponse = Read-Host "`nConfirmer la suppression du groupe '$GroupeRessource' ? (oui/non)"
if ($reponse -eq "oui") {
    az group delete --name $GroupeRessource --yes --no-wait
    Write-Host "Suppression lancee. Verification : az group show --name $GroupeRessource" -ForegroundColor Green
} else {
    Write-Host "Suppression annulee." -ForegroundColor Cyan
}
