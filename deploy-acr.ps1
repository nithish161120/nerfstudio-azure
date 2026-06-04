param(
    [Parameter(Mandatory = $true)]
    [string]$RegistryName,

    [string]$ImageName = "nerfstudio",

    [string]$Tag = (Get-Date -Format "yyyyMMdd-HHmmss"),

    [ValidateSet("RemoteAcrBuild", "LocalDockerPush")]
    [string]$Mode = "RemoteAcrBuild",

    [string]$ResourceGroup = ""
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is not installed or not available on PATH."
    }
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not (Test-Path ".\Dockerfile")) {
    throw "Dockerfile not found in $repoRoot."
}

Require-Command "az"

$imageRef = "${ImageName}:${Tag}"

if ($Mode -eq "RemoteAcrBuild") {
    Write-Host "Queuing Azure Container Registry remote build..."
    $args = @(
        "acr", "build",
        "--registry", $RegistryName,
        "--image", $imageRef,
        "--file", "Dockerfile",
        "--timeout", "7200"
    )

    if ($ResourceGroup) {
        $args += @("--resource-group", $ResourceGroup)
    }

    $args += "."
    az @args

    $loginServer = az acr show --name $RegistryName --query loginServer -o tsv
    Write-Host "Pushed image: $loginServer/$imageRef"
    exit 0
}

Require-Command "docker"

Write-Host "Logging in to Azure Container Registry..."
az acr login --name $RegistryName

$loginServer = az acr show --name $RegistryName --query loginServer -o tsv
$fullImageRef = "$loginServer/$imageRef"

Write-Host "Building local Docker image: $fullImageRef"
docker build --file Dockerfile --tag $fullImageRef .

Write-Host "Pushing image: $fullImageRef"
docker push $fullImageRef

Write-Host "Pushed image: $fullImageRef"
