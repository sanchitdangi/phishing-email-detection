# Python 3.11 Portable/Embeddable Bootstrap Script for Windows Powershell
# Downloads Python embeddable zip, configures it, installs pip, and verifies.

$ErrorActionPreference = "Stop"

$workspace = "C:\Users\USER\.gemini\antigravity\scratch"
$envDir = Join-Path $workspace "python_env"
$zipPath = Join-Path $workspace "python-3.11.9-embed-amd64.zip"
$pythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
$getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$getPipPath = Join-Path $workspace "get-pip.py"

Write-Host "Creating environment directory: $envDir"
if (-not (Test-Path $envDir)) {
    New-Item -ItemType Directory -Path $envDir | Out-Null
}

# 1. Download Python Embeddable Zip
if (-not (Test-Path $zipPath)) {
    Write-Host "Downloading Python 3.11.9 embeddable zip..."
    Invoke-WebRequest -Uri $pythonUrl -OutFile $zipPath -UserAgent "Mozilla/5.0"
} else {
    Write-Host "Python zip already downloaded."
}

# 2. Extract Zip
Write-Host "Extracting Python zip to $envDir..."
Expand-Archive -Path $zipPath -DestinationPath $envDir -Force

# 3. Enable site-packages by modifying python311._pth
$pthFile = Join-Path $envDir "python311._pth"
Write-Host "Configuring $pthFile..."
if (Test-Path $pthFile) {
    # We must uncomment 'import site' so that pip and site-packages work
    $content = Get-Content $pthFile
    $newContent = @()
    foreach ($line in $content) {
        if ($line -eq "#import site") {
            $newContent += "import site"
        } else {
            $newContent += $line
        }
    }
    # Add site-packages path explicitly just in case
    if ($newContent -notcontains "Lib\site-packages") {
        $newContent += "Lib\site-packages"
    }
    $newContent | Set-Content $pthFile
}

# 4. Download get-pip.py
if (-not (Test-Path $getPipPath)) {
    Write-Host "Downloading get-pip.py..."
    Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath -UserAgent "Mozilla/5.0"
}

# 5. Run get-pip.py using the embedded python
Write-Host "Installing pip..."
$pythonExe = Join-Path $envDir "python.exe"
& $pythonExe $getPipPath

# 6. Verify Python and Pip installation
Write-Host "Verifying installations..."
$pyVer = & $pythonExe --version
$pipVer = & $pythonExe -m pip --version

Write-Host "Successfully installed Python and Pip!"
Write-Host "Python Version: $pyVer"
Write-Host "Pip Version: $pipVer"
