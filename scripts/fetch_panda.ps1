# Fetches the Franka Panda model from MuJoCo Menagerie at a PINNED commit (reproducibility).
# Usage (from the repo root):  .\scripts\fetch_panda.ps1
$ErrorActionPreference = "Stop"
$sha  = "370c43965323c0944b433b202aac5058b5f907a9"
$url  = "https://github.com/google-deepmind/mujoco_menagerie.git"
$tmp  = Join-Path $env:TEMP "menagerie_$sha"
$dest = Join-Path (Get-Location) "assets/robots"

if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Force $tmp | Out-Null
git -C $tmp init -q
git -C $tmp remote add origin $url
git -C $tmp fetch --depth 1 origin $sha
git -C $tmp checkout -q FETCH_HEAD

New-Item -ItemType Directory -Force $dest | Out-Null
if (Test-Path "$dest/franka_emika_panda") { Remove-Item -Recurse -Force "$dest/franka_emika_panda" }
Copy-Item -Recurse "$tmp/franka_emika_panda" "$dest/franka_emika_panda"
Remove-Item -Recurse -Force $tmp
Write-Host "Panda model fetched at commit $sha -> assets/robots/franka_emika_panda"
