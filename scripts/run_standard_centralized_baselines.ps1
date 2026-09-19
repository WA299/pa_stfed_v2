param([string]$Device='cuda')
$ErrorActionPreference='Stop'
$grids=@('39_bus_semi_urban_reference_grid','50_bus_rural_reference_grid','56_bus_semi_urban_reference_grid','80_bus_rural_reference_grid')
foreach($g in $grids){ Write-Host "ASTGCN / $g"; python scripts/run_astgcn.py --grid $g --device $Device; if($LASTEXITCODE){ throw "ASTGCN failed: $g" } }
foreach($g in $grids){ Write-Host "GraphWaveNet / $g"; python scripts/run_graph_wavenet.py --grid $g --device $Device; if($LASTEXITCODE){ throw "GraphWaveNet failed: $g" } }
