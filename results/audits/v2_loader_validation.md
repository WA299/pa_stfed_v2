# V2 Canonical LV Loader Validation

Generated at (UTC): `2026-09-19T09:52:39.707626+00:00`

No model training or test metrics are performed.

| Grid | Nodes | Edges | Timestamps | P/Q aligned | Load buses | Distances valid | All checks |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| 39_bus_semi_urban_reference_grid | 39 | 38 | 8760 | True | 28 | True | True |
| 50_bus_rural_reference_grid | 50 | 49 | 8760 | True | 21 | True | True |
| 56_bus_semi_urban_reference_grid | 56 | 55 | 8760 | True | 44 | True | True |
| 80_bus_rural_reference_grid | 80 | 79 | 8760 | True | 32 | True | True |

## Shapes and splits

### 39_bus_semi_urban_reference_grid

Dynamic shape `(time, node, feature)`: `[8760, 39, 7]`; static shape `(node, feature)`: `[39, 4]`; edge feature shape `(edge, feature)`: `[38, 4]`.
Split sample counts: `{'train': 6132, 'validation': 1314, 'test': 1314}`.

| Distance matrix | Shape | Symmetric | Diagonal zero |
| --- | --- | --- | --- |
| `hop_distance` | `[39, 39]` | True | True |
| `cumulative_r_distance` | `[39, 39]` | True | True |
| `cumulative_x_distance` | `[39, 39]` | True | True |
| `impedance_abs_distance` | `[39, 39]` | True | True |
| `physical_line_length_distance` | `[39, 39]` | True | True |

### 50_bus_rural_reference_grid

Dynamic shape `(time, node, feature)`: `[8760, 50, 7]`; static shape `(node, feature)`: `[50, 4]`; edge feature shape `(edge, feature)`: `[49, 4]`.
Split sample counts: `{'train': 6132, 'validation': 1314, 'test': 1314}`.

| Distance matrix | Shape | Symmetric | Diagonal zero |
| --- | --- | --- | --- |
| `hop_distance` | `[50, 50]` | True | True |
| `cumulative_r_distance` | `[50, 50]` | True | True |
| `cumulative_x_distance` | `[50, 50]` | True | True |
| `impedance_abs_distance` | `[50, 50]` | True | True |
| `physical_line_length_distance` | `[50, 50]` | True | True |

### 56_bus_semi_urban_reference_grid

Dynamic shape `(time, node, feature)`: `[8760, 56, 7]`; static shape `(node, feature)`: `[56, 4]`; edge feature shape `(edge, feature)`: `[55, 4]`.
Split sample counts: `{'train': 6132, 'validation': 1314, 'test': 1314}`.

| Distance matrix | Shape | Symmetric | Diagonal zero |
| --- | --- | --- | --- |
| `hop_distance` | `[56, 56]` | True | True |
| `cumulative_r_distance` | `[56, 56]` | True | True |
| `cumulative_x_distance` | `[56, 56]` | True | True |
| `impedance_abs_distance` | `[56, 56]` | True | True |
| `physical_line_length_distance` | `[56, 56]` | True | True |

### 80_bus_rural_reference_grid

Dynamic shape `(time, node, feature)`: `[8760, 80, 7]`; static shape `(node, feature)`: `[80, 4]`; edge feature shape `(edge, feature)`: `[79, 4]`.
Split sample counts: `{'train': 6132, 'validation': 1314, 'test': 1314}`.

| Distance matrix | Shape | Symmetric | Diagonal zero |
| --- | --- | --- | --- |
| `hop_distance` | `[80, 80]` | True | True |
| `cumulative_r_distance` | `[80, 80]` | True | True |
| `cumulative_x_distance` | `[80, 80]` | True | True |
| `impedance_abs_distance` | `[80, 80]` | True | True |
| `physical_line_length_distance` | `[80, 80]` | True | True |

All checks pass: **True**.